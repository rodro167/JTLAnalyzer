"""FastAPI application exposing the JTL Analyzer specialist agents over HTTP.

This module defines the ASGI app served by::

    uvicorn jtl_analyzer.api.main:app --host 0.0.0.0 --port 8000

or equivalently by ``python -m jtl_analyzer.cli serve``.

Architecture note: the endpoints call the specialist agents directly and never
invoke an LLM. The orchestrator's LLM-generated plan is invariant in practice
(loader, then the four specialists in parallel), so planning per HTTP request
would add latency, cost, and non-determinism for nothing. The orchestrator
remains the CLI's execution path; both paths share the same tested agents.
"""

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from jtl_analyzer.agents import anomalies as anomalies_module
from jtl_analyzer.agents import errors as errors_module
from jtl_analyzer.agents import statistician as statistician_module
from jtl_analyzer.agents import trends as trends_module
from jtl_analyzer.api.dependencies import LoadedDataset, get_api_config
from jtl_analyzer.api.models import ErrorResponse, FileTooLargeError, HealthResponse
from jtl_analyzer.api.serialization import (
    analysis_result_to_dict,
    anomalies_report_to_dict,
    errors_report_to_dict,
    stats_report_to_dict,
    trends_report_to_dict,
)
from jtl_analyzer.core.exceptions import InvalidJTLError
from jtl_analyzer.core.models import AnalysisResult
from jtl_analyzer.i18n import get_message

logger = logging.getLogger(__name__)


def _package_version() -> str:
    """Return the installed ``jtl-analyzer`` version, or ``"unknown"``."""
    try:
        return version("jtl-analyzer")
    except PackageNotFoundError:  # pragma: no cover - only when not installed
        return "unknown"


API_VERSION = _package_version()

# Written as a literal rather than via `status.*`: Starlette renamed this
# constant (HTTP_413_REQUEST_ENTITY_TOO_LARGE -> HTTP_413_CONTENT_TOO_LARGE) and
# deprecated the old name, so either spelling breaks on some version permitted
# by our `fastapi>=0.109` floor. The status code itself is stable.
_HTTP_413 = 413

# Reusable OpenAPI response documentation for the analysis endpoints.
_ANALYSIS_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "The upload is not a valid JTL file."},
    413: {"model": ErrorResponse, "description": "The upload exceeds the configured size limit."},
    422: {"description": "The `file` field is missing or `warmup_seconds` is not a number."},
    500: {"model": ErrorResponse, "description": "Unexpected server-side failure."},
}

app = FastAPI(
    title="JTL Analyzer API",
    version=API_VERSION,
    summary="Analyze JMeter .jtl result files and return structured performance metrics.",
    description=(
        "Runs JTL files through the JTL Analyzer specialist agents and returns their "
        "reports as JSON.\n\n"
        "`POST /analyze` runs all four specialists and returns the combined result. The "
        "`/analyze/{agent}` endpoints run a single specialist when only part of the "
        "analysis is needed.\n\n"
        "Every endpoint is deterministic: identical input yields identical output, and no "
        "language model is involved in serving a request.\n\n"
        "Errors use a uniform body — `{\"error\": \"<code>\", \"message\": \"<detail>\"}` — "
        "where `error` is a stable code safe to branch on. The exception is HTTP 422, which "
        "uses FastAPI's own request-validation format."
    ),
)


# ---------------------------------------------------------------------------
# Upload size enforcement (layer 1 of 2)
#
# Rejects oversized uploads from the Content-Length header, before the body is
# read. That header is absent under chunked transfer encoding, so
# dependencies._spool_to_temp_file independently counts bytes as it streams;
# without that second layer the cap would be bypassable by any chunking client.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def enforce_max_upload_size(request: Request, call_next: Any) -> Any:
    """Reject requests whose declared ``Content-Length`` exceeds the upload limit.

    Args:
        request: The incoming request.
        call_next: The next handler in the middleware chain.

    Returns:
        A 413 ``JSONResponse`` when the declared length is over the limit,
        otherwise the downstream response.
    """
    config = get_api_config()
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            declared = 0
        if declared > config.max_upload_bytes:
            logger.warning(
                "Rejected upload of %d bytes (limit %d)", declared, config.max_upload_bytes
            )
            return _error_response(
                _HTTP_413,
                "file_too_large",
                get_message("ERROR_FILE_TOO_LARGE", max_mb=config.max_upload_mb),
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Exception handlers: domain/transport exception -> HTTP status
# ---------------------------------------------------------------------------

def _error_response(status_code: int, error: str, message: str) -> JSONResponse:
    """Build a ``JSONResponse`` matching the ``ErrorResponse`` schema."""
    return JSONResponse(status_code=status_code, content={"error": error, "message": message})


@app.exception_handler(InvalidJTLError)
async def handle_invalid_jtl(request: Request, exc: InvalidJTLError) -> JSONResponse:
    """Map ``InvalidJTLError`` to HTTP 400.

    Covers every rejection the loader can produce: an unparseable file, missing
    required columns, an XML body with no sample elements, and a
    ``warmup_seconds`` value that excludes all samples. These are one exception
    type and so share the ``invalid_jtl`` code; the ``message`` field, already
    localized by the loader through the i18n catalog, distinguishes them.
    """
    logger.info("Rejected invalid JTL upload: %s", exc)
    return _error_response(status.HTTP_400_BAD_REQUEST, "invalid_jtl", str(exc))


@app.exception_handler(FileTooLargeError)
async def handle_file_too_large(request: Request, exc: FileTooLargeError) -> JSONResponse:
    """Map ``FileTooLargeError`` to HTTP 413.

    Raised by the spooling byte counter, which catches uploads that omit or
    understate ``Content-Length``.
    """
    logger.warning("Upload exceeded size limit of %s MB", exc.max_upload_mb)
    return _error_response(
        _HTTP_413,
        "file_too_large",
        get_message("ERROR_FILE_TOO_LARGE", max_mb=exc.max_upload_mb),
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Map any unhandled exception to HTTP 500 without leaking internals.

    The traceback is written to the server log via ``logger.exception``; the
    response body carries only a fixed generic message, so no stack trace,
    file path, or exception text reaches the client.
    """
    logger.exception("Unhandled error while serving %s %s", request.method, request.url.path)
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        get_message("ERROR_INTERNAL"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Liveness check",
)
async def health() -> HealthResponse:
    """Report that the service is alive.

    Takes no input and touches neither the filesystem nor any external
    dependency, so it is safe to poll frequently from a load balancer or
    container orchestrator.

    Returns:
        ``{"status": "ok", "version": "<package version>"}`` with HTTP 200.
    """
    return HealthResponse(status="ok", version=API_VERSION)


@app.post(
    "/analyze",
    tags=["analysis"],
    summary="Run the full analysis",
    responses=_ANALYSIS_RESPONSES,
)
async def analyze(dataset: LoadedDataset) -> dict[str, Any]:
    """Run all four specialist agents over an uploaded JTL file.

    **Request** — ``multipart/form-data``:

    - ``file`` (required): the JTL file, CSV or XML format. Format is detected
      from the file's first bytes, not its extension.
    - ``warmup_seconds`` (optional, default ``0.0``): seconds to exclude from
      the start of the run, for discarding JVM/connection-pool warmup effects.

    **Response** — HTTP 200 with four top-level keys, one per agent:

    - ``stats``: global and per-feature mean, p50/p90/p95/p99, standard
      deviation, max, throughput, and error rate.
    - ``errors``: per-feature response code distribution.
    - ``anomalies``: per-feature upper-tail response-time outliers (IQR method).
    - ``trends``: per-feature temporal degradation windows (1-minute bins).

    Each report embeds the same ``dataset_metadata``, so it remains
    self-contained if split from the response. Timestamps are ISO 8601 strings.
    The parsed sample rows themselves are not returned — only aggregates.

    **Errors**: 400 ``invalid_jtl`` if the file cannot be parsed, is missing
    required columns, or ``warmup_seconds`` excludes every sample; 413
    ``file_too_large`` if the upload exceeds ``API_MAX_UPLOAD_MB``; 422 if
    ``file`` is absent; 500 ``internal_error`` otherwise.
    """
    result = AnalysisResult(
        stats=statistician_module.run(dataset),
        errors=errors_module.run(dataset),
        anomalies=anomalies_module.run(dataset),
        trends=trends_module.run(dataset),
    )
    return analysis_result_to_dict(result)


@app.post(
    "/analyze/statistician",
    tags=["analysis"],
    summary="Run only the statistician agent",
    responses=_ANALYSIS_RESPONSES,
)
async def analyze_statistician(dataset: LoadedDataset) -> dict[str, Any]:
    """Compute performance statistics for an uploaded JTL file.

    Same ``multipart/form-data`` request as ``POST /analyze`` (``file`` required,
    ``warmup_seconds`` optional).

    **Response** — HTTP 200 with the ``StatsReport`` alone, unwrapped: global
    metrics (``global_mean_ms``, ``global_p50_ms`` through ``global_p99_ms``,
    ``global_std_ms``, ``global_max_ms``, ``global_throughput``,
    ``global_error_rate``), a ``per_feature`` array of the same metrics per
    label, and ``dataset_metadata``. Global throughput uses the full run
    duration; per-feature throughput uses each feature's own active window.

    **Errors**: identical to ``POST /analyze``.
    """
    return stats_report_to_dict(statistician_module.run(dataset))


@app.post(
    "/analyze/errors",
    tags=["analysis"],
    summary="Run only the errors agent",
    responses=_ANALYSIS_RESPONSES,
)
async def analyze_errors(dataset: LoadedDataset) -> dict[str, Any]:
    """Compute the per-feature response code distribution for an uploaded JTL file.

    Same ``multipart/form-data`` request as ``POST /analyze`` (``file`` required,
    ``warmup_seconds`` optional).

    **Response** — HTTP 200 with the ``ErrorsReport`` alone, unwrapped:
    ``codes_by_feature`` maps each feature label to an array of
    ``{code, count, percentage}`` entries sorted by count descending, plus
    ``dataset_metadata``. Codes are strings, since JMeter also reports
    non-numeric values such as ``"Non HTTP response code: ..."``. Successful
    codes are included, not just failures — the report is a full distribution.

    **Errors**: identical to ``POST /analyze``.
    """
    return errors_report_to_dict(errors_module.run(dataset))


@app.post(
    "/analyze/anomalies",
    tags=["analysis"],
    summary="Run only the anomalies agent",
    responses=_ANALYSIS_RESPONSES,
)
async def analyze_anomalies(dataset: LoadedDataset) -> dict[str, Any]:
    """Detect response-time outliers per feature for an uploaded JTL file.

    Same ``multipart/form-data`` request as ``POST /analyze`` (``file`` required,
    ``warmup_seconds`` optional).

    Uses the IQR method on the upper tail only: a sample is anomalous when its
    elapsed time exceeds ``Q3 + 1.5 * IQR`` for its feature.

    **Response** — HTTP 200 with the ``AnomaliesReport`` alone, unwrapped:
    ``by_feature`` maps each label to its ``threshold_ms``, ``anomaly_count``,
    ``anomaly_rate``, and a ``samples`` array sorted by ``deviation_factor``
    descending, plus ``dataset_metadata``. Features with fewer than 20 samples,
    or a degenerate zero IQR, are returned with ``insufficient_data: true`` and
    no samples rather than being omitted.

    **Errors**: identical to ``POST /analyze``.
    """
    return anomalies_report_to_dict(anomalies_module.run(dataset))


@app.post(
    "/analyze/trends",
    tags=["analysis"],
    summary="Run only the trends agent",
    responses=_ANALYSIS_RESPONSES,
)
async def analyze_trends(dataset: LoadedDataset) -> dict[str, Any]:
    """Detect temporal degradation windows per feature for an uploaded JTL file.

    Same ``multipart/form-data`` request as ``POST /analyze`` (``file`` required,
    ``warmup_seconds`` optional).

    Samples are grouped into 1-minute bins. The reference baseline is the median
    of per-bin medians, weighting each minute equally regardless of its traffic
    volume. Bins reaching 2x that reference are flagged, and consecutive flagged
    bins are merged into a single window.

    **Response** — HTTP 200 with the ``TrendsReport`` alone, unwrapped:
    ``by_feature`` maps each label to a chronological ``windows`` array of
    ``{start_time, end_time, duration_seconds, window_median_ms,
    reference_median_ms, degradation_factor}``, plus ``dataset_metadata``.
    Features with fewer than 20 samples, no non-empty bins, or a zero reference
    median are returned with ``insufficient_data: true``. A feature that was
    analyzed but showed no degradation has an empty ``windows`` array and
    ``insufficient_data: false``.

    **Errors**: identical to ``POST /analyze``.
    """
    return trends_report_to_dict(trends_module.run(dataset))
