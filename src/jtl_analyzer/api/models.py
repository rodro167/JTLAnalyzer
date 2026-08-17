"""HTTP-specific request/response models and transport-level exceptions.

These are distinct from the domain models in ``core.models``. Pydantic is used
here (rather than frozen dataclasses) because these types define the API's
external contract and feed FastAPI's OpenAPI schema generation.

The analysis endpoints intentionally have no Pydantic response model. Mirroring
the ten nested report dataclasses would duplicate ``core.models`` and drift from
it on every field change; ``core.models`` stays the single source of truth, and
``api.serialization`` is the boundary that renders it as JSON.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body for ``GET /health``.

    Attributes:
        status: Always ``"ok"`` when the service is able to answer.
        version: Installed version of the ``jtl-analyzer`` package.
    """

    status: str = Field(examples=["ok"])
    version: str = Field(examples=["0.1.0"])


class ErrorResponse(BaseModel):
    """Uniform error body returned by every non-2xx response this API raises.

    FastAPI's own 422 validation failures use its default body shape instead.

    Attributes:
        error: Stable machine-readable code (``"invalid_jtl"``,
            ``"file_too_large"``, or ``"internal_error"``). Safe to branch on;
            unlike ``message``, it is not localized.
        message: Human-readable detail. Localized via the i18n catalog and
            never contains internal diagnostics such as stack traces.
    """

    error: str = Field(examples=["invalid_jtl"])
    message: str = Field(examples=["Missing required columns: elapsed, label"])


class FileTooLargeError(Exception):
    """Raised when an upload exceeds the configured size limit.

    Lives in the API layer rather than ``core.exceptions`` because it describes a
    transport constraint, not a property of the data: an oversized file is not an
    invalid JTL. Maps to HTTP 413.

    Args:
        max_upload_mb: The configured limit in megabytes, used to build the
            user-facing message.
    """

    def __init__(self, max_upload_mb: float) -> None:
        self.max_upload_mb = max_upload_mb
        super().__init__(f"Upload exceeds the maximum size of {max_upload_mb:g} MB")
