"""Statistician agent: compute a StatsReport from a NormalizedDataset."""

import logging

import numpy as np

from jtl_analyzer.core.models import FeatureStats, NormalizedDataset, StatsReport

logger = logging.getLogger(__name__)

_QUANTILES = [0.50, 0.90, 0.95, 0.99]


def _safe_throughput(count: int, duration_seconds: float) -> float:
    """Return count / duration_seconds, or 0.0 if duration is non-positive."""
    return float(count) / duration_seconds if duration_seconds > 0 else 0.0


def _feature_throughput(group, count: int) -> float:
    """Return per-feature throughput based on the feature's own time window.

    Computes count / (max_timestamp - min_timestamp) in seconds. Returns 0.0
    when the feature has only one sample or all samples share the same timestamp.
    """
    duration = (group["timestamp"].max() - group["timestamp"].min()).total_seconds()
    return float(count) / duration if duration > 0 else 0.0


def _safe_std(series) -> float:
    """Return std (ddof=1) of a Series, replacing NaN with 0.0."""
    value = series.std()
    return 0.0 if (value is None or np.isnan(value)) else float(value)


def run(dataset: NormalizedDataset) -> StatsReport:
    """Compute global and per-feature statistics from a normalized dataset.

    Pure pandas — no LLM calls. Groups by the ``label`` column to produce
    one ``FeatureStats`` entry per unique endpoint/feature name.

    Global throughput uses the total run duration from metadata. Per-feature
    throughput uses each feature's own active time window (max − min timestamp
    of that feature's samples).

    Args:
        dataset: The loaded, cleaned JTL dataset.

    Returns:
        A self-contained ``StatsReport`` with global metrics and per-feature
        breakdowns.
    """
    df = dataset.data
    n = len(df)
    duration = dataset.metadata.duration_seconds

    global_error_rate = float((~df["success"]).sum() / n) if n > 0 else 0.0

    q = df["elapsed"].quantile(_QUANTILES)

    per_feature: list[FeatureStats] = []
    for label, group in df.groupby("label", sort=True):
        count = len(group)
        fq = group["elapsed"].quantile(_QUANTILES)
        per_feature.append(
            FeatureStats(
                name=str(label),
                count=count,
                mean_ms=float(group["elapsed"].mean()),
                p50_ms=float(fq[0.50]),
                p90_ms=float(fq[0.90]),
                p95_ms=float(fq[0.95]),
                p99_ms=float(fq[0.99]),
                std_ms=_safe_std(group["elapsed"]),
                max_ms=float(group["elapsed"].max()),
                throughput=_feature_throughput(group, count),
                error_rate=float((~group["success"]).sum() / count) if count > 0 else 0.0,
            )
        )

    logger.debug(
        "Stats: %d samples, %d features, %.1f%% errors",
        n,
        len(per_feature),
        global_error_rate * 100,
    )

    return StatsReport(
        dataset_metadata=dataset.metadata,
        total_count=n,
        global_mean_ms=float(df["elapsed"].mean()),
        global_p50_ms=float(q[0.50]),
        global_p90_ms=float(q[0.90]),
        global_p95_ms=float(q[0.95]),
        global_p99_ms=float(q[0.99]),
        global_std_ms=_safe_std(df["elapsed"]),
        global_max_ms=float(df["elapsed"].max()),
        global_throughput=_safe_throughput(n, duration),
        global_error_rate=global_error_rate,
        per_feature=tuple(per_feature),
    )
