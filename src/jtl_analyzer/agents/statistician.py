"""Statistician agent: compute a StatsReport from a NormalizedDataset."""

import logging

from jtl_analyzer.core.models import FeatureStats, NormalizedDataset, StatsReport

logger = logging.getLogger(__name__)


def run(dataset: NormalizedDataset) -> StatsReport:
    """Compute global and per-feature statistics from a normalized dataset.

    Pure pandas — no LLM calls. Groups by the ``label`` column to produce
    one ``FeatureStats`` entry per unique endpoint/feature name.

    Args:
        dataset: The loaded, cleaned JTL dataset.

    Returns:
        A self-contained ``StatsReport`` with global metrics and per-feature
        breakdowns.
    """
    df = dataset.data
    n = len(df)

    global_error_rate = float((~df["success"]).sum() / n) if n > 0 else 0.0

    per_feature: list[FeatureStats] = []
    for label, group in df.groupby("label", sort=True):
        count = len(group)
        per_feature.append(
            FeatureStats(
                name=str(label),
                count=count,
                mean_ms=float(group["elapsed"].mean()),
                min_ms=float(group["elapsed"].min()),
                max_ms=float(group["elapsed"].max()),
                error_rate=float((~group["success"]).sum() / count) if count > 0 else 0.0,
            )
        )

    logger.debug("Stats: %d samples, %d features, %.1f%% errors", n, len(per_feature), global_error_rate * 100)

    return StatsReport(
        dataset_metadata=dataset.metadata,
        total_count=n,
        global_mean_ms=float(df["elapsed"].mean()),
        global_min_ms=float(df["elapsed"].min()),
        global_max_ms=float(df["elapsed"].max()),
        global_error_rate=global_error_rate,
        per_feature=tuple(per_feature),
    )
