"""Trends agent: detect temporal degradation windows per feature."""

import logging

import numpy as np
import pandas as pd

from jtl_analyzer.core.models import (
    DegradationWindow,
    FeatureTrends,
    NormalizedDataset,
    TrendsReport,
)

logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_DETECTION = 20
BIN_SIZE_SECONDS = 60
DEGRADATION_FACTOR_THRESHOLD = 2.0
# Allowed slack when deciding whether two consecutive bin starts are exactly
# BIN_SIZE_SECONDS apart (accounts for sub-second floating-point drift in
# pd.Grouper bin boundaries).
BIN_GAP_TOLERANCE_SECONDS = 1.0


def run(dataset: NormalizedDataset) -> TrendsReport:
    """Detect temporal degradation windows per feature using 1-minute binning.

    For each feature, samples are grouped into 1-minute bins. A reference
    median is computed as the median of per-bin medians (giving equal weight
    to each minute regardless of traffic volume). Bins whose median is at least
    ``DEGRADATION_FACTOR_THRESHOLD`` times the reference are marked degraded.
    Consecutive degraded bins are merged into ``DegradationWindow`` objects.

    Features are skipped (``insufficient_data=True``) when:
    - the feature has fewer than ``MIN_SAMPLES_FOR_DETECTION`` samples,
    - no non-empty 1-minute bins exist after grouping, or
    - the reference median is 0 (degenerate distribution).

    Pure pandas — no LLM calls.

    Args:
        dataset: The loaded, cleaned JTL dataset.

    Returns:
        A ``TrendsReport`` with per-feature degradation window detection results.
    """
    df = dataset.data
    by_feature: dict[str, FeatureTrends] = {}

    for label, group in df.groupby("label", sort=True):
        feature_name = str(label)
        total = len(group)

        if total < MIN_SAMPLES_FOR_DETECTION:
            by_feature[feature_name] = FeatureTrends(
                feature=feature_name,
                total_samples=total,
                windows=(),
                insufficient_data=True,
            )
            continue

        # Collect non-empty 1-minute bin medians.
        bin_medians: dict = {}  # pd.Timestamp -> float
        for bin_start, bin_group in group.groupby(pd.Grouper(key="timestamp", freq="1min")):
            if len(bin_group) == 0:
                continue
            bin_medians[bin_start] = float(bin_group["elapsed"].median())

        if not bin_medians:
            by_feature[feature_name] = FeatureTrends(
                feature=feature_name,
                total_samples=total,
                windows=(),
                insufficient_data=True,
            )
            continue

        # Equal weight per minute, regardless of per-bin traffic volume.
        reference_median = float(np.median(list(bin_medians.values())))

        # Zero reference would cause meaningless degradation factors (division by zero).
        if reference_median == 0.0:
            by_feature[feature_name] = FeatureTrends(
                feature=feature_name,
                total_samples=total,
                windows=(),
                insufficient_data=True,
            )
            continue

        threshold = DEGRADATION_FACTOR_THRESHOLD * reference_median
        degraded_starts = sorted(
            ts for ts, med in bin_medians.items() if med >= threshold
        )

        if not degraded_starts:
            by_feature[feature_name] = FeatureTrends(
                feature=feature_name,
                total_samples=total,
                windows=(),
                insufficient_data=False,
            )
            continue

        # Group consecutive degraded bin starts into windows.
        window_groups: list[list] = []
        current: list = [degraded_starts[0]]
        for ts in degraded_starts[1:]:
            gap = (ts - current[-1]).total_seconds()
            if abs(gap - BIN_SIZE_SECONDS) <= BIN_GAP_TOLERANCE_SECONDS:
                current.append(ts)
            else:
                window_groups.append(current)
                current = [ts]
        window_groups.append(current)

        detected: list[DegradationWindow] = []
        for wg in window_groups:
            win_start = wg[0]
            win_end = wg[-1] + pd.Timedelta(seconds=BIN_SIZE_SECONDS)
            win_start_dt = win_start.to_pydatetime()
            win_end_dt = win_end.to_pydatetime()
            duration = (win_end_dt - win_start_dt).total_seconds()

            mask = (group["timestamp"] >= win_start) & (group["timestamp"] < win_end)
            window_elapsed = group.loc[mask, "elapsed"]
            window_median = float(window_elapsed.median()) if len(window_elapsed) > 0 else 0.0

            detected.append(
                DegradationWindow(
                    start_time=win_start_dt,
                    end_time=win_end_dt,
                    duration_seconds=duration,
                    window_median_ms=window_median,
                    reference_median_ms=reference_median,
                    degradation_factor=window_median / reference_median,
                )
            )

        by_feature[feature_name] = FeatureTrends(
            feature=feature_name,
            total_samples=total,
            windows=tuple(sorted(detected, key=lambda w: w.start_time)),
            insufficient_data=False,
        )

    logger.debug("Trends: %d features processed", len(by_feature))
    return TrendsReport(dataset_metadata=dataset.metadata, by_feature=by_feature)
