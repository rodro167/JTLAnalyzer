"""Anomalies agent: detect outlier samples per feature using the IQR method."""

import logging

from jtl_analyzer.core.models import (
    AnomaliesReport,
    AnomalousSample,
    FeatureAnomalies,
    NormalizedDataset,
)

logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_DETECTION = 20

_SKIPPED = dict(anomaly_count=0, anomaly_rate=0.0, threshold_ms=0.0, samples=(), insufficient_data=True)


def run(dataset: NormalizedDataset) -> AnomaliesReport:
    """Detect upper-tail response-time outliers per feature using the IQR method.

    For each feature with at least MIN_SAMPLES_FOR_DETECTION samples, computes
    Q1, Q3, IQR, and threshold = Q3 + 1.5 * IQR. Samples with elapsed > threshold
    are flagged as anomalous. Only the upper tail is considered.

    Features with fewer than MIN_SAMPLES_FOR_DETECTION samples, or with IQR == 0
    (degenerate distribution where the threshold would equal Q3, producing false
    positives for any value marginally above it), are marked insufficient_data=True
    and skipped.

    Pure pandas — no LLM calls.

    Args:
        dataset: The loaded, cleaned JTL dataset.

    Returns:
        An ``AnomaliesReport`` with per-feature detection results.
    """
    df = dataset.data
    by_feature: dict[str, FeatureAnomalies] = {}

    for label, group in df.groupby("label", sort=True):
        feature_name = str(label)
        total = len(group)

        if total < MIN_SAMPLES_FOR_DETECTION:
            by_feature[feature_name] = FeatureAnomalies(feature=feature_name, total_samples=total, **_SKIPPED)
            continue

        q = group["elapsed"].quantile([0.25, 0.75])
        q1, q3 = float(q[0.25]), float(q[0.75])
        iqr = q3 - q1

        # IQR == 0 means all values cluster at the same point (degenerate distribution).
        # The threshold would equal Q3, flagging any value infinitesimally above it as
        # an anomaly with a near-1.0 deviation factor — these are meaningless false positives.
        if iqr == 0.0:
            by_feature[feature_name] = FeatureAnomalies(feature=feature_name, total_samples=total, **_SKIPPED)
            continue

        threshold = q3 + 1.5 * iqr
        outlier_rows = group[group["elapsed"] > threshold]

        samples: list[AnomalousSample] = [
            AnomalousSample(
                timestamp=row["timestamp"].to_pydatetime(),
                elapsed_ms=float(row["elapsed"]),
                feature=feature_name,
                response_code=str(row["responseCode"]),
                deviation_factor=float(row["elapsed"]) / threshold,
            )
            for _, row in outlier_rows.iterrows()
        ]
        samples.sort(key=lambda s: s.deviation_factor, reverse=True)

        anomaly_count = len(samples)
        by_feature[feature_name] = FeatureAnomalies(
            feature=feature_name,
            total_samples=total,
            anomaly_count=anomaly_count,
            anomaly_rate=float(anomaly_count) / total,
            threshold_ms=threshold,
            samples=tuple(samples),
            insufficient_data=False,
        )

    logger.debug("Anomalies: %d features processed", len(by_feature))
    return AnomaliesReport(dataset_metadata=dataset.metadata, by_feature=by_feature)
