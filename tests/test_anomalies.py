"""Unit and integration tests for agents/anomalies.py."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from jtl_analyzer.agents import anomalies as anomalies_agent
from jtl_analyzer.agents import loader as loader_agent
from jtl_analyzer.core.models import (
    AnomaliesReport,
    AnomalousSample,
    DatasetMetadata,
    FeatureAnomalies,
    NormalizedDataset,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> NormalizedDataset:
    return loader_agent.run(str(FIXTURES / name))


def _make_dataset(rows: list[dict]) -> NormalizedDataset:
    """Build a NormalizedDataset in memory from a list of row dicts."""
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["elapsed"] = df["elapsed"].astype(float)
    df["success"] = df["success"].map({"true": True, "false": False})
    start = df["timestamp"].min().to_pydatetime()
    end = df["timestamp"].max().to_pydatetime()
    metadata = DatasetMetadata(
        file_path="<in-memory>",
        row_count=len(df),
        start_time=start,
        end_time=end,
        duration_seconds=(end - start).total_seconds(),
    )
    return NormalizedDataset(metadata=metadata, data=df)


def _rows(label: str, elapsed_values: list[int | float], base_ts: int = 1700000000000) -> list[dict]:
    """Build row dicts for a single feature with sequential timestamps."""
    return [
        {
            "timestamp": base_ts + i * 1000,
            "elapsed": e,
            "label": label,
            "responseCode": "200",
            "success": "true",
        }
        for i, e in enumerate(elapsed_values)
    ]


class TestBasicDetection:
    """23 normal samples (100–122ms) + 2 outliers (1000ms, 2000ms) = 25 total.

    Q1=106, Q3=118, IQR=12, threshold=136. Both outliers exceed it.
    """

    def _dataset(self) -> NormalizedDataset:
        normal = list(range(100, 123))          # 23 values: 100..122
        outliers = [1000, 2000]
        return _make_dataset(_rows("Feature", normal + outliers))

    def test_feature_present(self):
        report = anomalies_agent.run(self._dataset())
        assert "Feature" in report.by_feature

    def test_anomaly_count(self):
        report = anomalies_agent.run(self._dataset())
        assert report.by_feature["Feature"].anomaly_count == 2

    def test_both_outliers_in_samples(self):
        report = anomalies_agent.run(self._dataset())
        elapsed_values = {s.elapsed_ms for s in report.by_feature["Feature"].samples}
        assert 1000.0 in elapsed_values
        assert 2000.0 in elapsed_values

    def test_sorted_by_deviation_descending(self):
        report = anomalies_agent.run(self._dataset())
        samples = report.by_feature["Feature"].samples
        assert samples[0].elapsed_ms == 2000.0
        assert samples[0].deviation_factor > samples[1].deviation_factor

    def test_insufficient_data_false(self):
        report = anomalies_agent.run(self._dataset())
        assert report.by_feature["Feature"].insufficient_data is False

    def test_threshold_positive(self):
        report = anomalies_agent.run(self._dataset())
        assert report.by_feature["Feature"].threshold_ms > 0


class TestInsufficientData:
    def test_too_few_samples_flags_insufficient(self):
        # 15 samples — below MIN_SAMPLES_FOR_DETECTION (20)
        dataset = _make_dataset(_rows("Feature", list(range(100, 115))))
        report = anomalies_agent.run(dataset)
        fa = report.by_feature["Feature"]
        assert fa.insufficient_data is True

    def test_too_few_samples_has_zero_anomalies(self):
        dataset = _make_dataset(_rows("Feature", list(range(100, 115))))
        report = anomalies_agent.run(dataset)
        assert report.by_feature["Feature"].anomaly_count == 0

    def test_too_few_samples_has_empty_samples_tuple(self):
        dataset = _make_dataset(_rows("Feature", list(range(100, 115))))
        report = anomalies_agent.run(dataset)
        assert report.by_feature["Feature"].samples == ()

    def test_iqr_zero_flags_insufficient(self):
        # 25 identical samples — IQR == 0, detection must be skipped
        dataset = _make_dataset(_rows("Feature", [150] * 25))
        report = anomalies_agent.run(dataset)
        assert report.by_feature["Feature"].insufficient_data is True

    def test_iqr_zero_has_zero_anomalies(self):
        dataset = _make_dataset(_rows("Feature", [150] * 25))
        report = anomalies_agent.run(dataset)
        assert report.by_feature["Feature"].anomaly_count == 0


class TestNoAnomalies:
    """30 uniform samples 100–129ms. threshold ≈ 143.5ms > max(129ms) → no anomalies."""

    def _dataset(self) -> NormalizedDataset:
        return _make_dataset(_rows("Feature", list(range(100, 130))))

    def test_insufficient_data_false(self):
        report = anomalies_agent.run(self._dataset())
        assert report.by_feature["Feature"].insufficient_data is False

    def test_no_anomalies_detected(self):
        report = anomalies_agent.run(self._dataset())
        assert report.by_feature["Feature"].anomaly_count == 0

    def test_threshold_was_computed(self):
        report = anomalies_agent.run(self._dataset())
        assert report.by_feature["Feature"].threshold_ms > 0

    def test_samples_empty(self):
        report = anomalies_agent.run(self._dataset())
        assert report.by_feature["Feature"].samples == ()


class TestPerFeatureIsolation:
    """Feature A: 100–129ms (30 samples). Feature B: 1000–1029ms (30 samples).

    Each feature's threshold must be derived from its own distribution only.
    A's fast values must not be flagged because of B's slow baseline.
    """

    def _dataset(self) -> NormalizedDataset:
        rows_a = _rows("A", list(range(100, 130)), base_ts=1700000000000)
        rows_b = _rows("B", list(range(1000, 1030)), base_ts=1700000100000)
        return _make_dataset(rows_a + rows_b)

    def test_both_features_present(self):
        report = anomalies_agent.run(self._dataset())
        assert "A" in report.by_feature
        assert "B" in report.by_feature

    def test_feature_a_has_no_anomalies(self):
        report = anomalies_agent.run(self._dataset())
        assert report.by_feature["A"].anomaly_count == 0

    def test_feature_b_has_no_anomalies(self):
        report = anomalies_agent.run(self._dataset())
        assert report.by_feature["B"].anomaly_count == 0

    def test_thresholds_are_independent(self):
        report = anomalies_agent.run(self._dataset())
        # A's threshold must be far below B's typical values
        assert report.by_feature["A"].threshold_ms < 500
        # B's threshold must be far above A's typical values
        assert report.by_feature["B"].threshold_ms > 500


class TestReportStructure:
    def test_anomalies_report_is_frozen(self):
        report = anomalies_agent.run(_load("multi_feature.jtl"))
        with pytest.raises(FrozenInstanceError):
            report.dataset_metadata = None  # type: ignore

    def test_feature_anomalies_is_frozen(self):
        dataset = _make_dataset(_rows("F", list(range(100, 130))))
        report = anomalies_agent.run(dataset)
        fa = report.by_feature["F"]
        with pytest.raises(FrozenInstanceError):
            fa.anomaly_count = 99  # type: ignore

    def test_anomalous_sample_is_frozen(self):
        normal = list(range(100, 123))
        dataset = _make_dataset(_rows("F", normal + [1000, 2000]))
        report = anomalies_agent.run(dataset)
        sample = report.by_feature["F"].samples[0]
        with pytest.raises(FrozenInstanceError):
            sample.elapsed_ms = 0.0  # type: ignore

    def test_by_feature_is_dict(self):
        report = anomalies_agent.run(_load("multi_feature.jtl"))
        assert isinstance(report.by_feature, dict)

    def test_samples_is_tuple(self):
        dataset = _make_dataset(_rows("F", list(range(100, 130))))
        report = anomalies_agent.run(dataset)
        assert isinstance(report.by_feature["F"].samples, tuple)

    def test_return_type(self):
        report = anomalies_agent.run(_load("multi_feature.jtl"))
        assert isinstance(report, AnomaliesReport)


class TestRealisticFixture:
    """multi_feature.jtl has only 3 samples per feature — all should be insufficient_data."""

    def test_search_insufficient(self):
        report = anomalies_agent.run(_load("multi_feature.jtl"))
        assert report.by_feature["Search"].insufficient_data is True

    def test_checkout_insufficient(self):
        report = anomalies_agent.run(_load("multi_feature.jtl"))
        assert report.by_feature["Checkout"].insufficient_data is True


class TestRealisticLargeFixture:
    """20260516.jtl — real-world data. 'Get Movies By Year' has 2386 samples,
    max=7437ms vs p95=449ms, guaranteeing anomalies."""

    def test_get_movies_by_year_present(self):
        report = anomalies_agent.run(_load("20260516.jtl"))
        assert "Get Movies By Year" in report.by_feature

    def test_get_movies_by_year_not_insufficient(self):
        report = anomalies_agent.run(_load("20260516.jtl"))
        assert report.by_feature["Get Movies By Year"].insufficient_data is False

    def test_get_movies_by_year_has_anomalies(self):
        report = anomalies_agent.run(_load("20260516.jtl"))
        assert report.by_feature["Get Movies By Year"].anomaly_count > 0

    def test_get_movies_by_year_worst_sample_is_7437(self):
        report = anomalies_agent.run(_load("20260516.jtl"))
        worst = report.by_feature["Get Movies By Year"].samples[0]
        assert worst.elapsed_ms == pytest.approx(7437.0)
