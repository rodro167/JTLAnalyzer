"""Unit tests for agents/statistician.py."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from jtl_analyzer.agents import loader, statistician
from jtl_analyzer.core.models import DatasetMetadata, NormalizedDataset, StatsReport

FIXTURES = Path(__file__).parent / "fixtures"


def _report(fixture_name: str) -> StatsReport:
    ds = loader.run(str(FIXTURES / fixture_name))
    return statistician.run(ds)


def _in_memory_dataset(
    elapsed: list[float],
    labels: list[str],
    duration_seconds: float = 5.0,
) -> NormalizedDataset:
    """Build a minimal NormalizedDataset without touching the filesystem."""
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([ts] * len(elapsed)),
            "elapsed": [float(e) for e in elapsed],
            "label": labels,
            "success": [True] * len(elapsed),
        }
    )
    meta = DatasetMetadata(
        file_path="fake.jtl",
        row_count=len(elapsed),
        start_time=ts,
        end_time=ts,
        duration_seconds=duration_seconds,
    )
    return NormalizedDataset(metadata=meta, data=df)


class TestGlobalStats:
    def test_small_clean_total_count(self):
        assert _report("small_clean.jtl").total_count == 6

    def test_small_clean_mean(self):
        assert _report("small_clean.jtl").global_mean_ms == pytest.approx(200.0)

    def test_small_clean_max(self):
        assert _report("small_clean.jtl").global_max_ms == pytest.approx(300.0)

    def test_small_clean_error_rate_zero(self):
        assert _report("small_clean.jtl").global_error_rate == pytest.approx(0.0)

    # Percentiles — sorted([100,200,300,100,200,300]) = [100,100,200,200,300,300]
    # q=0.5 → index 2.5 → (200+200)/2 = 200.0
    # q=0.9 → index 4.5 → (300+300)/2 = 300.0
    # q=0.95 → index 4.75 → 300.0   q=0.99 → index 4.95 → 300.0
    def test_small_clean_p50(self):
        assert _report("small_clean.jtl").global_p50_ms == pytest.approx(200.0)

    def test_small_clean_p90(self):
        assert _report("small_clean.jtl").global_p90_ms == pytest.approx(300.0)

    def test_small_clean_p95(self):
        assert _report("small_clean.jtl").global_p95_ms == pytest.approx(300.0)

    def test_small_clean_p99(self):
        assert _report("small_clean.jtl").global_p99_ms == pytest.approx(300.0)

    # std(ddof=1) of [100,200,300,100,200,300]: mean=200, SS=40000, var=8000, std≈89.4427
    def test_small_clean_std(self):
        assert _report("small_clean.jtl").global_std_ms == pytest.approx(89.4427, rel=1e-3)

    # throughput = 6 / 5.0 = 1.2
    def test_small_clean_throughput(self):
        assert _report("small_clean.jtl").global_throughput == pytest.approx(1.2, rel=1e-3)

    def test_with_errors_total_count(self):
        assert _report("with_errors.jtl").total_count == 6

    def test_with_errors_mean(self):
        # (100+200+300+150+250+350) / 6 = 225.0
        assert _report("with_errors.jtl").global_mean_ms == pytest.approx(225.0)

    def test_with_errors_error_rate(self):
        assert _report("with_errors.jtl").global_error_rate == pytest.approx(2 / 6)

    # sorted([100,200,300,150,250,350]) = [100,150,200,250,300,350]
    # p50 → index 2.5 → (200+250)/2 = 225.0
    # p90 → index 4.5 → (300+350)/2 = 325.0
    # p95 → index 4.75 → 300 + 0.75*(350-300) = 337.5
    # p99 → index 4.95 → 300 + 0.95*(350-300) = 347.5
    def test_with_errors_p50(self):
        assert _report("with_errors.jtl").global_p50_ms == pytest.approx(225.0)

    def test_with_errors_p90(self):
        assert _report("with_errors.jtl").global_p90_ms == pytest.approx(325.0)

    def test_with_errors_p95(self):
        assert _report("with_errors.jtl").global_p95_ms == pytest.approx(337.5)

    def test_with_errors_p99(self):
        assert _report("with_errors.jtl").global_p99_ms == pytest.approx(347.5)

    # std(ddof=1): mean=225, SS=43750, var=8750, std≈93.5414
    def test_with_errors_std(self):
        assert _report("with_errors.jtl").global_std_ms == pytest.approx(93.5414, rel=1e-3)

    # throughput = 6 / 5.0 = 1.2
    def test_with_errors_throughput(self):
        assert _report("with_errors.jtl").global_throughput == pytest.approx(1.2, rel=1e-3)

    def test_multi_feature_total_count(self):
        assert _report("multi_feature.jtl").total_count == 6

    def test_multi_feature_mean(self):
        # (50+60+70+500+600+700) / 6 = 330.0
        assert _report("multi_feature.jtl").global_mean_ms == pytest.approx(330.0)

    def test_multi_feature_max(self):
        assert _report("multi_feature.jtl").global_max_ms == pytest.approx(700.0)

    def test_multi_feature_error_rate(self):
        assert _report("multi_feature.jtl").global_error_rate == pytest.approx(1 / 6)


class TestPerFeatureStats:
    def test_small_clean_single_feature(self):
        report = _report("small_clean.jtl")
        assert len(report.per_feature) == 1
        assert report.per_feature[0].name == "Homepage"

    def test_multi_feature_count(self):
        report = _report("multi_feature.jtl")
        assert len(report.per_feature) == 2

    def test_multi_feature_search_stats(self):
        # Search elapsed: [50, 60, 70]; sorted same; duration=5s
        # p50 → index 1.0 → 60.0
        # p90 → index 1.8 → 60 + 0.8*(70-60) = 68.0
        # p95 → index 1.9 → 69.0
        # std(ddof=1): mean=60, SS=200, var=100, std=10.0
        # throughput = 3/5 = 0.6
        report = _report("multi_feature.jtl")
        by_name = {fs.name: fs for fs in report.per_feature}
        search = by_name["Search"]
        assert search.count == 3
        assert search.mean_ms == pytest.approx(60.0)
        assert search.p50_ms == pytest.approx(60.0)
        assert search.p90_ms == pytest.approx(68.0)
        assert search.p95_ms == pytest.approx(69.0)
        assert search.max_ms == pytest.approx(70.0)
        assert search.std_ms == pytest.approx(10.0)
        # feature_duration = (1700000002000 - 1700000000000) / 1000 = 2.0s → 3/2 = 1.5
        assert search.throughput == pytest.approx(1.5, rel=1e-3)
        assert search.error_rate == pytest.approx(0.0)

    def test_multi_feature_checkout_stats(self):
        # Checkout elapsed: [500, 600, 700]; sorted same; duration=5s
        # p50 → index 1.0 → 600.0
        # p90 → index 1.8 → 680.0
        # p95 → index 1.9 → 690.0
        # std(ddof=1): mean=600, SS=20000, var=10000, std=100.0
        # throughput = 3/5 = 0.6
        report = _report("multi_feature.jtl")
        by_name = {fs.name: fs for fs in report.per_feature}
        checkout = by_name["Checkout"]
        assert checkout.count == 3
        assert checkout.mean_ms == pytest.approx(600.0)
        assert checkout.p50_ms == pytest.approx(600.0)
        assert checkout.p90_ms == pytest.approx(680.0)
        assert checkout.p95_ms == pytest.approx(690.0)
        assert checkout.max_ms == pytest.approx(700.0)
        assert checkout.std_ms == pytest.approx(100.0)
        # feature_duration = (1700000005000 - 1700000003000) / 1000 = 2.0s → 3/2 = 1.5
        assert checkout.throughput == pytest.approx(1.5, rel=1e-3)
        assert checkout.error_rate == pytest.approx(1 / 3)

    def test_with_errors_single_feature_error_rate(self):
        report = _report("with_errors.jtl")
        assert len(report.per_feature) == 1
        assert report.per_feature[0].error_rate == pytest.approx(2 / 6)


class TestReportStructure:
    def test_report_is_frozen(self):
        report = _report("small_clean.jtl")
        with pytest.raises(Exception):
            report.total_count = 999  # type: ignore[misc]

    def test_dataset_metadata_embedded(self):
        ds = loader.run(str(FIXTURES / "small_clean.jtl"))
        report = statistician.run(ds)
        assert report.dataset_metadata == ds.metadata

    def test_per_feature_is_tuple(self):
        report = _report("small_clean.jtl")
        assert isinstance(report.per_feature, tuple)

    def test_feature_stats_new_fields_present(self):
        fs = _report("small_clean.jtl").per_feature[0]
        assert hasattr(fs, "p50_ms")
        assert hasattr(fs, "p90_ms")
        assert hasattr(fs, "p95_ms")
        assert hasattr(fs, "p99_ms")
        assert hasattr(fs, "std_ms")
        assert hasattr(fs, "throughput")

    def test_feature_stats_min_ms_removed(self):
        fs = _report("small_clean.jtl").per_feature[0]
        assert not hasattr(fs, "min_ms")

    def test_stats_report_new_fields_present(self):
        report = _report("small_clean.jtl")
        assert hasattr(report, "global_p50_ms")
        assert hasattr(report, "global_p90_ms")
        assert hasattr(report, "global_p95_ms")
        assert hasattr(report, "global_p99_ms")
        assert hasattr(report, "global_std_ms")
        assert hasattr(report, "global_throughput")

    def test_stats_report_global_min_ms_removed(self):
        report = _report("small_clean.jtl")
        assert not hasattr(report, "global_min_ms")


class TestEdgeCases:
    def test_global_throughput_is_zero_when_dataset_duration_is_zero(self):
        # Samples have distinct timestamps so feature duration > 0, but the
        # dataset-level duration_seconds is 0 — only global_throughput is 0.
        ts_base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [ts_base, ts_base.replace(second=1), ts_base.replace(second=2)]
                ),
                "elapsed": [100.0, 200.0, 300.0],
                "label": ["A", "A", "A"],
                "success": [True, True, True],
            }
        )
        meta = DatasetMetadata(
            file_path="fake.jtl",
            row_count=3,
            start_time=ts_base,
            end_time=ts_base,
            duration_seconds=0.0,
        )
        ds = NormalizedDataset(metadata=meta, data=df)
        report = statistician.run(ds)
        assert report.global_throughput == pytest.approx(0.0)

    def test_feature_throughput_is_zero_when_all_samples_share_timestamp(self):
        # All samples for the feature land at the same instant → feature duration = 0.
        ds = _in_memory_dataset([100.0, 200.0, 300.0], ["A", "A", "A"], duration_seconds=5.0)
        report = statistician.run(ds)
        assert report.per_feature[0].throughput == pytest.approx(0.0)

    def test_single_sample_feature_throughput_is_zero(self):
        ds = _in_memory_dataset([150.0], ["A"], duration_seconds=5.0)
        report = statistician.run(ds)
        assert report.per_feature[0].throughput == pytest.approx(0.0)

    def test_single_sample_std_is_zero(self):
        ds = _in_memory_dataset([150.0], ["A"], duration_seconds=5.0)
        report = statistician.run(ds)
        assert report.global_std_ms == pytest.approx(0.0)
        assert report.per_feature[0].std_ms == pytest.approx(0.0)

    def test_single_sample_feature_std_is_zero(self):
        # Feature "Multi" has 3 samples (meaningful std); "Solo" has exactly 1.
        ds = _in_memory_dataset(
            [100.0, 200.0, 300.0, 999.0],
            ["Multi", "Multi", "Multi", "Solo"],
            duration_seconds=10.0,
        )
        report = statistician.run(ds)
        by_name = {fs.name: fs for fs in report.per_feature}

        solo = by_name["Solo"]
        assert solo.std_ms == pytest.approx(0.0), "std of a single sample must be 0.0, not NaN"

        multi = by_name["Multi"]
        # std(ddof=1) of [100,200,300]: mean=200, SS=20000, var=10000, std=100.0
        assert multi.std_ms == pytest.approx(100.0, rel=1e-3), "multi-sample std should be non-zero"
