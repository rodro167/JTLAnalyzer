"""Unit tests for agents/statistician.py."""

from pathlib import Path

import pytest

from jtl_analyzer.agents import loader, statistician
from jtl_analyzer.core.models import StatsReport

FIXTURES = Path(__file__).parent / "fixtures"


def _report(fixture_name: str) -> StatsReport:
    ds = loader.run(str(FIXTURES / fixture_name))
    return statistician.run(ds)


class TestGlobalStats:
    def test_small_clean_total_count(self):
        assert _report("small_clean.jtl").total_count == 6

    def test_small_clean_mean(self):
        assert _report("small_clean.jtl").global_mean_ms == pytest.approx(200.0)

    def test_small_clean_min(self):
        assert _report("small_clean.jtl").global_min_ms == pytest.approx(100.0)

    def test_small_clean_max(self):
        assert _report("small_clean.jtl").global_max_ms == pytest.approx(300.0)

    def test_small_clean_error_rate_zero(self):
        assert _report("small_clean.jtl").global_error_rate == pytest.approx(0.0)

    def test_with_errors_total_count(self):
        assert _report("with_errors.jtl").total_count == 6

    def test_with_errors_mean(self):
        # (100+200+300+150+250+350) / 6 = 225.0
        assert _report("with_errors.jtl").global_mean_ms == pytest.approx(225.0)

    def test_with_errors_error_rate(self):
        assert _report("with_errors.jtl").global_error_rate == pytest.approx(2 / 6)

    def test_multi_feature_total_count(self):
        assert _report("multi_feature.jtl").total_count == 6

    def test_multi_feature_mean(self):
        # (50+60+70+500+600+700) / 6 = 330.0
        assert _report("multi_feature.jtl").global_mean_ms == pytest.approx(330.0)

    def test_multi_feature_min(self):
        assert _report("multi_feature.jtl").global_min_ms == pytest.approx(50.0)

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
        report = _report("multi_feature.jtl")
        by_name = {fs.name: fs for fs in report.per_feature}
        search = by_name["Search"]
        assert search.count == 3
        assert search.mean_ms == pytest.approx(60.0)
        assert search.min_ms == pytest.approx(50.0)
        assert search.max_ms == pytest.approx(70.0)
        assert search.error_rate == pytest.approx(0.0)

    def test_multi_feature_checkout_stats(self):
        report = _report("multi_feature.jtl")
        by_name = {fs.name: fs for fs in report.per_feature}
        checkout = by_name["Checkout"]
        assert checkout.count == 3
        assert checkout.mean_ms == pytest.approx(600.0)
        assert checkout.min_ms == pytest.approx(500.0)
        assert checkout.max_ms == pytest.approx(700.0)
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
