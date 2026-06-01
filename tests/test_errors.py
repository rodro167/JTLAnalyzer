"""Unit tests for agents/errors.py."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from jtl_analyzer.agents import errors as errors_agent
from jtl_analyzer.agents import loader as loader_agent
from jtl_analyzer.core.models import DatasetMetadata, ErrorsReport, NormalizedDataset, ResponseCodeBreakdown

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


class TestSingleFeatureCodes:
    """small_clean.jtl: 6 Homepage rows, all responseCode=200, all success."""

    def test_single_feature_key(self):
        report = errors_agent.run(_load("small_clean.jtl"))
        assert set(report.codes_by_feature.keys()) == {"Homepage"}

    def test_single_code_200(self):
        report = errors_agent.run(_load("small_clean.jtl"))
        breakdowns = report.codes_by_feature["Homepage"]
        assert len(breakdowns) == 1
        assert breakdowns[0].code == "200"

    def test_code_count_and_percentage(self):
        report = errors_agent.run(_load("small_clean.jtl"))
        bd = report.codes_by_feature["Homepage"][0]
        assert bd.count == 6
        assert bd.percentage == pytest.approx(1.0)


class TestMultipleCodes:
    """with_errors.jtl: Login, 4×200, 2×500."""

    def test_two_codes_present(self):
        report = errors_agent.run(_load("with_errors.jtl"))
        assert len(report.codes_by_feature["Login"]) == 2

    def test_code_200_stats(self):
        report = errors_agent.run(_load("with_errors.jtl"))
        bd_200 = next(b for b in report.codes_by_feature["Login"] if b.code == "200")
        assert bd_200.count == 4
        assert bd_200.percentage == pytest.approx(4 / 6)

    def test_code_500_stats(self):
        report = errors_agent.run(_load("with_errors.jtl"))
        bd_500 = next(b for b in report.codes_by_feature["Login"] if b.code == "500")
        assert bd_500.count == 2
        assert bd_500.percentage == pytest.approx(2 / 6)

    def test_sorted_by_count_descending(self):
        report = errors_agent.run(_load("with_errors.jtl"))
        breakdowns = report.codes_by_feature["Login"]
        assert breakdowns[0].code == "200"
        assert breakdowns[1].code == "500"


class TestMultiFeatureSeparation:
    """multi_feature.jtl: Search (3×200) and Checkout (2×200, 1×500)."""

    def test_both_features_present(self):
        report = errors_agent.run(_load("multi_feature.jtl"))
        assert "Search" in report.codes_by_feature
        assert "Checkout" in report.codes_by_feature

    def test_search_only_has_200(self):
        report = errors_agent.run(_load("multi_feature.jtl"))
        codes = {b.code for b in report.codes_by_feature["Search"]}
        assert codes == {"200"}

    def test_checkout_has_200_and_500(self):
        report = errors_agent.run(_load("multi_feature.jtl"))
        codes = {b.code for b in report.codes_by_feature["Checkout"]}
        assert codes == {"200", "500"}

    def test_no_cross_contamination(self):
        report = errors_agent.run(_load("multi_feature.jtl"))
        search_total = sum(b.count for b in report.codes_by_feature["Search"])
        checkout_total = sum(b.count for b in report.codes_by_feature["Checkout"])
        assert search_total == 3
        assert checkout_total == 3

    def test_checkout_percentages_sum_to_one(self):
        report = errors_agent.run(_load("multi_feature.jtl"))
        total_pct = sum(b.percentage for b in report.codes_by_feature["Checkout"])
        assert total_pct == pytest.approx(1.0)


class TestReportStructure:
    def test_errors_report_is_frozen(self):
        report = errors_agent.run(_load("small_clean.jtl"))
        with pytest.raises(FrozenInstanceError):
            report.dataset_metadata = None  # type: ignore

    def test_response_code_breakdown_is_frozen(self):
        bd = ResponseCodeBreakdown(code="200", count=5, percentage=1.0)
        with pytest.raises(FrozenInstanceError):
            bd.count = 0  # type: ignore

    def test_codes_by_feature_is_dict(self):
        report = errors_agent.run(_load("small_clean.jtl"))
        assert isinstance(report.codes_by_feature, dict)

    def test_return_type_is_errors_report(self):
        report = errors_agent.run(_load("small_clean.jtl"))
        assert isinstance(report, ErrorsReport)


class TestNonHttpCodes:
    """Verify long non-HTTP response codes are preserved as-is."""

    LONG_CODE = "Non HTTP response code: java.net.SocketTimeoutException"

    def test_non_http_code_key_preserved(self):
        dataset = _make_dataset([
            {"timestamp": 1700000000000, "elapsed": 100, "label": "API", "responseCode": self.LONG_CODE, "success": "false"},
            {"timestamp": 1700000001000, "elapsed": 200, "label": "API", "responseCode": "200", "success": "true"},
        ])
        report = errors_agent.run(dataset)
        codes = {b.code for b in report.codes_by_feature["API"]}
        assert self.LONG_CODE in codes

    def test_non_http_code_count_and_percentage(self):
        dataset = _make_dataset([
            {"timestamp": 1700000000000, "elapsed": 100, "label": "API", "responseCode": self.LONG_CODE, "success": "false"},
            {"timestamp": 1700000001000, "elapsed": 200, "label": "API", "responseCode": "200", "success": "true"},
        ])
        report = errors_agent.run(dataset)
        bd = next(b for b in report.codes_by_feature["API"] if b.code == self.LONG_CODE)
        assert bd.count == 1
        assert bd.percentage == pytest.approx(0.5)
