"""Unit tests for agents/loader.py."""

from pathlib import Path

import pytest

from jtl_analyzer.agents import loader
from jtl_analyzer.core.exceptions import InvalidJTLError
from jtl_analyzer.core.models import NormalizedDataset

FIXTURES = Path(__file__).parent / "fixtures"


class TestLoadValidFiles:
    def test_small_clean_returns_normalized_dataset(self):
        ds = loader.run(str(FIXTURES / "small_clean.jtl"))
        assert isinstance(ds, NormalizedDataset)

    def test_small_clean_row_count(self):
        ds = loader.run(str(FIXTURES / "small_clean.jtl"))
        assert ds.metadata.row_count == 6

    def test_small_clean_required_columns_present(self):
        ds = loader.run(str(FIXTURES / "small_clean.jtl"))
        assert {"timestamp", "elapsed", "label", "success"}.issubset(ds.data.columns)

    def test_small_clean_duration(self):
        ds = loader.run(str(FIXTURES / "small_clean.jtl"))
        assert ds.metadata.duration_seconds == pytest.approx(5.0)

    def test_small_clean_all_success(self):
        ds = loader.run(str(FIXTURES / "small_clean.jtl"))
        assert ds.data["success"].all()

    def test_with_errors_row_count(self):
        ds = loader.run(str(FIXTURES / "with_errors.jtl"))
        assert ds.metadata.row_count == 6

    def test_with_errors_success_count(self):
        ds = loader.run(str(FIXTURES / "with_errors.jtl"))
        assert ds.data["success"].sum() == 4

    def test_with_errors_failure_count(self):
        ds = loader.run(str(FIXTURES / "with_errors.jtl"))
        assert (~ds.data["success"]).sum() == 2

    def test_multi_feature_row_count(self):
        ds = loader.run(str(FIXTURES / "multi_feature.jtl"))
        assert ds.metadata.row_count == 6

    def test_multi_feature_labels(self):
        ds = loader.run(str(FIXTURES / "multi_feature.jtl"))
        assert set(ds.data["label"].unique()) == {"Search", "Checkout"}

    def test_metadata_file_path_is_absolute(self):
        ds = loader.run(str(FIXTURES / "small_clean.jtl"))
        assert Path(ds.metadata.file_path).is_absolute()

    def test_elapsed_column_is_float(self):
        ds = loader.run(str(FIXTURES / "small_clean.jtl"))
        assert ds.data["elapsed"].dtype == float

    def test_success_column_is_bool(self):
        ds = loader.run(str(FIXTURES / "small_clean.jtl"))
        assert ds.data["success"].dtype == bool


class TestLoadErrors:
    def test_missing_file_raises_invalid_jtl_error(self):
        with pytest.raises(InvalidJTLError):
            loader.run("/nonexistent/path/file.jtl")

    def test_missing_file_message_contains_path(self):
        path = "/nonexistent/path/file.jtl"
        with pytest.raises(InvalidJTLError, match="not found"):
            loader.run(path)

    def test_missing_columns_raises_invalid_jtl_error(self, tmp_path):
        bad = tmp_path / "bad.jtl"
        bad.write_text("timeStamp,elapsed\n1700000000000,100\n")
        with pytest.raises(InvalidJTLError, match="Missing required columns"):
            loader.run(str(bad))

    def test_missing_columns_message_names_columns(self, tmp_path):
        bad = tmp_path / "bad.jtl"
        bad.write_text("timeStamp,elapsed\n1700000000000,100\n")
        with pytest.raises(InvalidJTLError, match="label"):
            loader.run(str(bad))

    def test_empty_file_raises_invalid_jtl_error(self, tmp_path):
        empty = tmp_path / "empty.jtl"
        empty.write_text("")
        with pytest.raises(InvalidJTLError):
            loader.run(str(empty))


class TestFormatDetection:
    def test_detects_xml_for_xml_fixture(self):
        assert loader._detect_format(str(FIXTURES / "small_clean_xml.jtl")) == "xml"

    def test_detects_csv_for_csv_fixture(self):
        assert loader._detect_format(str(FIXTURES / "small_clean.jtl")) == "csv"


class TestLoadXMLFiles:
    def test_small_clean_xml_returns_normalized_dataset(self):
        ds = loader.run(str(FIXTURES / "small_clean_xml.jtl"))
        assert isinstance(ds, NormalizedDataset)

    def test_small_clean_xml_row_count(self):
        ds = loader.run(str(FIXTURES / "small_clean_xml.jtl"))
        assert ds.metadata.row_count == 6

    def test_small_clean_xml_required_columns_present(self):
        ds = loader.run(str(FIXTURES / "small_clean_xml.jtl"))
        assert {"timestamp", "elapsed", "label", "success"}.issubset(ds.data.columns)

    def test_small_clean_xml_elapsed_is_float(self):
        ds = loader.run(str(FIXTURES / "small_clean_xml.jtl"))
        assert ds.data["elapsed"].dtype == float

    def test_small_clean_xml_success_is_bool(self):
        ds = loader.run(str(FIXTURES / "small_clean_xml.jtl"))
        assert ds.data["success"].dtype == bool

    def test_small_clean_xml_one_failure(self):
        ds = loader.run(str(FIXTURES / "small_clean_xml.jtl"))
        assert (~ds.data["success"]).sum() == 1

    def test_mixed_xml_elements_row_count(self):
        ds = loader.run(str(FIXTURES / "mixed_xml_elements.jtl"))
        assert ds.metadata.row_count == 6

    def test_mixed_xml_elements_labels(self):
        ds = loader.run(str(FIXTURES / "mixed_xml_elements.jtl"))
        assert set(ds.data["label"].unique()) == {"Search", "API"}


class TestXMLErrors:
    def test_malformed_xml_raises_invalid_jtl_error(self, tmp_path):
        bad = tmp_path / "bad.jtl"
        bad.write_text('<?xml version="1.0"?>\n<testResults><broken>')
        with pytest.raises(InvalidJTLError):
            loader.run(str(bad))


class TestXMLTruncation:
    def test_truncated_xml_returns_partial_dataset(self):
        ds = loader.run(str(FIXTURES / "truncated_xml.jtl"))
        assert isinstance(ds, NormalizedDataset)
        assert ds.metadata.row_count == 4

    def test_zero_samples_malformed_xml_still_raises(self, tmp_path):
        bad = tmp_path / "garbage.jtl"
        bad.write_text('<?xml version="1.0"?>\n<garbage')
        with pytest.raises(InvalidJTLError):
            loader.run(str(bad))

    def test_truncated_xml_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="jtl_analyzer.agents.loader"):
            loader.run(str(FIXTURES / "truncated_xml.jtl"))
        assert any("truncated" in r.message.lower() for r in caplog.records)


class TestWarmup:
    def _make_csv(self, tmp_path, duration_seconds: int, count: int) -> "Path":
        """CSV with ``count`` samples evenly spread over ``duration_seconds``."""
        base_ts = 1_700_000_000_000  # ms
        interval_ms = (duration_seconds * 1000) // (count - 1) if count > 1 else 1000
        header = "timeStamp,elapsed,label,success,responseCode"
        rows = [f"{base_ts + i * interval_ms},100,Feature,true,200" for i in range(count)]
        f = tmp_path / "warmup.jtl"
        f.write_text(header + "\n" + "\n".join(rows))
        return f

    def test_warmup_excludes_initial_samples(self, tmp_path):
        # 11 samples at t=0,1,...,10s; warmup=5s keeps t>=5s → 6 samples
        f = self._make_csv(tmp_path, duration_seconds=10, count=11)
        ds = loader.run(str(f), warmup_seconds=5)
        assert ds.metadata.row_count == 6

    def test_warmup_zero_keeps_everything(self, tmp_path):
        f = self._make_csv(tmp_path, duration_seconds=10, count=11)
        ds = loader.run(str(f), warmup_seconds=0.0)
        assert ds.metadata.row_count == 11

    def test_warmup_exceeds_dataset_raises(self, tmp_path):
        f = self._make_csv(tmp_path, duration_seconds=10, count=11)
        with pytest.raises(InvalidJTLError, match="excluded all samples"):
            loader.run(str(f), warmup_seconds=100)

    def test_warmup_metadata_reflects_applied_warmup(self, tmp_path):
        f = self._make_csv(tmp_path, duration_seconds=10, count=11)
        ds = loader.run(str(f), warmup_seconds=5)
        assert ds.metadata.warmup_seconds == 5.0
        assert ds.metadata.original_row_count == 11
        assert ds.metadata.original_row_count > ds.metadata.row_count


_LARGE_XML_FIXTURE = FIXTURES / "large_xml_real.jtl"


@pytest.mark.skipif(
    not _LARGE_XML_FIXTURE.exists(),
    reason="large_xml_real.jtl not present in fixtures",
)
def test_large_xml_real_loads_successfully():
    ds = loader.run(str(_LARGE_XML_FIXTURE))
    assert len(ds.data) > 1000
