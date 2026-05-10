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
