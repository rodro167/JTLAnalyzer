"""Unit tests for agents/trends.py."""

from pathlib import Path

import pandas as pd
import pytest

from jtl_analyzer.agents import loader, trends
from jtl_analyzer.core.models import (
    DatasetMetadata,
    DegradationWindow,
    FeatureTrends,
    NormalizedDataset,
    TrendsReport,
)

FIXTURES = Path(__file__).parent / "fixtures"
_BASE = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_dataset(
    samples_by_minute: dict[int, tuple[int, float]],
    label: str = "API",
) -> NormalizedDataset:
    """Build an in-memory NormalizedDataset from minute-keyed sample specs.

    Args:
        samples_by_minute: Mapping from minute offset to (count, elapsed_ms).
            Each minute produces ``count`` samples spread at 1-second intervals
            within that minute.
        label: The feature label for all rows.
    """
    rows = []
    for minute, (count, elapsed) in sorted(samples_by_minute.items()):
        for i in range(count):
            rows.append(
                {
                    "timestamp": _BASE + pd.Timedelta(minutes=minute, seconds=i),
                    "elapsed": float(elapsed),
                    "label": label,
                    "success": True,
                    "responseCode": "200",
                }
            )
    df = pd.DataFrame(rows)
    start = df["timestamp"].min().to_pydatetime()
    end = df["timestamp"].max().to_pydatetime()
    meta = DatasetMetadata(
        file_path="test.jtl",
        row_count=len(df),
        original_row_count=len(df),
        start_time=start,
        end_time=end,
        duration_seconds=(end - start).total_seconds(),
        warmup_seconds=0.0,
    )
    return NormalizedDataset(metadata=meta, data=df)


# ---------------------------------------------------------------------------
# TestBasicDetection
# ---------------------------------------------------------------------------

class TestBasicDetection:
    """Single degraded minute in a 5-minute run."""

    # minutes 0-2 normal (100ms), minute 3 degraded (300ms = 3x), minute 4 normal
    _SPECS = {0: (12, 100.0), 1: (12, 100.0), 2: (12, 100.0), 3: (12, 300.0), 4: (12, 100.0)}

    def test_one_window_detected(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert len(report.by_feature["API"].windows) == 1

    def test_insufficient_data_false(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert report.by_feature["API"].insufficient_data is False

    def test_window_starts_at_degraded_minute(self):
        report = trends.run(_make_dataset(self._SPECS))
        expected = (_BASE + pd.Timedelta(minutes=3)).to_pydatetime()
        assert report.by_feature["API"].windows[0].start_time == expected

    def test_window_duration_is_one_minute(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert report.by_feature["API"].windows[0].duration_seconds == pytest.approx(60.0)

    def test_degradation_factor_approx_three(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert report.by_feature["API"].windows[0].degradation_factor == pytest.approx(3.0, rel=0.01)


# ---------------------------------------------------------------------------
# TestConsecutiveBinsMerged
# ---------------------------------------------------------------------------

class TestConsecutiveBinsMerged:
    """Three consecutive degraded minutes merge into one window."""

    # minutes 0-1 normal, 2-4 degraded, 5-6 normal
    _SPECS = {
        0: (12, 100.0), 1: (12, 100.0),
        2: (12, 300.0), 3: (12, 300.0), 4: (12, 300.0),
        5: (12, 100.0), 6: (12, 100.0),
    }

    def test_exactly_one_window(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert len(report.by_feature["API"].windows) == 1

    def test_window_duration_three_minutes(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert report.by_feature["API"].windows[0].duration_seconds == pytest.approx(180.0)

    def test_window_start_at_minute_two(self):
        report = trends.run(_make_dataset(self._SPECS))
        expected = (_BASE + pd.Timedelta(minutes=2)).to_pydatetime()
        assert report.by_feature["API"].windows[0].start_time == expected

    def test_window_end_at_minute_five(self):
        report = trends.run(_make_dataset(self._SPECS))
        expected = (_BASE + pd.Timedelta(minutes=5)).to_pydatetime()
        assert report.by_feature["API"].windows[0].end_time == expected


# ---------------------------------------------------------------------------
# TestNonConsecutiveBinsKeptSeparate
# ---------------------------------------------------------------------------

class TestNonConsecutiveBinsKeptSeparate:
    """Two degraded minutes separated by normal minutes form two windows."""

    # minutes 0-1 normal, 2 degraded, 3-4 normal, 5 degraded, 6 normal
    _SPECS = {
        0: (12, 100.0), 1: (12, 100.0),
        2: (12, 300.0),
        3: (12, 100.0), 4: (12, 100.0),
        5: (12, 300.0),
        6: (12, 100.0),
    }

    def test_two_windows_detected(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert len(report.by_feature["API"].windows) == 2

    def test_first_window_at_minute_two(self):
        report = trends.run(_make_dataset(self._SPECS))
        expected = (_BASE + pd.Timedelta(minutes=2)).to_pydatetime()
        assert report.by_feature["API"].windows[0].start_time == expected

    def test_second_window_at_minute_five(self):
        report = trends.run(_make_dataset(self._SPECS))
        expected = (_BASE + pd.Timedelta(minutes=5)).to_pydatetime()
        assert report.by_feature["API"].windows[1].start_time == expected

    def test_each_window_one_minute_duration(self):
        report = trends.run(_make_dataset(self._SPECS))
        for w in report.by_feature["API"].windows:
            assert w.duration_seconds == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# TestEmptyBinsIgnored
# ---------------------------------------------------------------------------

class TestEmptyBinsIgnored:
    """Samples only at minutes 0, 1, 4 — minutes 2 and 3 empty.

    The gap between minute 1 and minute 4 must not create spurious windows.
    """

    # 36 samples total (>= 20): minutes 0, 1 normal; minute 4 degraded
    _SPECS = {0: (12, 100.0), 1: (12, 100.0), 4: (12, 300.0)}

    def test_exactly_one_window(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert len(report.by_feature["API"].windows) == 1

    def test_window_at_minute_four(self):
        report = trends.run(_make_dataset(self._SPECS))
        expected = (_BASE + pd.Timedelta(minutes=4)).to_pydatetime()
        assert report.by_feature["API"].windows[0].start_time == expected

    def test_no_error_raised(self):
        # Primary guard: empty bins must not cause exceptions
        report = trends.run(_make_dataset(self._SPECS))
        assert isinstance(report, TrendsReport)


# ---------------------------------------------------------------------------
# TestInsufficientData
# ---------------------------------------------------------------------------

class TestInsufficientData:
    """Feature with fewer than 20 samples is skipped."""

    _SPECS = {0: (15, 100.0)}  # 15 < MIN_SAMPLES_FOR_DETECTION

    def test_insufficient_data_true(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert report.by_feature["API"].insufficient_data is True

    def test_windows_empty(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert report.by_feature["API"].windows == ()

    def test_total_samples_recorded(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert report.by_feature["API"].total_samples == 15


# ---------------------------------------------------------------------------
# TestNoDegradation
# ---------------------------------------------------------------------------

class TestNoDegradation:
    """Stable elapsed values across all bins — no windows expected."""

    _SPECS = {i: (12, 100.0) for i in range(5)}  # 60 samples, all 100ms

    def test_no_windows(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert report.by_feature["API"].windows == ()

    def test_insufficient_data_false(self):
        report = trends.run(_make_dataset(self._SPECS))
        assert report.by_feature["API"].insufficient_data is False


# ---------------------------------------------------------------------------
# TestPerFeatureIsolation
# ---------------------------------------------------------------------------

class TestPerFeatureIsolation:
    """Two features in the same dataset are analyzed independently."""

    def _build_multi_feature_dataset(self) -> NormalizedDataset:
        rows = []
        # Stable: 5 minutes at 100ms
        for minute in range(5):
            for i in range(12):
                rows.append({
                    "timestamp": _BASE + pd.Timedelta(minutes=minute, seconds=i),
                    "elapsed": 100.0,
                    "label": "Stable",
                    "success": True,
                    "responseCode": "200",
                })
        # Degrading: minutes 0-2 normal, minute 3 degraded (300ms), minute 4 normal
        for minute in range(5):
            elapsed = 300.0 if minute == 3 else 100.0
            for i in range(12):
                rows.append({
                    "timestamp": _BASE + pd.Timedelta(minutes=minute, seconds=i),
                    "elapsed": elapsed,
                    "label": "Degrading",
                    "success": True,
                    "responseCode": "200",
                })
        df = pd.DataFrame(rows)
        start = df["timestamp"].min().to_pydatetime()
        end = df["timestamp"].max().to_pydatetime()
        meta = DatasetMetadata(
            file_path="test.jtl",
            row_count=len(df),
            original_row_count=len(df),
            start_time=start,
            end_time=end,
            duration_seconds=(end - start).total_seconds(),
            warmup_seconds=0.0,
        )
        return NormalizedDataset(metadata=meta, data=df)

    def test_stable_feature_has_no_windows(self):
        report = trends.run(self._build_multi_feature_dataset())
        assert report.by_feature["Stable"].windows == ()

    def test_degrading_feature_has_one_window(self):
        report = trends.run(self._build_multi_feature_dataset())
        assert len(report.by_feature["Degrading"].windows) == 1

    def test_both_features_present(self):
        report = trends.run(self._build_multi_feature_dataset())
        assert set(report.by_feature.keys()) == {"Stable", "Degrading"}


# ---------------------------------------------------------------------------
# TestReportStructure
# ---------------------------------------------------------------------------

class TestReportStructure:
    def test_trends_report_is_frozen(self):
        report = trends.run(_make_dataset({0: (12, 100.0), 1: (12, 100.0)}))
        with pytest.raises(AttributeError):
            report.by_feature = {}  # type: ignore[misc]

    def test_feature_trends_is_frozen(self):
        report = trends.run(_make_dataset({0: (25, 100.0)}))
        ft = next(iter(report.by_feature.values()))
        with pytest.raises(AttributeError):
            ft.feature = "mutated"  # type: ignore[misc]

    def test_degradation_window_is_frozen(self):
        # Build a dataset that produces at least one window
        specs = {0: (12, 100.0), 1: (12, 100.0), 2: (12, 100.0), 3: (12, 300.0), 4: (12, 100.0)}
        report = trends.run(_make_dataset(specs))
        w = report.by_feature["API"].windows[0]
        with pytest.raises(AttributeError):
            w.duration_seconds = 0.0  # type: ignore[misc]

    def test_by_feature_is_dict(self):
        report = trends.run(_make_dataset({0: (25, 100.0)}))
        assert isinstance(report.by_feature, dict)

    def test_windows_is_tuple(self):
        report = trends.run(_make_dataset({i: (12, 100.0) for i in range(5)}))
        ft = next(iter(report.by_feature.values()))
        assert isinstance(ft.windows, tuple)

    def test_report_contains_dataset_metadata(self):
        dataset = _make_dataset({0: (25, 100.0)})
        report = trends.run(dataset)
        assert report.dataset_metadata is dataset.metadata


# ---------------------------------------------------------------------------
# TestRealisticLargeFixture  (20260516.jtl — restMovies, 146k CSV samples)
# ---------------------------------------------------------------------------

_FIXTURE_CSV = FIXTURES / "20260516.jtl"


class TestRealisticLargeFixture:
    """Structure and coverage tests against the committed 20260516.jtl fixture.

    The restMovies dataset is temporally stable: per-minute bin medians vary
    at most ~30% from the reference, so no degradation windows are expected at
    the 2x threshold.  These tests verify the agent runs cleanly and covers
    every feature label — not that windows are detected.
    """

    @pytest.fixture(scope="class")
    def analysis(self):
        dataset = loader.run(str(_FIXTURE_CSV))
        return dataset, trends.run(dataset)

    def test_returns_trends_report(self, analysis):
        _, report = analysis
        assert isinstance(report, TrendsReport)

    def test_all_feature_labels_present(self, analysis):
        dataset, report = analysis
        labels = set(dataset.data["label"].unique())
        assert set(report.by_feature.keys()) == labels

    def test_every_feature_has_a_result(self, analysis):
        dataset, report = analysis
        for label in dataset.data["label"].unique():
            ft = report.by_feature[label]
            assert isinstance(ft, FeatureTrends)


# ---------------------------------------------------------------------------
# TestRealisticXMLFixture  (WaterShuttles XML — burstier traffic, 2x+ degradation)
# ---------------------------------------------------------------------------

_FIXTURE_XML = FIXTURES / "2025_04_09WaterShuttlesFullMW1.jtl"


@pytest.mark.skipif(
    not _FIXTURE_XML.exists(),
    reason="2025_04_09WaterShuttlesFullMW1.jtl not present in fixtures",
)
class TestRealisticXMLFixture:
    """Integration test against the WaterShuttles XML fixture.

    Unlike the stable restMovies dataset, the WaterShuttles traffic has
    several features with per-minute medians well above 2x their reference,
    so at least one degradation window must be detected.
    """

    @pytest.fixture(scope="class")
    def report(self):
        dataset = loader.run(str(_FIXTURE_XML))
        return trends.run(dataset)

    def test_returns_trends_report(self, report):
        assert isinstance(report, TrendsReport)

    def test_at_least_one_degradation_window_detected(self, report):
        features_with_windows = [
            ft for ft in report.by_feature.values()
            if not ft.insufficient_data and ft.windows
        ]
        assert len(features_with_windows) > 0, (
            "No degradation windows detected in the WaterShuttles fixture. "
            "Check BIN_SIZE_SECONDS or DEGRADATION_FACTOR_THRESHOLD constants."
        )
