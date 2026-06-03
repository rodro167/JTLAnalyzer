"""Core data models: dataset containers, statistical reports, and orchestration plan structures."""


from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class DatasetMetadata:
    """Descriptive metadata extracted from a loaded JTL file.

    Attributes:
        file_path: Absolute path to the source JTL file.
        row_count: Number of samples after loading, normalization, and warmup
            filtering.
        start_time: Earliest timestamp in the (post-warmup) dataset.
        end_time: Latest timestamp in the (post-warmup) dataset.
        duration_seconds: Wall-clock duration of the test run in seconds.
        warmup_seconds: Duration in seconds excluded from the start of the
            dataset. 0.0 when no warmup filtering was applied.
        original_row_count: Number of samples before warmup filtering. Equals
            row_count when warmup_seconds is 0.0.
    """

    file_path: str
    row_count: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    warmup_seconds: float = 0.0
    original_row_count: int = 0


@dataclass(frozen=True)
class NormalizedDataset:
    """A loaded, cleaned JTL dataset together with its metadata.

    ``data`` is excluded from ``__eq__`` and ``__hash__`` because DataFrames are
    mutable and not hashable. ``frozen=True`` prevents accidental reassignment;
    in-place mutation of ``data`` is prohibited by convention.

    Attributes:
        metadata: Descriptive metadata for the source file.
        data: Cleaned DataFrame with at minimum the columns ``timestamp``,
            ``elapsed``, ``label``, and ``success``.
    """

    metadata: DatasetMetadata
    data: pd.DataFrame = field(compare=False)


@dataclass(frozen=True)
class FeatureStats:
    """Statistics for a single endpoint or feature label.

    Attributes:
        name: The ``label`` value from the JTL file identifying this feature.
        count: Number of samples.
        mean_ms: Mean elapsed time in milliseconds.
        p50_ms: Median (50th percentile) elapsed time in milliseconds.
        p90_ms: 90th percentile elapsed time in milliseconds.
        p95_ms: 95th percentile elapsed time in milliseconds.
        p99_ms: 99th percentile elapsed time in milliseconds.
        std_ms: Standard deviation of elapsed time in milliseconds (ddof=1).
            Set to 0.0 when count is 1 (std of a single value is undefined).
        max_ms: Maximum elapsed time in milliseconds.
        throughput: Requests per second for this feature, computed as count
            divided by the feature's own active time window (max − min timestamp
            of that feature's samples, in seconds). 0.0 when the feature has
            only one sample or all samples share the same timestamp.
        error_rate: Fraction of failed requests, in the range [0.0, 1.0].
    """

    name: str
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    max_ms: float
    throughput: float
    error_rate: float


@dataclass(frozen=True)
class StatsReport:
    """Aggregated statistics produced by the statistician agent.

    Attributes:
        dataset_metadata: Source metadata, making this report self-contained.
        total_count: Total number of samples across all features.
        global_mean_ms: Mean elapsed time across all samples (ms).
        global_p50_ms: Median (50th percentile) elapsed time across all samples (ms).
        global_p90_ms: 90th percentile elapsed time across all samples (ms).
        global_p95_ms: 95th percentile elapsed time across all samples (ms).
        global_p99_ms: 99th percentile elapsed time across all samples (ms).
        global_std_ms: Standard deviation of elapsed time across all samples (ms,
            ddof=1). 0.0 when total_count is 1.
        global_max_ms: Maximum elapsed time across all samples (ms).
        global_throughput: Requests per second across all samples. 0.0 if
            duration is zero or negative.
        global_error_rate: Fraction of failed requests globally, in [0.0, 1.0].
        per_feature: Per-feature breakdown, one entry per unique label.
    """

    dataset_metadata: DatasetMetadata
    total_count: int
    global_mean_ms: float
    global_p50_ms: float
    global_p90_ms: float
    global_p95_ms: float
    global_p99_ms: float
    global_std_ms: float
    global_max_ms: float
    global_throughput: float
    global_error_rate: float
    per_feature: tuple[FeatureStats, ...]


@dataclass(frozen=True)
class ResponseCodeBreakdown:
    """A single response code's frequency within a feature.

    Attributes:
        code: The response code as it appears in the JTL (string, since JMeter
            also reports non-numeric codes like "Non HTTP response code: ...").
        count: Number of samples with this code for the feature.
        percentage: Fraction of the feature's total samples with this code,
            in the range [0.0, 1.0].
    """

    code: str
    count: int
    percentage: float


@dataclass(frozen=True)
class ErrorsReport:
    """Per-feature response code distribution produced by the errors agent.

    Attributes:
        dataset_metadata: Source metadata, making this report self-contained.
        codes_by_feature: Mapping from feature label to the tuple of response
            code breakdowns for that feature. Each tuple is sorted by count
            descending (most frequent code first).
    """

    dataset_metadata: DatasetMetadata
    codes_by_feature: dict[str, tuple[ResponseCodeBreakdown, ...]]


@dataclass(frozen=True)
class AnomalousSample:
    """A single sample identified as anomalous within its feature.

    Attributes:
        timestamp: When the sample occurred.
        elapsed_ms: Response time in milliseconds.
        feature: The feature label this sample belongs to.
        response_code: HTTP code or non-HTTP code string from the JTL.
        deviation_factor: How many times the sample's elapsed exceeds the
            feature's upper IQR threshold. Computed as elapsed_ms / threshold_ms.
            A value of 3.0 means the sample is 3x the threshold.
    """

    timestamp: datetime
    elapsed_ms: float
    feature: str
    response_code: str
    deviation_factor: float


@dataclass(frozen=True)
class FeatureAnomalies:
    """All anomalies detected for one feature, with context.

    Attributes:
        feature: The feature label.
        total_samples: Total samples of this feature in the dataset.
        anomaly_count: Number of samples flagged as anomalous.
        anomaly_rate: anomaly_count / total_samples, in [0.0, 1.0].
        threshold_ms: The IQR upper bound used (Q3 + 1.5 * IQR), in milliseconds.
            Set to 0.0 when detection was skipped (insufficient_data=True).
        samples: Tuple of anomalous samples, sorted by deviation_factor descending.
        insufficient_data: True if the feature had fewer than MIN_SAMPLES_FOR_DETECTION
            samples, or if IQR == 0 (degenerate distribution), and detection was skipped.
    """

    feature: str
    total_samples: int
    anomaly_count: int
    anomaly_rate: float
    threshold_ms: float
    samples: tuple[AnomalousSample, ...]
    insufficient_data: bool


@dataclass(frozen=True)
class AnomaliesReport:
    """Per-feature anomaly detection results produced by the anomalies agent.

    Attributes:
        dataset_metadata: Source metadata, making this report self-contained.
        by_feature: Mapping from feature label to its detection results.
    """

    dataset_metadata: DatasetMetadata
    by_feature: dict[str, FeatureAnomalies]


@dataclass(frozen=True)
class DegradationWindow:
    """A contiguous time window where a feature's response times degraded.

    Attributes:
        start_time: Start of the degraded window (timestamp of first degraded bin).
        end_time: End of the degraded window (timestamp of last degraded bin's
            end, i.e. last bin start + BIN_SIZE_SECONDS).
        duration_seconds: Length of the window in seconds.
        window_median_ms: Median elapsed within the window.
        reference_median_ms: Median elapsed for the feature outside any degraded
            window (the "normal" baseline, computed as median of per-bin medians).
        degradation_factor: window_median_ms / reference_median_ms.
    """

    start_time: datetime
    end_time: datetime
    duration_seconds: float
    window_median_ms: float
    reference_median_ms: float
    degradation_factor: float


@dataclass(frozen=True)
class FeatureTrends:
    """Temporal trend analysis for one feature.

    Attributes:
        feature: The feature label.
        total_samples: Total samples of this feature in the (post-warmup) dataset.
        windows: All detected degradation windows, sorted chronologically.
        insufficient_data: True if the feature has fewer than 20 samples, if
            it has no non-empty 1-minute bins, or if the reference median is 0
            (degenerate distribution). Detection was skipped in all these cases.
    """

    feature: str
    total_samples: int
    windows: tuple[DegradationWindow, ...]
    insufficient_data: bool


@dataclass(frozen=True)
class TrendsReport:
    """Per-feature temporal trend analysis produced by the trends agent.

    Attributes:
        dataset_metadata: Source metadata, making this report self-contained.
        by_feature: Mapping from feature label to its trend analysis.
    """

    dataset_metadata: DatasetMetadata
    by_feature: dict[str, FeatureTrends]


@dataclass(frozen=True)
class AnalysisResult:
    """Combined output of a full JTL analysis pipeline.

    Attributes:
        stats: Performance metrics produced by the statistician agent.
        errors: Per-feature response code distribution produced by the errors agent.
        anomalies: Per-feature anomaly detection produced by the anomalies agent.
        trends: Per-feature temporal degradation analysis produced by the trends
            agent.
    """

    stats: StatsReport
    errors: ErrorsReport
    anomalies: AnomaliesReport
    trends: TrendsReport


class PlanStep(BaseModel):
    """A single step in an orchestration plan, as returned by the LLM.

    Validated by pydantic because it is parsed directly from LLM JSON output.
    Frozen after construction so no downstream code can mutate a parsed plan.

    Attributes:
        step_id: Unique label for this step within the plan (e.g. ``"load"``,
            ``"analyze"``). Used as the key in ``depends_on`` references.
        agent: Name of the registered agent to invoke (e.g. ``"loader"``,
            ``"statistician"``). The orchestrator rejects any name that is not
            registered, raising ``PlanValidationError``.
        depends_on: List of ``step_id`` values that must complete successfully
            before this step is eligible to run. An empty list means the step
            has no prerequisites and may run immediately. The orchestrator
            validates that all referenced IDs exist and that the resulting
            graph is acyclic.
        params: Arbitrary key-value parameters forwarded verbatim to the agent
            at execution time (e.g. ``{"file_path": "/path/to/test.jtl"}``).
            The dict's contents remain mutable by Python's rules; callers must
            not modify params after the plan is constructed.
    """

    model_config = {"frozen": True}

    step_id: str
    agent: str
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """A validated orchestration plan produced by the orchestrator.

    Parsed from LLM JSON output and validated by pydantic. Frozen after
    construction so that the executor always operates on a stable, immutable
    plan regardless of how many agents it passes through.

    Attributes:
        steps: The complete list of operations the executor will run. Order in
            the list is not guaranteed to be topological — the executor performs
            its own topological sort based on each step's ``depends_on`` field.
            Every ``step_id`` referenced in any ``depends_on`` must appear in
            this list; the orchestrator raises ``PlanValidationError`` otherwise.
    """

    model_config = {"frozen": True}

    steps: list[PlanStep]
