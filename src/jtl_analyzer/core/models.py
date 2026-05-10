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
        row_count: Number of samples after loading and normalization.
        start_time: Earliest timestamp in the dataset.
        end_time: Latest timestamp in the dataset.
        duration_seconds: Wall-clock duration of the test run in seconds.
    """

    file_path: str
    row_count: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float


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
        min_ms: Minimum elapsed time in milliseconds.
        max_ms: Maximum elapsed time in milliseconds.
        error_rate: Fraction of failed requests, in the range [0.0, 1.0].
    """

    name: str
    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    error_rate: float


@dataclass(frozen=True)
class StatsReport:
    """Aggregated statistics produced by the statistician agent.

    Attributes:
        dataset_metadata: Source metadata, making this report self-contained.
        total_count: Total number of samples across all features.
        global_mean_ms: Mean elapsed time across all samples (ms).
        global_min_ms: Minimum elapsed time across all samples (ms).
        global_max_ms: Maximum elapsed time across all samples (ms).
        global_error_rate: Fraction of failed requests globally, in [0.0, 1.0].
        per_feature: Per-feature breakdown, one entry per unique label.
    """

    dataset_metadata: DatasetMetadata
    total_count: int
    global_mean_ms: float
    global_min_ms: float
    global_max_ms: float
    global_error_rate: float
    per_feature: tuple[FeatureStats, ...]


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
