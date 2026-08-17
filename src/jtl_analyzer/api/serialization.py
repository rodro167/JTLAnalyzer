"""Convert domain reports into JSON-serializable dictionaries.

The domain models in ``core.models`` are frozen dataclasses holding values that
``json.dumps`` cannot encode directly: ``datetime`` objects, numpy scalars
produced by pandas aggregations, and occasionally non-finite floats. Each
public function here returns a structure containing only ``dict``, ``list``,
``str``, ``int``, ``float``, ``bool``, and ``None``.

Conversion rules applied by the recursive pass:

- ``datetime`` and ``pandas.Timestamp`` become ISO 8601 strings.
- ``tuple`` becomes ``list``.
- numpy scalars become their Python equivalents.
- Non-finite floats (``NaN``, ``inf``, ``-inf``) become ``None``, since JSON
  has no representation for them.
- ``pandas.DataFrame`` values are dropped.
"""

import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from jtl_analyzer.core.models import (
    AnalysisResult,
    AnomaliesReport,
    ErrorsReport,
    StatsReport,
    TrendsReport,
)

# Sentinel marking values the recursive pass drops from its parent container.
_DROP = object()


def _jsonify(value: Any) -> Any:
    """Recursively convert a value into a JSON-serializable equivalent.

    Returns:
        The converted value, or the module-private ``_DROP`` sentinel for values
        that must be omitted from their containing dict or list.
    """
    # NaTType subclasses datetime, so it must be tested before the date branch
    # below — its .isoformat() returns the string "NaT", not a valid timestamp.
    if value is pd.NaT:
        return None

    # datetime is a subclass of date, so this covers both; pd.Timestamp is a
    # datetime subclass and therefore also handled here.
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, pd.DataFrame):
        return _DROP

    if isinstance(value, np.generic):
        # .item() yields the native Python scalar (np.float64 -> float, etc.).
        return _jsonify(value.item())

    if isinstance(value, float):
        # JSON has no NaN/Infinity literal. Degenerate pandas aggregations can
        # produce these, so map them to null rather than emitting invalid JSON.
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            converted = _jsonify(item)
            if converted is not _DROP:
                result[str(key)] = converted
        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        return [item for item in (_jsonify(v) for v in value) if item is not _DROP]

    return str(value)


def _report_to_dict(report: Any) -> dict[str, Any]:
    """Flatten a frozen dataclass report into a JSON-serializable dict.

    ``asdict`` recurses through nested dataclasses, dicts, lists, and tuples;
    ``_jsonify`` then normalizes the leaf values it leaves untouched.

    Raises:
        TypeError: If ``report`` is not a dataclass instance.
    """
    if not is_dataclass(report) or isinstance(report, type):
        raise TypeError(f"Expected a dataclass instance, got {type(report).__name__}")
    return _jsonify(asdict(report))


def stats_report_to_dict(report: StatsReport) -> dict[str, Any]:
    """Serialize a ``StatsReport`` to a JSON-serializable dict.

    Args:
        report: The report produced by the statistician agent.

    Returns:
        A dict with the report's global metrics, its ``dataset_metadata``, and a
        ``per_feature`` list of per-feature statistics.
    """
    return _report_to_dict(report)


def errors_report_to_dict(report: ErrorsReport) -> dict[str, Any]:
    """Serialize an ``ErrorsReport`` to a JSON-serializable dict.

    Args:
        report: The report produced by the errors agent.

    Returns:
        A dict with ``dataset_metadata`` and ``codes_by_feature``, the latter
        mapping each feature label to a list of response code breakdowns.
    """
    return _report_to_dict(report)


def anomalies_report_to_dict(report: AnomaliesReport) -> dict[str, Any]:
    """Serialize an ``AnomaliesReport`` to a JSON-serializable dict.

    Args:
        report: The report produced by the anomalies agent.

    Returns:
        A dict with ``dataset_metadata`` and ``by_feature``, the latter mapping
        each feature label to its detection results and anomalous samples.
    """
    return _report_to_dict(report)


def trends_report_to_dict(report: TrendsReport) -> dict[str, Any]:
    """Serialize a ``TrendsReport`` to a JSON-serializable dict.

    Args:
        report: The report produced by the trends agent.

    Returns:
        A dict with ``dataset_metadata`` and ``by_feature``, the latter mapping
        each feature label to its detected degradation windows.
    """
    return _report_to_dict(report)


def analysis_result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    """Serialize a full ``AnalysisResult`` to a JSON-serializable dict.

    Args:
        result: The combined output of the four specialist agents.

    Returns:
        A dict with the keys ``stats``, ``errors``, ``anomalies``, and
        ``trends``, each holding that agent's serialized report.
    """
    return {
        "stats": stats_report_to_dict(result.stats),
        "errors": errors_report_to_dict(result.errors),
        "anomalies": anomalies_report_to_dict(result.anomalies),
        "trends": trends_report_to_dict(result.trends),
    }
