"""Loader agent: parse a CSV-format or XML-format JTL file into a NormalizedDataset."""

import logging
import os

import pandas as pd
from lxml import etree

from jtl_analyzer.core.exceptions import InvalidJTLError
from jtl_analyzer.core.models import DatasetMetadata, NormalizedDataset
from jtl_analyzer.i18n import get_message

logger = logging.getLogger(__name__)

# JTL column name → normalized name
_COLUMN_MAP: dict[str, str] = {"timeStamp": "timestamp"}

# Columns that must be present after normalization
REQUIRED_COLUMNS: frozenset[str] = frozenset({"timestamp", "elapsed", "label", "success"})

# XML element tags that represent individual samples
_SAMPLE_TAGS: list[str] = ["httpSample", "sample"]


def _detect_format(file_path: str) -> str:
    """Inspect the first bytes of a file to determine JTL format.

    Returns:
        ``"xml"`` if the file starts with an XML declaration (after optional
        UTF-8 BOM and whitespace); ``"csv"`` otherwise.
    """
    with open(file_path, "rb") as fh:
        raw = fh.read(256)
    if raw.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
        raw = raw[3:]
    return "xml" if raw.lstrip().startswith(b"<?xml") else "csv"


def _parse_csv(file_path: str) -> pd.DataFrame:
    """Parse a CSV-format JTL file into a raw DataFrame.

    Returns the DataFrame as-is from pandas (column names not yet normalized).

    Raises:
        InvalidJTLError: If the file cannot be read or parsed.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as exc:
        raise InvalidJTLError(
            get_message("ERROR_UNREADABLE_FILE", file_path=file_path, reason=str(exc))
        ) from exc


def _parse_xml(file_path: str) -> pd.DataFrame:
    """Parse an XML-format JTL file into a raw DataFrame using streaming.

    Uses ``lxml.etree.iterparse`` for memory-efficient processing of large files.
    Each processed element is cleared immediately to release memory.

    Returns a DataFrame with columns ``timeStamp``, ``elapsed``, ``label``,
    ``success``, and ``responseCode`` (names not yet normalized).

    Raises:
        InvalidJTLError: If the XML is malformed or contains no sample elements.
    """
    rows: list[dict] = []
    try:
        context = etree.iterparse(file_path, events=("end",), tag=_SAMPLE_TAGS)
        for _event, elem in context:
            rows.append(
                {
                    "timeStamp": int(elem.get("ts", 0)),
                    "elapsed": int(elem.get("t", 0)),
                    "label": elem.get("lb", ""),
                    "success": elem.get("s", "false"),
                    "responseCode": elem.get("rc", ""),
                }
            )
            elem.clear()
    except etree.XMLSyntaxError as exc:
        if not rows:
            raise InvalidJTLError(
                get_message("ERROR_MALFORMED_XML", file_path=file_path, reason=str(exc))
            ) from exc
        logger.warning(
            get_message("WARNING_TRUNCATED_FILE", count=len(rows), file_path=file_path)
        )

    if not rows:
        raise InvalidJTLError(get_message("ERROR_EMPTY_DATASET", file_path=file_path))

    return pd.DataFrame(rows)


def run(file_path: str, warmup_seconds: float = 0.0) -> NormalizedDataset:
    """Load and normalize a CSV-format or XML-format JTL file.

    Detects the file format from its first bytes. For CSV files, expects the
    standard JMeter CSV output. For XML files, extracts ``<httpSample>`` and
    ``<sample>`` elements using streaming parsing via ``lxml``.

    Renames ``timeStamp`` to ``timestamp``, converts elapsed to float,
    parses success as bool, and converts the timestamp column to UTC datetimes.
    When ``warmup_seconds`` is greater than zero, all samples whose timestamp
    falls within the first ``warmup_seconds`` from the earliest timestamp in
    the raw dataset are excluded before returning.

    Args:
        file_path: Path to the JTL file (CSV or XML format).
        warmup_seconds: Seconds to exclude from the start of the dataset.
            Defaults to 0.0 (no exclusion).

    Returns:
        A ``NormalizedDataset`` with cleaned data and descriptive metadata.

    Raises:
        InvalidJTLError: If the file is missing, unreadable, malformed, lacks
            any of the required columns (``timestamp``, ``elapsed``, ``label``,
            ``success``), or if ``warmup_seconds`` is large enough to exclude
            all samples.
    """
    if not os.path.exists(file_path):
        raise InvalidJTLError(get_message("ERROR_FILE_NOT_FOUND", file_path=file_path))

    fmt = _detect_format(file_path)
    df = _parse_xml(file_path) if fmt == "xml" else _parse_csv(file_path)

    df = df.rename(columns=_COLUMN_MAP)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise InvalidJTLError(
            get_message("ERROR_MISSING_COLUMNS", columns=", ".join(sorted(missing)))
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["elapsed"] = df["elapsed"].astype(float)
    df["success"] = (
        df["success"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    )

    original_row_count = len(df)

    if warmup_seconds > 0.0:
        cutoff = df["timestamp"].min() + pd.Timedelta(seconds=warmup_seconds)
        df = df[df["timestamp"] >= cutoff]
        if df.empty:
            raise InvalidJTLError(
                get_message(
                    "ERROR_WARMUP_EXCLUDES_ALL",
                    seconds=warmup_seconds,
                    file_path=file_path,
                )
            )

    start_time = df["timestamp"].min().to_pydatetime()
    end_time = df["timestamp"].max().to_pydatetime()

    metadata = DatasetMetadata(
        file_path=os.path.abspath(file_path),
        row_count=len(df),
        start_time=start_time,
        end_time=end_time,
        duration_seconds=(end_time - start_time).total_seconds(),
        warmup_seconds=warmup_seconds,
        original_row_count=original_row_count,
    )

    logger.debug("Loaded %d rows from %s", len(df), file_path)
    return NormalizedDataset(metadata=metadata, data=df)
