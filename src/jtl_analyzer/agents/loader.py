"""Loader agent: parse a CSV-format JTL file into a NormalizedDataset."""

import logging
import os

import pandas as pd

from jtl_analyzer.core.exceptions import InvalidJTLError
from jtl_analyzer.core.models import DatasetMetadata, NormalizedDataset
from jtl_analyzer.i18n import get_message

logger = logging.getLogger(__name__)

# JTL column name → normalized name
_COLUMN_MAP: dict[str, str] = {"timeStamp": "timestamp"}

# Columns that must be present after normalization
REQUIRED_COLUMNS: frozenset[str] = frozenset({"timestamp", "elapsed", "label", "success"})


def run(file_path: str) -> NormalizedDataset:
    """Load and normalize a CSV-format JTL file.

    Renames ``timeStamp`` to ``timestamp``, converts elapsed to float,
    parses success as bool, and converts the timestamp column to UTC datetimes.

    Args:
        file_path: Path to the JTL CSV file.

    Returns:
        A ``NormalizedDataset`` with cleaned data and descriptive metadata.

    Raises:
        InvalidJTLError: If the file is missing, unreadable, or lacks any
            of the required columns (``timestamp``, ``elapsed``, ``label``,
            ``success``).
    """
    if not os.path.exists(file_path):
        raise InvalidJTLError(get_message("ERROR_FILE_NOT_FOUND", file_path=file_path))

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise InvalidJTLError(
            get_message("ERROR_UNREADABLE_FILE", file_path=file_path, reason=str(exc))
        ) from exc

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

    start_time = df["timestamp"].min().to_pydatetime()
    end_time = df["timestamp"].max().to_pydatetime()

    metadata = DatasetMetadata(
        file_path=os.path.abspath(file_path),
        row_count=len(df),
        start_time=start_time,
        end_time=end_time,
        duration_seconds=(end_time - start_time).total_seconds(),
    )

    logger.debug("Loaded %d rows from %s", len(df), file_path)
    return NormalizedDataset(metadata=metadata, data=df)
