"""Shared FastAPI dependencies for the analysis endpoints.

All five analysis endpoints accept the same multipart request and need the same
preparation: enforce the upload size cap, spool the upload to a real file on
disk, load it, then clean up. That sequence lives here once, as a ``yield``
dependency so cleanup is guaranteed even when a handler raises.

A temporary file is required because ``agents.loader.run`` takes a path:
``_detect_format`` opens the file by path to sniff its first bytes, and
``lxml.etree.iterparse`` streams from a path. Spooling in chunks also keeps a
large upload off the heap and gives the byte-counting size check a place to run.
"""

import logging
import os
import tempfile
from dataclasses import replace
from typing import Annotated, Iterator

from fastapi import Depends, File, Form, UploadFile

from jtl_analyzer.agents import loader as loader_module
from jtl_analyzer.api.models import FileTooLargeError
from jtl_analyzer.config import ApiConfig, load_api_config
from jtl_analyzer.core.models import NormalizedDataset

logger = logging.getLogger(__name__)

# Chunk size for spooling uploads to disk (1 MiB).
_CHUNK_SIZE = 1024 * 1024

# Loaded once at import time; the API's limits do not change at runtime.
_api_config: ApiConfig = load_api_config()


def get_api_config() -> ApiConfig:
    """Return the process-wide API configuration.

    Exposed as a dependency so tests can override the size limit via
    ``app.dependency_overrides`` without mutating the environment.

    Returns:
        The ``ApiConfig`` loaded at import time.
    """
    return _api_config


def _spool_to_temp_file(upload: UploadFile, config: ApiConfig) -> str:
    """Write an upload to a temporary file, enforcing the byte limit as it streams.

    Args:
        upload: The incoming multipart file.
        config: API configuration supplying the size limit.

    Returns:
        Absolute path to the temporary file. The caller owns deletion.

    Raises:
        FileTooLargeError: If the upload exceeds the limit. The partial
            temporary file is removed before raising.
    """
    total = 0
    max_bytes = config.max_upload_bytes
    fd, path = tempfile.mkstemp(suffix=".jtl", prefix="jtl_upload_")
    try:
        with os.fdopen(fd, "wb") as handle:
            while chunk := upload.file.read(_CHUNK_SIZE):
                total += len(chunk)
                if total > max_bytes:
                    # Report the configured limit, not one recomputed from
                    # max_bytes, so the message matches API_MAX_UPLOAD_MB exactly.
                    raise FileTooLargeError(config.max_upload_mb)
                handle.write(chunk)
    except BaseException:
        os.unlink(path)
        raise
    return path


def loaded_dataset(
    file: Annotated[UploadFile, File(description="The JTL file to analyze (CSV or XML format).")],
    config: Annotated[ApiConfig, Depends(get_api_config)],
    warmup_seconds: Annotated[
        float,
        Form(description="Seconds to exclude from the start of the test run.", ge=0.0),
    ] = 0.0,
) -> Iterator[NormalizedDataset]:
    """Load an uploaded JTL file into a ``NormalizedDataset``.

    The temporary file backing the upload is deleted once the endpoint that
    depends on this has finished responding.

    Args:
        file: The uploaded JTL file.
        config: Injected API configuration supplying the upload size limit.
        warmup_seconds: Seconds to exclude from the start of the dataset.

    Yields:
        The loaded, normalized dataset, with ``metadata.file_path`` rewritten to
        the client's uploaded filename rather than the server-side temp path.

    Raises:
        FileTooLargeError: If the upload exceeds the configured size limit.
        InvalidJTLError: If the file is not a parseable JTL, lacks required
            columns, or ``warmup_seconds`` excludes every sample.
    """
    path = _spool_to_temp_file(file, config)
    try:
        dataset = loader_module.run(path, warmup_seconds=warmup_seconds)
        # The loader records the path it read, which here is a temp file that is
        # gone by the time the client sees it — and whose name discloses the
        # server's temp layout. Substitute the client's own filename so the
        # metadata is meaningful to them and the response leaks no local paths.
        # Every report embeds this metadata, so one substitution covers all four.
        yield replace(
            dataset,
            metadata=replace(dataset.metadata, file_path=file.filename or "upload.jtl"),
        )
    finally:
        try:
            os.unlink(path)
        except OSError:  # pragma: no cover - best-effort cleanup
            logger.warning("Could not remove temporary upload %s", path)


# Convenience alias so endpoint signatures stay readable.
LoadedDataset = Annotated[NormalizedDataset, Depends(loaded_dataset)]
