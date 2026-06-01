"""Errors agent: compute per-feature response code distribution from a NormalizedDataset."""

import logging

from jtl_analyzer.core.models import ErrorsReport, NormalizedDataset, ResponseCodeBreakdown

logger = logging.getLogger(__name__)


def run(dataset: NormalizedDataset) -> ErrorsReport:
    """Compute per-feature response code counts and percentages.

    Groups samples by feature label, then by response code within each group.
    Relies on the ``responseCode`` column being present in the dataset (populated
    by the loader for both CSV and XML formats).

    Pure pandas — no LLM calls.

    Args:
        dataset: The loaded, cleaned JTL dataset.

    Returns:
        An ``ErrorsReport`` whose ``codes_by_feature`` maps each feature label to
        a tuple of ``ResponseCodeBreakdown`` entries sorted by count descending.
    """
    df = dataset.data
    codes_by_feature: dict[str, tuple[ResponseCodeBreakdown, ...]] = {}

    for label, group in df.groupby("label", sort=True):
        total = len(group)
        breakdowns: list[ResponseCodeBreakdown] = []
        for code, code_group in group.groupby("responseCode", sort=False):
            count = len(code_group)
            breakdowns.append(
                ResponseCodeBreakdown(
                    code=str(code),
                    count=count,
                    percentage=count / total if total > 0 else 0.0,
                )
            )
        breakdowns.sort(key=lambda b: b.count, reverse=True)
        codes_by_feature[str(label)] = tuple(breakdowns)

    logger.debug("Errors: %d features processed", len(codes_by_feature))
    return ErrorsReport(
        dataset_metadata=dataset.metadata,
        codes_by_feature=codes_by_feature,
    )
