"""CLI entry point: the ``jtl-analyzer analyze`` command."""

import argparse
import sys

from jtl_analyzer.agents.orchestrator import Orchestrator
from jtl_analyzer.config import load_config
from jtl_analyzer.core.models import StatsReport
from jtl_analyzer.i18n import get_message
from jtl_analyzer.providers.factory import create_provider


def _print_report(report: StatsReport, lang: str) -> None:
    print(get_message("REPORT_HEADER", lang))
    print(get_message("REPORT_FILE", lang, file_path=report.dataset_metadata.file_path))
    print(get_message("REPORT_DURATION", lang, duration=report.dataset_metadata.duration_seconds))
    print(get_message("REPORT_TOTAL_SAMPLES", lang, count=report.total_count))
    print(get_message("REPORT_GLOBAL_MEAN", lang, mean=report.global_mean_ms))
    print(get_message("REPORT_GLOBAL_MIN", lang, min=report.global_min_ms))
    print(get_message("REPORT_GLOBAL_MAX", lang, max=report.global_max_ms))
    print(get_message("REPORT_GLOBAL_ERROR_RATE", lang, rate=report.global_error_rate))
    print(get_message("REPORT_FEATURES_HEADER", lang))
    for fs in report.per_feature:
        print(
            get_message(
                "REPORT_FEATURE_ROW",
                lang,
                name=fs.name,
                count=fs.count,
                mean=fs.mean_ms,
                min=fs.min_ms,
                max=fs.max_ms,
                rate=fs.error_rate,
            )
        )


def main() -> None:
    """Entry point for the ``jtl-analyzer`` CLI."""
    parser = argparse.ArgumentParser(prog="jtl-analyzer", description="JTL Analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_cmd = sub.add_parser("analyze", help="Analyze a JTL file")
    analyze_cmd.add_argument("file", help="Path to the JTL file")

    args = parser.parse_args()

    if args.command == "analyze":
        try:
            config = load_config()
            provider = create_provider(config.llm_provider, config.api_key, config.llm_model)
            orchestrator = Orchestrator(provider)
            print(get_message("ANALYZING_FILE", config.output_language, file_path=args.file))
            report = orchestrator.analyze(args.file)
            _print_report(report, config.output_language)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()