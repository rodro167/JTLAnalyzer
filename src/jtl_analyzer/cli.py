"""CLI entry point: the ``jtl-analyzer analyze`` command."""

import argparse
import sys

from jtl_analyzer.agents.orchestrator import Orchestrator
from jtl_analyzer.config import load_config
from jtl_analyzer.core.models import AnalysisResult
from jtl_analyzer.i18n import get_message
from jtl_analyzer.providers.factory import create_provider


def _print_report(result: AnalysisResult, lang: str) -> None:
    report = result.stats
    print(get_message("REPORT_HEADER", lang))
    print(get_message("REPORT_FILE", lang, file_path=report.dataset_metadata.file_path))
    print(get_message("REPORT_DURATION", lang, duration=report.dataset_metadata.duration_seconds))
    print(get_message("REPORT_TOTAL_SAMPLES", lang, count=report.total_count))
    print(get_message("REPORT_GLOBAL_MEAN", lang, mean=report.global_mean_ms))
    print(get_message("REPORT_GLOBAL_P90", lang, p90=report.global_p90_ms))
    print(get_message("REPORT_GLOBAL_P95", lang, p95=report.global_p95_ms))
    print(get_message("REPORT_GLOBAL_MAX", lang, max=report.global_max_ms))
    print(get_message("REPORT_GLOBAL_THROUGHPUT", lang, throughput=report.global_throughput))
    print(get_message("REPORT_GLOBAL_ERROR_RATE", lang, rate=report.global_error_rate))
    print(get_message("REPORT_FEATURES_HEADER", lang))
    print(get_message("REPORT_FEATURES_TABLE_HEADER", lang))
    for fs in report.per_feature:
        print(
            get_message(
                "REPORT_FEATURE_ROW",
                lang,
                name=fs.name,
                count=fs.count,
                mean=fs.mean_ms,
                p90=fs.p90_ms,
                p95=fs.p95_ms,
                max=fs.max_ms,
                throughput=fs.throughput,
                rate=fs.error_rate,
            )
        )
        breakdowns = result.errors.codes_by_feature.get(fs.name, ())
        if breakdowns:
            print(get_message("REPORT_RESPONSE_CODES_HEADER", lang))
            for bd in breakdowns:
                print(
                    get_message(
                        "REPORT_RESPONSE_CODE_ROW",
                        lang,
                        code=bd.code,
                        percentage=bd.percentage,
                        count=bd.count,
                    )
                )
        fa = result.anomalies.by_feature.get(fs.name)
        if fa is not None:
            if fa.insufficient_data:
                print(get_message("REPORT_ANOMALIES_INSUFFICIENT", lang))
            elif fa.anomaly_count == 0:
                print(get_message("REPORT_ANOMALIES_NONE", lang, threshold=fa.threshold_ms))
            else:
                print(get_message("REPORT_ANOMALIES_HEADER", lang, count=fa.anomaly_count, threshold=fa.threshold_ms))
                shown = fa.samples[:5]
                for sample in shown:
                    print(
                        get_message(
                            "REPORT_ANOMALY_ROW",
                            lang,
                            timestamp=sample.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            elapsed=sample.elapsed_ms,
                            factor=sample.deviation_factor,
                            code=sample.response_code,
                        )
                    )
                remaining = len(fa.samples) - len(shown)
                if remaining > 0:
                    print(get_message("REPORT_ANOMALIES_MORE", lang, count=remaining))


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
            result = orchestrator.analyze(args.file)
            _print_report(result, config.output_language)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
