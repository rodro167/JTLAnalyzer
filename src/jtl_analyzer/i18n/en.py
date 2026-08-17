"""English message catalog. All user-facing strings for lang='en'."""

# CLI
ANALYZING_FILE = "Analyzing {file_path} ..."
SERVING_API = "Starting JTL Analyzer API on http://{host}:{port} (docs at /docs)"

# Errors shown to users
ERROR_FILE_NOT_FOUND = "File not found: {file_path}"
ERROR_UNREADABLE_FILE = "Could not read '{file_path}': {reason}"
ERROR_MISSING_COLUMNS = "Missing required columns: {columns}"
ERROR_UNKNOWN_PROVIDER = (
    "Unknown LLM provider: '{provider}'. Supported: anthropic, openai, gemini."
)
ERROR_MISSING_API_KEY = (
    "API key for provider '{provider}' is not set. "
    "Add {var_name} to your .env file."
)

ERROR_FILE_TOO_LARGE = "Upload exceeds the maximum size of {max_mb:g} MB"
ERROR_INTERNAL = "An unexpected error occurred"

ERROR_MALFORMED_XML = "Could not parse XML in '{file_path}': {reason}"
ERROR_EMPTY_DATASET = "No sample elements found in '{file_path}'"
WARNING_TRUNCATED_FILE = "File appears truncated; recovered {count} samples from '{file_path}'"
ERROR_WARMUP_EXCLUDES_ALL = "Warmup of {seconds:.1f}s excluded all samples from '{file_path}'"

# Report labels
REPORT_HEADER = "=== JTL Analysis Report ==="
REPORT_FILE = "File        : {file_path}"
REPORT_DURATION = "Duration    : {duration:.1f}s"
REPORT_TOTAL_SAMPLES = "Samples     : {count}"
REPORT_GLOBAL_MEAN = "Mean        : {mean:.1f} ms"
REPORT_GLOBAL_P90 = "p90         : {p90:.1f} ms"
REPORT_GLOBAL_P95 = "p95         : {p95:.1f} ms"
REPORT_GLOBAL_MAX = "Max         : {max:.1f} ms"
REPORT_GLOBAL_THROUGHPUT = "Throughput  : {throughput:.1f} req/s"
REPORT_GLOBAL_ERROR_RATE = "Error rate  : {rate:.1%}"
REPORT_FEATURES_HEADER = "\n--- Per-feature breakdown ---"
REPORT_FEATURES_TABLE_HEADER = (
    "  {:<30}  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9}  {:>7}".format(
        "Feature", "Count", "Mean", "p90", "p95", "Max", "Tput(r/s)", "Errors"
    )
)
REPORT_FEATURE_ROW = (
    "  {name:<30}  {count:>5}  {mean:>7.1f}ms  {p90:>7.1f}ms"
    "  {p95:>7.1f}ms  {max:>7.1f}ms  {throughput:>9.1f}  {rate:>6.1%}"
)
REPORT_RESPONSE_CODES_HEADER = "    Response codes:"
REPORT_RESPONSE_CODE_ROW = "      {code}: {percentage:.1%} ({count})"
REPORT_ANOMALIES_HEADER = "    Anomalies ({count} detected, threshold {threshold:.1f}ms):"
REPORT_ANOMALY_ROW = "      {timestamp} - {elapsed:.1f}ms ({factor:.1f}x over threshold) [{code}]"
REPORT_ANOMALIES_MORE = "      [{count} more]"
REPORT_ANOMALIES_NONE = "    Anomalies: none detected (threshold {threshold:.1f}ms)"
REPORT_ANOMALIES_INSUFFICIENT = "    Anomalies: insufficient data (< 20 samples)"
REPORT_TRENDS_HEADER = "    Degradation windows ({count} detected, reference {reference:.1f}ms):"
REPORT_TREND_WINDOW_ROW = "      {start} to {end} ({duration:.1f}min, median {median:.1f}ms, {factor:.1f}x reference)"
REPORT_TRENDS_NONE = "    Degradation windows: none detected"
REPORT_TRENDS_INSUFFICIENT = "    Degradation windows: insufficient data (< 20 samples)"
