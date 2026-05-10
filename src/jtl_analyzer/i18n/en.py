"""English message catalog. All user-facing strings for lang='en'."""

# CLI
ANALYZING_FILE = "Analyzing {file_path} ..."

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

# Report labels
REPORT_HEADER = "=== JTL Analysis Report ==="
REPORT_FILE = "File        : {file_path}"
REPORT_DURATION = "Duration    : {duration:.1f}s"
REPORT_TOTAL_SAMPLES = "Samples     : {count}"
REPORT_GLOBAL_MEAN = "Mean        : {mean:.1f} ms"
REPORT_GLOBAL_MIN = "Min         : {min:.1f} ms"
REPORT_GLOBAL_MAX = "Max         : {max:.1f} ms"
REPORT_GLOBAL_ERROR_RATE = "Error rate  : {rate:.1%}"
REPORT_FEATURES_HEADER = "\n--- Per-feature breakdown ---"
REPORT_FEATURE_ROW = (
    "  {name:<30} count={count:>5}  mean={mean:>7.1f}ms"
    "  min={min:>7.1f}ms  max={max:>7.1f}ms  errors={rate:.1%}"
)
