# JTL Analyzer

A multi-agent system that analyzes JMeter test results (.jtl files) and
generates Excel reports. Supports analysis of individual test runs and
comparison across multiple runs (in future milestones).

## Current state: Milestone 1 complete, plus Milestone 2.5 (REST API)

The analysis core is complete. The loader accepts both CSV-format and
XML-format JTL files (auto-detected from the file's first bytes) and supports
warmup exclusion. Four specialist agents run on top of it:

- **statistician**: global and per-feature count, mean, p50/p90/p95/p99,
  standard deviation, max, throughput, and error rate. Global throughput uses
  the full run duration; per-feature throughput uses each feature's own
  active window.
- **errors**: per-feature response code distribution, sorted by frequency.
- **anomalies**: upper-tail response-time outliers per feature via the IQR
  method (threshold `Q3 + 1.5 * IQR`).
- **trends**: temporal degradation windows per feature using 1-minute bins.

The system is reachable through two interfaces: the CLI (`analyze`) and a
REST API (Milestone 2.5, see below). Milestone 2 (the comparator agent) is
not yet implemented.

## Architecture

A system of specialized agents coordinated by a planner agent. Core rule:

- **Specialist agents** (loader, statistician, errors, anomalies, trends; in
  future milestones: comparator, visualizer, reporter) are **deterministic
  Python code**. They do not invoke LLMs. They produce dataclasses with
  numbers, lists, and temporal references — never prose. Each specialist
  exposes `run(dataset: NormalizedDataset) -> <X>Report`.
- Only **two roles use LLMs**: the orchestrator (decides which steps to
  execute) and, in future milestones, the narrator (turns findings into
  interpretive prose).
- All LLM interaction goes through the `LLMProvider` abstraction
  (`src/jtl_analyzer/providers/`). **Do not call provider SDKs directly
  from other modules.**

### Orchestrator flow

1. Receives the user's request along with metadata (paths to JTLs, options).
2. Generates a `Plan` — a list of `PlanStep` with dependencies — by invoking
   the LLM in structured JSON mode.
3. Validates the plan: each step must reference a registered agent, and
   dependencies must form an acyclic graph.
4. Executes steps in topological order, passing outputs from previous steps
   as inputs according to declared dependencies.
5. Returns the final structured result.

Currently one intent is supported ("analyze a JTL"), which produces a
five-step plan: `loader` → {`statistician`, `errors`, `anomalies`, `trends`},
where the four specialists depend only on the loader and are independent of
each other. `Orchestrator.analyze` rejects a plan that omits any of the four.
The architecture is ready for additional intents without structural changes.

`warmup_seconds` is never embedded in the LLM-generated plan. The orchestrator
captures it in a closure that overrides the `loader` entry in the execution
registry for that call only, so the parsed `Plan` stays immutable.

### API flow (Milestone 2.5)

The REST API **deliberately bypasses the orchestrator** and calls the
specialist agents directly. The LLM-generated plan is invariant in practice
(loader, then the four specialists), so planning per HTTP request would add
latency, cost, and non-determinism for no benefit.

Two invocation paths coexist, sharing the same agents:

1. **CLI**: user → `cli.py` → `Orchestrator` → LLM-generated plan → agents.
2. **API**: HTTP client → FastAPI endpoint → agents.

This is the only sanctioned place where agents are invoked without the
orchestrator. It is safe precisely because the agents are deterministic and
independently tested; the orchestrator adds routing, not correctness.

### LLM provider abstraction

All providers implement `LLMProvider` (in `providers/base.py`):

```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse: ...
```

For `json_mode=True`, the most portable implementation across providers is
prompt engineering: instruct the model to respond with valid JSON only and
parse the response. Providers offer native APIs for structured output
(Anthropic via tool use, OpenAI via `response_format`, Gemini via
`response_mime_type`) that can be incorporated in later milestones if prompt
engineering proves unstable.

### REST API layer (Milestone 2.5)

`src/jtl_analyzer/api/` is a peer of `agents/`, `core/`, `providers/`, and
`i18n/`. It contains no analysis logic — only HTTP concerns.

| Module | Responsibility |
| --- | --- |
| `main.py` | FastAPI app, size middleware, exception handlers, endpoints |
| `dependencies.py` | Shared upload → temp file → `NormalizedDataset` dependency |
| `models.py` | Pydantic HTTP models and `FileTooLargeError` |
| `serialization.py` | Domain reports → JSON-serializable dicts |

**Endpoints**: `GET /health`; `POST /analyze` (all four agents); and
`POST /analyze/{statistician,errors,anomalies,trends}` for a single agent.
All analysis endpoints take `multipart/form-data` with a required `file`
field and an optional `warmup_seconds` float. Swagger UI is at `/docs`,
ReDoc at `/redoc`.

**Uploads become temp files.** `loader.run` takes a path (`_detect_format`
sniffs bytes by path; `lxml.iterparse` streams from a path), so the
dependency spools the upload to a temp file in chunks and deletes it in a
`finally`. It also rewrites `metadata.file_path` to the client's own filename,
so responses never disclose server-side temp paths.

**Upload size is enforced in two layers.** Middleware rejects on the
`Content-Length` header before the body is read; the spooling loop
independently counts bytes. Both are required — `Content-Length` is absent
under chunked transfer encoding, so the middleware alone is bypassable.

**No Pydantic models for analysis responses.** Mirroring the ten nested
report dataclasses would duplicate `core/models.py` and drift from it.
`core/models.py` is the single source of truth; `api/serialization.py` is the
boundary that renders it as JSON. Only `HealthResponse` and `ErrorResponse`
are typed.

**Error mapping** (uniform body `{"error": "<code>", "message": "<detail>"}`,
except FastAPI's own 422 format):

| Condition | Status | `error` |
| --- | --- | --- |
| `InvalidJTLError` — unparseable, missing columns, or warmup excludes all samples | 400 | `invalid_jtl` |
| `Content-Length` over limit, or spooled bytes over limit | 413 | `file_too_large` |
| Missing `file` field, or invalid `warmup_seconds` | 422 | *(FastAPI default)* |
| Any unhandled exception | 500 | `internal_error` |

The 500 handler logs the traceback via `logger.exception` and returns only a
fixed generic message — never a stack trace, path, or exception text.

**Config is separate.** `load_api_config()` reads only `API_HOST`,
`API_PORT`, and `API_MAX_UPLOAD_MB`. It deliberately does not reuse
`load_config()`, which raises when a provider API key is absent: the API
never calls an LLM and must start without LLM credentials.

### Internationalization (i18n)

The system is designed to support multiple output languages without code
changes outside dedicated modules.

- **Code, identifiers, comments, docstrings, and logs**: always in English.
- **User-facing output** (CLI messages, Excel report labels, narrator prose):
  driven by an `output_language` setting (default: `"en"`).
- Static UI strings live in `src/jtl_analyzer/i18n/` as per-language
  modules (`en.py`, `es.py`, ...). Modules outside `i18n/` import message
  keys, never hardcoded user-facing strings.
- The narrator agent (future milestone) receives `output_language` as a
  parameter and instructs the LLM to respond in that language.

Only English is implemented so far. Spanish and others will be added when
there is enough output surface to justify it.

API error `message` fields go through the catalog; API error `error` codes do
not — they are stable machine-readable identifiers, not user-facing prose, and
must never be localized.

## Stack

- Python 3.13+
- pandas (data manipulation)
- lxml (streaming XML parsing for large JTL files)
- pydantic (LLM plan validation, API request/response models)
- FastAPI + uvicorn (REST API), python-multipart (file uploads)
- pytest, httpx (testing; `httpx` backs FastAPI's `TestClient`)
- python-dotenv (loads `.env`)
- Official LLM provider SDKs (`anthropic`; `openai` and
  `google-generativeai` in future milestones)
- pip + venv (environment management)

## Conventions

- **Naming**: modules and files in `snake_case`, classes in `PascalCase`,
  functions and variables in `snake_case`.
- **Type hints**: required on every public function and class method. Use
  modern syntax (`list[str]` instead of `List[str]`, `X | None` instead of
  `Optional[X]`).
- **Data models**: use `@dataclass(frozen=True)` for reports and immutable
  models. Reserve pydantic for cases that need validation from external JSON
  (parsing the plan returned by the LLM, for example).
- **Errors**: define domain-specific exceptions in `core/exceptions.py`
  (`InvalidJTLError`, `PlanValidationError`, `ProviderError`, ...). Do not
  return `None` or `False` to signal failure.
- **Logging**: use the standard `logging` module, configured in `config.py`.
  Do not use `print()` except in `cli.py`.
- **Docstrings**: Google style, required on public classes and on the
  public API of each module.

## Commands

```bash
# Initial setup
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux/macOS
pip install -e ".[dev]"

# Configuration
cp .env.example .env
# Edit .env and set your API keys

# Run all tests
pytest

# Run a specific test file
pytest tests/test_loader.py -v

# Run the CLI against a sample JTL
python -m jtl_analyzer.cli analyze tests/fixtures/small_clean.jtl
python -m jtl_analyzer.cli analyze tests/fixtures/small_clean.jtl --warmup 30

# Start the REST API (reads API_HOST / API_PORT from .env)
python -m jtl_analyzer.cli serve
python -m jtl_analyzer.cli serve --host 127.0.0.1 --port 8000 --reload

# ...or run uvicorn directly
uvicorn jtl_analyzer.api.main:app --host 0.0.0.0 --port 8000

# Call the API
curl http://localhost:8000/health
curl -X POST -F "file=@tests/fixtures/small_clean.jtl" \
     -F "warmup_seconds=30" http://localhost:8000/analyze
```

## Environment variables

Defined in `.env` (never commit it). See `.env.example` for the template.

- `LLM_PROVIDER`: `"anthropic"` | `"openai"` | `"gemini"` (default: `"anthropic"`)
- `LLM_MODEL`: model identifier
- `OUTPUT_LANGUAGE`: ISO 639-1 code, e.g. `"en"`, `"es"` (default: `"en"`)
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`: as required by
  the configured provider

API-only settings (read by `load_api_config()`; the API needs none of the
variables above, since it never calls an LLM):

- `API_HOST`: bind address (default: `0.0.0.0`)
- `API_PORT`: listen port (default: `8000`)
- `API_MAX_UPLOAD_MB`: maximum upload size in MB (default: `200`)

## Important contribution rules

1. **Never** call a provider SDK directly outside `providers/`. Always go
   through `LLMProvider`.
2. **Never** invoke an LLM from a specialist agent. If a feature seems to
   need "interpretation", the specialist produces structured data and an
   LLM-using agent (orchestrator or, in the future, narrator) interprets it.
3. **Every new agent** must (a) register with the orchestrator under a
   unique name, (b) consume and produce dataclasses defined in
   `core/models.py`, and (c) have unit tests.
4. **Test specialists first**: they are pure code, trivially testable.
   Orchestrator tests must mock `LLMProvider` to avoid external calls. API
   tests need no provider mock at all — if one becomes necessary, the API has
   picked up an LLM dependency it should not have.
5. **Never commit** large JTL files (`samples/` is gitignored), `.env`,
   `.venv/`, or generated outputs (final Excel files).
6. **No hardcoded user-facing strings** outside `i18n/`. CLI messages,
   exception messages shown to users, API error `message` fields, and report
   labels go through the message catalog. API `error` codes are the exception:
   they are machine-readable identifiers and must stay stable and unlocalized.
7. **The API layer holds no analysis logic.** Anything in
   `src/jtl_analyzer/api/` that computes a metric belongs in an agent instead.
   The API's job is transport: parse the request, call agents, serialize the
   result, and map errors to status codes.

## Roadmap

- ~~**Milestone 1**: add trends, anomalies, and errors agents. Extend the
  statistician (percentiles, throughput). XML JTL support and warmup
  exclusion in the loader.~~ **Done.** (`OpenAIProvider` and
  `GeminiProvider` remain outstanding; only `AnthropicProvider` is
  implemented.)
- ~~**Milestone 2.5**: expose the system as a REST API (FastAPI), bypassing
  the orchestrator.~~ **Done.**
- **Milestone 2**: comparator agent (multi-JTL) with statistical
  significance tests. When added, expose it as a new API endpoint accepting
  multiple uploads — and extend the orchestrator's prompt and registry, since
  the CLI reaches it through the LLM-generated plan.
- **Milestone 3**: visualizer and reporter with the Excel template embedded
  as a package resource.
- **Milestone 4**: narrator agent. Streamlit interface for conversational /
  granular mode. Spanish translation of the i18n catalog.

Note on the narrator: it uses an LLM, so it belongs on the orchestrator path,
not the API's direct-agent path. Exposing it over HTTP means either a separate
endpoint that accepts a provider config, or routing that one endpoint through
the orchestrator — the current "API never calls an LLM" invariant would need
an explicit, documented exception rather than quiet erosion.