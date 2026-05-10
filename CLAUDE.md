# JTL Analyzer

A multi-agent system that analyzes JMeter test results (.jtl files) and
generates Excel reports. Supports analysis of individual test runs and
comparison across multiple runs (in future milestones).

## Current state: Milestone 0

Minimal working core. Only the orchestrator, the loader, and a basic
statistician are implemented. The system accepts a single CSV-format JTL
file and produces basic metrics (count, average, error rate) globally and
per feature.

## Architecture

A system of specialized agents coordinated by a planner agent. Core rule:

- **Specialist agents** (loader, statistician; in future milestones: trends,
  anomalies, errors, comparator, visualizer, reporter) are **deterministic
  Python code**. They do not invoke LLMs. They produce dataclasses with
  numbers, lists, and temporal references — never prose.
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

In Milestone 0, only one intent is supported ("analyze a JTL"), which
produces a canonical two-step plan: `loader` → `statistician`. The
architecture is ready for additional intents without structural changes.

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

For `json_mode=True`, the most portable implementation across providers in
Milestone 0 is prompt engineering: instruct the model to respond with valid
JSON only and parse the response. Providers offer native APIs for structured
output (Anthropic via tool use, OpenAI via `response_format`, Gemini via
`response_mime_type`) that can be incorporated in later milestones if prompt
engineering proves unstable.

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

In Milestone 0 only English is implemented. Spanish and others will be added
when there is enough output surface to justify it.

## Stack

- Python 3.13+
- pandas (data manipulation)
- pytest (testing)
- python-dotenv (loads `.env`)
- Official LLM provider SDKs (`anthropic` in Milestone 0; `openai` and
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
```

## Environment variables

Defined in `.env` (never commit it). See `.env.example` for the template.

- `LLM_PROVIDER`: `"anthropic"` | `"openai"` | `"gemini"` (default: `"anthropic"`)
- `LLM_MODEL`: model identifier
- `OUTPUT_LANGUAGE`: ISO 639-1 code, e.g. `"en"`, `"es"` (default: `"en"`)
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`: as required by
  the configured provider

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
   Orchestrator tests must mock `LLMProvider` to avoid external calls.
5. **Never commit** large JTL files (`samples/` is gitignored), `.env`,
   `.venv/`, or generated outputs (final Excel files).
6. **No hardcoded user-facing strings** outside `i18n/`. CLI messages,
   exception messages shown to users, and report labels go through the
   message catalog.

## Roadmap

- **Milestone 1**: implement `OpenAIProvider` and `GeminiProvider`. Add
  trends, anomalies, and errors agents. Extend the statistician
  (percentiles, throughput).
- **Milestone 2**: comparator agent (multi-JTL) with statistical
  significance tests.
- **Milestone 3**: visualizer and reporter with the Excel template embedded
  as a package resource.
- **Milestone 4**: narrator agent. Streamlit interface for conversational /
  granular mode. Spanish translation of the i18n catalog.