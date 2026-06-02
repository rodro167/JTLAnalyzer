"""Orchestrator agent: generate a plan via LLM and execute specialist agents."""

import logging
from collections import defaultdict
from typing import Any, Callable, cast

from pydantic import ValidationError

from jtl_analyzer.agents import anomalies as anomalies_module
from jtl_analyzer.agents import errors as errors_module
from jtl_analyzer.agents import loader as loader_module
from jtl_analyzer.agents import statistician as statistician_module
from jtl_analyzer.core.exceptions import PlanValidationError, ProviderError
from jtl_analyzer.core.models import (
    AnalysisResult,
    AnomaliesReport,
    ErrorsReport,
    NormalizedDataset,
    Plan,
    PlanStep,
    StatsReport,
)
from jtl_analyzer.providers.base import LLMProvider, Message

logger = logging.getLogger(__name__)

# Type alias for agent callable: (params, dependency-outputs) → result
AgentFn = Callable[[dict[str, Any], dict[str, Any]], Any]


# ---------------------------------------------------------------------------
# Agent wrapper functions — bridge plan params/inputs to agent signatures.
# ---------------------------------------------------------------------------

def _run_loader(params: dict[str, Any], inputs: dict[str, Any]) -> NormalizedDataset:
    return loader_module.run(params["file_path"])


def _run_statistician(params: dict[str, Any], inputs: dict[str, Any]) -> StatsReport:
    dataset: NormalizedDataset = next(iter(inputs.values()))
    return statistician_module.run(dataset)


def _run_errors(params: dict[str, Any], inputs: dict[str, Any]) -> ErrorsReport:
    dataset: NormalizedDataset = next(iter(inputs.values()))
    return errors_module.run(dataset)


def _run_anomalies(params: dict[str, Any], inputs: dict[str, Any]) -> AnomaliesReport:
    dataset: NormalizedDataset = next(iter(inputs.values()))
    return anomalies_module.run(dataset)


# ---------------------------------------------------------------------------
# Default registry and registration API.
#
# Design decision: DEFAULT_REGISTRY is a module-level mutable dict, and
# register_agent() mutates it. Milestone 1 agents call register_agent() from
# their own modules; the application entry point (cli.py) imports those
# modules to trigger registration. The Orchestrator class accepts an
# optional registry override for testing without touching the global state.
# Adding a new agent requires zero changes to Orchestrator's core logic.
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY: dict[str, AgentFn] = {
    "loader": _run_loader,
    "statistician": _run_statistician,
    "errors": _run_errors,
    "anomalies": _run_anomalies,
}


def register_agent(name: str, fn: AgentFn) -> None:
    """Add or replace an agent in the default registry.

    Args:
        name: Unique agent name the orchestrator resolves at plan execution time.
        fn: Callable with signature ``(params: dict, inputs: dict) -> Any``.
            ``params`` comes from ``PlanStep.params``; ``inputs`` is a mapping
            from each dependency's ``step_id`` to its output.
    """
    DEFAULT_REGISTRY[name] = fn


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------

class Orchestrator:
    """Plans and executes a pipeline of specialist agents.

    The orchestrator asks the LLM to produce a ``Plan`` (a list of
    ``PlanStep`` objects with dependency declarations), validates the plan,
    and executes steps in topological order, threading each step's output
    into its dependents' ``inputs`` dict.

    Args:
        provider: LLM provider used for plan generation.
        registry: Agent registry mapping names to callables. Defaults to
            ``DEFAULT_REGISTRY`` (loader + statistician). Pass a custom dict
            in tests to avoid side-effects on the global registry.
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: dict[str, AgentFn] | None = None,
    ) -> None:
        self._provider = provider
        self._registry: dict[str, AgentFn] = (
            registry if registry is not None else dict(DEFAULT_REGISTRY)
        )

    def analyze(self, file_path: str) -> AnalysisResult:
        """Analyze a JTL file end-to-end and return a combined AnalysisResult.

        Args:
            file_path: Path to the JTL file to analyze.

        Returns:
            An ``AnalysisResult`` containing the ``StatsReport``, ``ErrorsReport``,
            and ``AnomaliesReport`` from their respective agents.

        Raises:
            ProviderError: If the LLM call fails or returns non-JSON output.
            PlanValidationError: If the plan references an unknown agent, a
                missing ``step_id``, contains a dependency cycle, or omits any
                of the three required agents (statistician, errors, anomalies).
        """
        plan = self._generate_plan(file_path)
        self._validate_plan(plan)
        results = self._execute_plan(plan)

        stats: StatsReport | None = None
        errors: ErrorsReport | None = None
        anomalies: AnomaliesReport | None = None
        for step in plan.steps:
            if step.agent == "statistician":
                stats = cast(StatsReport, results[step.step_id])
            elif step.agent == "errors":
                errors = cast(ErrorsReport, results[step.step_id])
            elif step.agent == "anomalies":
                anomalies = cast(AnomaliesReport, results[step.step_id])

        missing = [
            name
            for name, val in [("statistician", stats), ("errors", errors), ("anomalies", anomalies)]
            if val is None
        ]
        if missing:
            raise PlanValidationError(
                f"Plan is missing required agent(s): {', '.join(missing)}. "
                "A full analysis requires loader, statistician, errors, and anomalies steps."
            )

        return AnalysisResult(
            stats=cast(StatsReport, stats),
            errors=cast(ErrorsReport, errors),
            anomalies=cast(AnomaliesReport, anomalies),
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _generate_plan(self, file_path: str) -> Plan:
        messages = self._build_prompt(file_path)
        try:
            response = self._provider.complete(messages, json_mode=True)
        except ProviderError:
            raise
        try:
            plan = Plan.model_validate_json(response.content)
        except ValidationError as exc:
            raise PlanValidationError(
                f"LLM returned a plan with invalid schema: {exc}"
            ) from exc
        logger.debug("Plan generated: %d steps", len(plan.steps))
        return plan

    def _build_prompt(self, file_path: str) -> list[Message]:
        agents_list = "\n".join(f"- {name}" for name in sorted(self._registry))
        schema = (
            '{"steps": [{"step_id": "string", "agent": "string", '
            '"depends_on": ["step_id", ...], "params": {...}}]}'
        )
        return [
            Message(
                role="system",
                content=(
                    "You are a planning agent for a JMeter test result analysis system.\n\n"
                    f"Available agents:\n{agents_list}\n\n"
                    "Agent contracts:\n"
                    '- loader: loads a JTL file. Required params: {"file_path": "<path>"}\n'
                    "- statistician: computes performance statistics (mean, percentiles, "
                    "throughput) from the loaded dataset. Depends on: loader.\n"
                    "- errors: computes per-feature response code distribution from the loaded "
                    "dataset. Depends on: loader. Independent of statistician.\n"
                    "- anomalies: detects upper-tail response-time outliers per feature using "
                    "the IQR method. Depends on: loader. Independent of statistician and errors.\n\n"
                    "A full analysis requires all four agents. "
                    "statistician, errors, and anomalies all depend on loader and run independently.\n\n"
                    f"Respond with a JSON plan matching this schema:\n{schema}\n"
                    "Output valid JSON only."
                ),
            ),
            Message(
                role="user",
                content=f"Analyze the JTL file at: {file_path}",
            ),
        ]

    def _validate_plan(self, plan: Plan) -> None:
        step_ids = {s.step_id for s in plan.steps}
        for step in plan.steps:
            if step.agent not in self._registry:
                raise PlanValidationError(
                    f"Plan references unknown agent '{step.agent}'. "
                    f"Registered agents: {sorted(self._registry)}"
                )
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise PlanValidationError(
                        f"Step '{step.step_id}' depends on unknown step '{dep}'"
                    )
        _topological_sort(plan.steps)  # raises PlanValidationError if cyclic

    def _execute_plan(self, plan: Plan) -> dict[str, Any]:
        ordered = _topological_sort(plan.steps)
        results: dict[str, Any] = {}
        for step in ordered:
            inputs = {dep: results[dep] for dep in step.depends_on}
            logger.debug("Executing step '%s' via agent '%s'", step.step_id, step.agent)
            results[step.step_id] = self._registry[step.agent](step.params, inputs)
        return results


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------

def _topological_sort(steps: list[PlanStep]) -> list[PlanStep]:
    """Return steps in topological order.

    Raises:
        PlanValidationError: If the dependency graph contains a cycle.
    """
    index = {s.step_id: s for s in steps}
    in_degree: dict[str, int] = {s.step_id: 0 for s in steps}
    dependents: dict[str, list[str]] = defaultdict(list)

    for s in steps:
        for dep in s.depends_on:
            in_degree[s.step_id] += 1
            dependents[dep].append(s.step_id)

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    ordered: list[PlanStep] = []

    while queue:
        sid = queue.pop(0)
        ordered.append(index[sid])
        for child in dependents[sid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(ordered) != len(steps):
        raise PlanValidationError("Plan dependency graph contains a cycle")

    return ordered
