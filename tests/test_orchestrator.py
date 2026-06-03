"""Unit tests for agents/orchestrator.py. LLMProvider is always mocked."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jtl_analyzer.agents.orchestrator import Orchestrator
from jtl_analyzer.core.exceptions import PlanValidationError
from jtl_analyzer.core.models import (
    AnalysisResult,
    AnomaliesReport,
    ErrorsReport,
    StatsReport,
    TrendsReport,
)
from jtl_analyzer.providers.base import LLMProvider, LLMResponse

FIXTURES = Path(__file__).parent / "fixtures"
CLEAN = str(FIXTURES / "small_clean.jtl")


def _mock_provider(plan_json: str) -> LLMProvider:
    """Return a mock LLMProvider whose complete() returns the given JSON string."""
    provider = MagicMock(spec=LLMProvider)
    provider.complete.return_value = LLMResponse(
        content=plan_json,
        model="mock-model",
        input_tokens=10,
        output_tokens=50,
    )
    return provider


def _valid_plan(file_path: str) -> str:
    return json.dumps({
        "steps": [
            {
                "step_id": "load",
                "agent": "loader",
                "depends_on": [],
                "params": {"file_path": file_path},
            },
            {
                "step_id": "analyze",
                "agent": "statistician",
                "depends_on": ["load"],
                "params": {},
            },
            {
                "step_id": "errors",
                "agent": "errors",
                "depends_on": ["load"],
                "params": {},
            },
            {
                "step_id": "anomalies",
                "agent": "anomalies",
                "depends_on": ["load"],
                "params": {},
            },
            {
                "step_id": "trends",
                "agent": "trends",
                "depends_on": ["load"],
                "params": {},
            },
        ]
    })


class TestHappyPath:
    def test_returns_analysis_result(self):
        orch = Orchestrator(_mock_provider(_valid_plan(CLEAN)))
        result = orch.analyze(CLEAN)
        assert isinstance(result, AnalysisResult)

    def test_result_contains_stats_report(self):
        orch = Orchestrator(_mock_provider(_valid_plan(CLEAN)))
        result = orch.analyze(CLEAN)
        assert isinstance(result.stats, StatsReport)

    def test_result_contains_errors_report(self):
        orch = Orchestrator(_mock_provider(_valid_plan(CLEAN)))
        result = orch.analyze(CLEAN)
        assert isinstance(result.errors, ErrorsReport)

    def test_result_contains_anomalies_report(self):
        orch = Orchestrator(_mock_provider(_valid_plan(CLEAN)))
        result = orch.analyze(CLEAN)
        assert isinstance(result.anomalies, AnomaliesReport)

    def test_result_contains_trends_report(self):
        orch = Orchestrator(_mock_provider(_valid_plan(CLEAN)))
        result = orch.analyze(CLEAN)
        assert isinstance(result.trends, TrendsReport)

    def test_stats_report_has_correct_count(self):
        orch = Orchestrator(_mock_provider(_valid_plan(CLEAN)))
        result = orch.analyze(CLEAN)
        assert result.stats.total_count == 6

    def test_provider_called_exactly_once(self):
        provider = _mock_provider(_valid_plan(CLEAN))
        Orchestrator(provider).analyze(CLEAN)
        provider.complete.assert_called_once()

    def test_provider_called_with_json_mode(self):
        provider = _mock_provider(_valid_plan(CLEAN))
        Orchestrator(provider).analyze(CLEAN)
        _, kwargs = provider.complete.call_args
        assert kwargs.get("json_mode") is True


class TestPlanValidation:
    def test_unknown_agent_raises_plan_validation_error(self):
        bad = json.dumps({
            "steps": [
                {
                    "step_id": "load",
                    "agent": "no_such_agent",
                    "depends_on": [],
                    "params": {"file_path": CLEAN},
                }
            ]
        })
        with pytest.raises(PlanValidationError, match="no_such_agent"):
            Orchestrator(_mock_provider(bad)).analyze(CLEAN)

    def test_missing_dependency_step_raises_plan_validation_error(self):
        bad = json.dumps({
            "steps": [
                {
                    "step_id": "analyze",
                    "agent": "statistician",
                    "depends_on": ["ghost_step"],
                    "params": {},
                }
            ]
        })
        with pytest.raises(PlanValidationError):
            Orchestrator(_mock_provider(bad)).analyze(CLEAN)

    def test_cyclic_dependency_raises_plan_validation_error(self):
        cyclic = json.dumps({
            "steps": [
                {
                    "step_id": "a",
                    "agent": "loader",
                    "depends_on": ["b"],
                    "params": {"file_path": CLEAN},
                },
                {
                    "step_id": "b",
                    "agent": "statistician",
                    "depends_on": ["a"],
                    "params": {},
                },
            ]
        })
        with pytest.raises(PlanValidationError, match="cycle"):
            Orchestrator(_mock_provider(cyclic)).analyze(CLEAN)

    def test_invalid_plan_schema_raises_plan_validation_error(self):
        # Valid JSON but wrong schema (missing required fields)
        bad = json.dumps({"not_steps": []})
        with pytest.raises(PlanValidationError):
            Orchestrator(_mock_provider(bad)).analyze(CLEAN)


class TestMissingAgent:
    def test_plan_without_errors_step_raises(self):
        plan_no_errors = json.dumps({
            "steps": [
                {"step_id": "load", "agent": "loader", "depends_on": [], "params": {"file_path": CLEAN}},
                {"step_id": "analyze", "agent": "statistician", "depends_on": ["load"], "params": {}},
            ]
        })
        with pytest.raises(PlanValidationError, match="errors"):
            Orchestrator(_mock_provider(plan_no_errors)).analyze(CLEAN)

    def test_plan_without_statistician_step_raises(self):
        plan_no_stats = json.dumps({
            "steps": [
                {"step_id": "load", "agent": "loader", "depends_on": [], "params": {"file_path": CLEAN}},
                {"step_id": "errors", "agent": "errors", "depends_on": ["load"], "params": {}},
            ]
        })
        with pytest.raises(PlanValidationError, match="statistician"):
            Orchestrator(_mock_provider(plan_no_stats)).analyze(CLEAN)

    def test_plan_without_anomalies_step_raises(self):
        plan_no_anomalies = json.dumps({
            "steps": [
                {"step_id": "load", "agent": "loader", "depends_on": [], "params": {"file_path": CLEAN}},
                {"step_id": "analyze", "agent": "statistician", "depends_on": ["load"], "params": {}},
                {"step_id": "errors", "agent": "errors", "depends_on": ["load"], "params": {}},
            ]
        })
        with pytest.raises(PlanValidationError, match="anomalies"):
            Orchestrator(_mock_provider(plan_no_anomalies)).analyze(CLEAN)

    def test_plan_without_trends_step_raises(self):
        plan_no_trends = json.dumps({
            "steps": [
                {"step_id": "load", "agent": "loader", "depends_on": [], "params": {"file_path": CLEAN}},
                {"step_id": "analyze", "agent": "statistician", "depends_on": ["load"], "params": {}},
                {"step_id": "errors", "agent": "errors", "depends_on": ["load"], "params": {}},
                {"step_id": "anomalies", "agent": "anomalies", "depends_on": ["load"], "params": {}},
            ]
        })
        with pytest.raises(PlanValidationError, match="trends"):
            Orchestrator(_mock_provider(plan_no_trends)).analyze(CLEAN)


class TestRegistry:
    def test_custom_registry_is_isolated_from_default(self):
        """Injecting a custom registry must not mutate DEFAULT_REGISTRY."""
        from jtl_analyzer.agents.orchestrator import DEFAULT_REGISTRY

        custom = dict(DEFAULT_REGISTRY)
        custom["dummy"] = lambda p, i: None
        orch = Orchestrator(_mock_provider(_valid_plan(CLEAN)), registry=custom)
        orch.analyze(CLEAN)

        assert "dummy" not in DEFAULT_REGISTRY

    def test_register_agent_adds_to_default_registry(self):
        from jtl_analyzer.agents.orchestrator import DEFAULT_REGISTRY, register_agent

        register_agent("test_agent_xyz", lambda p, i: None)
        assert "test_agent_xyz" in DEFAULT_REGISTRY
        del DEFAULT_REGISTRY["test_agent_xyz"]  # clean up
