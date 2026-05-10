"""Domain-specific exceptions for JTL Analyzer."""


class JTLAnalyzerError(Exception):
    """Base class for all JTL Analyzer domain exceptions."""


class InvalidJTLError(JTLAnalyzerError):
    """Raised when a JTL file is missing, unreadable, or lacks required columns."""


class PlanValidationError(JTLAnalyzerError):
    """Raised when a plan references an unknown agent or its dependency graph contains a cycle."""


class ProviderError(JTLAnalyzerError):
    """Raised when an LLM provider call fails (network error, auth failure, or malformed response)."""


class AgentError(JTLAnalyzerError):
    """Raised when a specialist agent encounters an unexpected failure during execution."""
