"""Provider factory: instantiate the configured LLM provider."""

from jtl_analyzer.core.exceptions import JTLAnalyzerError
from jtl_analyzer.providers.base import LLMProvider


def create_provider(provider_name: str, api_key: str, model: str) -> LLMProvider:
    """Instantiate and return the LLM provider for the given name.

    Args:
        provider_name: One of ``"anthropic"``, ``"openai"``, ``"gemini"``.
        api_key: API key for the provider.
        model: Model identifier forwarded to the provider.

    Returns:
        A concrete ``LLMProvider`` instance.

    Raises:
        JTLAnalyzerError: If ``provider_name`` is not supported in this milestone.
    """
    if provider_name == "anthropic":
        from jtl_analyzer.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=api_key, model=model)

    raise JTLAnalyzerError(
        f"Provider '{provider_name}' is not yet implemented. "
        "Supported in Milestone 0: anthropic."
    )
