"""Application configuration: load from environment variables and validate."""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from jtl_analyzer.core.exceptions import JTLAnalyzerError
from jtl_analyzer.i18n import get_message

logger = logging.getLogger(__name__)

_PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


@dataclass(frozen=True)
class Config:
    """Application configuration derived from environment variables.

    Attributes:
        llm_provider: Name of the active LLM provider (e.g. ``"anthropic"``).
        llm_model: Model identifier forwarded to the provider.
        output_language: ISO 639-1 code controlling user-facing output language.
        api_key: API key for the active provider, sourced from the environment.
    """

    llm_provider: str
    llm_model: str
    output_language: str
    api_key: str


def load_config() -> Config:
    """Load and validate application configuration from the environment.

    Reads a ``.env`` file in the working directory via ``python-dotenv``,
    then validates that the chosen provider is known and its API key is present.

    Returns:
        A validated, frozen ``Config`` instance.

    Raises:
        JTLAnalyzerError: If the provider name is unknown or its API key is absent.
    """
    load_dotenv()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", "claude-opus-4-5")
    language = os.getenv("OUTPUT_LANGUAGE", "en")

    key_env_var = _PROVIDER_KEY_ENV.get(provider)
    if key_env_var is None:
        raise JTLAnalyzerError(get_message("ERROR_UNKNOWN_PROVIDER", provider=provider))

    api_key = os.getenv(key_env_var, "")
    if not api_key:
        raise JTLAnalyzerError(
            get_message("ERROR_MISSING_API_KEY", provider=provider, var_name=key_env_var)
        )

    logger.info("Config loaded: provider=%s model=%s lang=%s", provider, model, language)
    return Config(llm_provider=provider, llm_model=model, output_language=language, api_key=api_key)
