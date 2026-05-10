"""Anthropic Claude provider implementation."""

import json
import logging

import anthropic

from jtl_analyzer.core.exceptions import ProviderError
from jtl_analyzer.providers.base import LLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic Claude API.

    Args:
        api_key: Anthropic API key.
        model: Model identifier (e.g. ``"claude-opus-4-5"``).
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send messages to Anthropic Claude and return the completion.

        Args:
            messages: Conversation history. ``system`` role messages are
                extracted and merged into a single system prompt.
            max_tokens: Upper bound on generated tokens.
            temperature: Sampling temperature; 0.0 for deterministic output.
            json_mode: When True, appends an instruction to return valid JSON
                only and validates that the response parses as JSON.

        Returns:
            The model response wrapped in an ``LLMResponse``.

        Raises:
            ProviderError: If the API call fails or ``json_mode`` is True and
                the response is not valid JSON.
        """
        system_parts = [m.content for m in messages if m.role == "system"]
        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]

        system: str | anthropic.NotGiven
        if system_parts or json_mode:
            parts = list(system_parts)
            if json_mode:
                parts.append(
                    "Respond with valid JSON only. "
                    "Do not include any text outside the JSON object."
                )
            system = "\n\n".join(parts)
        else:
            system = anthropic.NOT_GIVEN

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=chat_messages,
            )
        except anthropic.APIError as exc:
            raise ProviderError(str(exc)) from exc

        content = response.content[0].text

        if json_mode:
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                raise ProviderError(f"Provider returned non-JSON response: {exc}") from exc

        logger.debug(
            "Anthropic call: model=%s in=%d out=%d",
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
