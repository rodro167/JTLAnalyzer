"""Anthropic Claude provider implementation."""

import json
import logging
import re

import anthropic

from jtl_analyzer.core.exceptions import ProviderError
from jtl_analyzer.providers.base import LLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)

# Matches an optional ```json or ``` opening fence, captures content, closing fence.
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")

# TODO: Migrate to native structured-output APIs (Anthropic tool use,
# OpenAI response_format, Gemini response_mime_type) once the LLMProvider
# abstraction is extended to carry a provider-agnostic schema descriptor.
# That eliminates this class of bug at the source instead of patching
# responses after the fact. Target: Milestone 1+.


def _strip_code_fences(text: str) -> str:
    """Return *text* with any surrounding Markdown code fence removed.

    Handles the following cases produced by LLMs when asked for JSON:

    * ````` ```json\\n{...}\\n``` ````` — fence with language tag
    * ````` ```\\n{...}\\n``` ````` — fence without language tag
    * ``{...}`` — already plain JSON, returned unchanged
    * Any amount of whitespace before/after the fence or the JSON itself

    If no fence is found the input is returned stripped of leading/trailing
    whitespace. If a fence is found but the inner content is empty, an empty
    string is returned (the caller's JSON parser will raise from there).
    """
    match = _CODE_FENCE_RE.search(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic Claude API.

    Args:
        api_key: Anthropic API key.
        model: Model identifier (e.g. ``"claude-sonnet-4-6"``).
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
                only, strips any Markdown code fences from the response, and
                validates that the result parses as JSON.

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
            content = _strip_code_fences(content)
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
