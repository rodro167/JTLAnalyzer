"""LLM provider abstraction: Message, LLMResponse, and the LLMProvider ABC."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from jtl_analyzer.core.exceptions import ProviderError  # noqa: F401  (re-exported for callers)


@dataclass(frozen=True)
class Message:
    """A single message in a conversation with an LLM.

    Attributes:
        role: The speaker role — one of ``"system"``, ``"user"``, or ``"assistant"``.
        content: The text content of the message.
    """

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """The response returned by an LLM provider after a completion call.

    Attributes:
        content: The generated text.
        model: The model identifier echoed back by the provider.
        input_tokens: Number of tokens in the prompt (for cost tracking).
        output_tokens: Number of tokens in the completion (for cost tracking).
    """

    content: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    """Abstract base class that all LLM provider implementations must satisfy.

    Concrete providers (e.g. ``AnthropicProvider``) live in this package and are
    the only modules permitted to import provider SDKs directly.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a list of messages to the LLM and return the completion.

        Args:
            messages: Conversation history, including the current prompt.
            max_tokens: Upper bound on the number of tokens to generate.
            temperature: Sampling temperature; 0.0 for deterministic output.
            json_mode: When True, instruct the provider to return valid JSON only.

        Returns:
            The model's response wrapped in an ``LLMResponse``.

        Raises:
            ProviderError: If the underlying API call fails for any reason.
        """
