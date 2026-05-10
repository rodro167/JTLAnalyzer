"""Unit tests for providers/anthropic.py: _strip_code_fences and json_mode flow."""

from unittest.mock import MagicMock, patch

import pytest

from jtl_analyzer.core.exceptions import ProviderError
from jtl_analyzer.providers.anthropic import _strip_code_fences
from jtl_analyzer.providers.base import Message


# ---------------------------------------------------------------------------
# _strip_code_fences — pure-function tests, no mocking needed
# ---------------------------------------------------------------------------

class TestStripCodeFences:
    def test_plain_json_returned_unchanged(self):
        payload = '{"key": "value"}'
        assert _strip_code_fences(payload) == payload

    def test_fenced_with_json_language_tag(self):
        raw = '```json\n{"key": "value"}\n```'
        assert _strip_code_fences(raw) == '{"key": "value"}'

    def test_fenced_without_language_tag(self):
        raw = '```\n{"key": "value"}\n```'
        assert _strip_code_fences(raw) == '{"key": "value"}'

    def test_whitespace_before_and_after_fence_stripped(self):
        raw = '  \n```json\n  {"key": "value"}  \n```\n  '
        assert _strip_code_fences(raw) == '{"key": "value"}'

    def test_multiline_json_internal_whitespace_preserved(self):
        raw = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        assert _strip_code_fences(raw) == '{\n  "a": 1,\n  "b": 2\n}'

    def test_empty_string_returns_empty(self):
        assert _strip_code_fences("") == ""

    def test_no_closing_fence_returns_stripped_input(self):
        # Malformed output: opening fence present but no closing fence.
        # The regex does not match; we return the stripped original so
        # the caller's JSON parser raises with a meaningful error.
        raw = "```json\n{}"
        assert _strip_code_fences(raw) == raw.strip()

    def test_text_before_fence_still_extracts_json(self):
        # Some models prepend a prose sentence before the code block.
        raw = 'Here is the plan:\n```json\n{"steps": []}\n```'
        assert _strip_code_fences(raw) == '{"steps": []}'


# ---------------------------------------------------------------------------
# AnthropicProvider.complete() with json_mode — mocked Anthropic client
# ---------------------------------------------------------------------------

def _make_provider(response_text: str):
    """Return an AnthropicProvider whose underlying client is fully mocked."""
    from jtl_analyzer.providers.anthropic import AnthropicProvider

    with patch("jtl_analyzer.providers.anthropic.anthropic.Anthropic"):
        provider = AnthropicProvider(api_key="test-key", model="test-model")

    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=response_text)]
    mock_msg.model = "mock-model"
    mock_msg.usage.input_tokens = 10
    mock_msg.usage.output_tokens = 20
    provider._client.messages.create.return_value = mock_msg
    return provider


class TestAnthropicProviderJsonMode:
    def test_plain_json_passes_through(self):
        provider = _make_provider('{"step": 1}')
        result = provider.complete([Message(role="user", content="go")], json_mode=True)
        assert result.content == '{"step": 1}'

    def test_fenced_json_content_is_stripped_before_return(self):
        provider = _make_provider('```json\n{"step": 1}\n```')
        result = provider.complete([Message(role="user", content="go")], json_mode=True)
        assert result.content == '{"step": 1}'

    def test_fenced_no_tag_content_is_stripped_before_return(self):
        provider = _make_provider('```\n{"step": 1}\n```')
        result = provider.complete([Message(role="user", content="go")], json_mode=True)
        assert result.content == '{"step": 1}'

    def test_fenced_invalid_json_raises_provider_error(self):
        # Fence is stripped correctly, but the inner content is not valid JSON.
        provider = _make_provider("```json\n{not valid json}\n```")
        with pytest.raises(ProviderError, match="non-JSON"):
            provider.complete([Message(role="user", content="go")], json_mode=True)

    def test_plain_invalid_json_raises_provider_error(self):
        provider = _make_provider("not json at all")
        with pytest.raises(ProviderError, match="non-JSON"):
            provider.complete([Message(role="user", content="go")], json_mode=True)

    def test_json_mode_false_does_not_strip_or_validate(self):
        # Fenced content should be returned as-is when json_mode=False.
        raw = "```json\n{not valid}\n```"
        provider = _make_provider(raw)
        result = provider.complete([Message(role="user", content="go")], json_mode=False)
        assert result.content == raw
