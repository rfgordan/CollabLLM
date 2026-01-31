"""Tests for the metrics module."""

import json
import pytest
from unittest.mock import MagicMock, patch

from collabllm.metrics.interactivity import (
    evaluate_interactivity,
    InteractivityResult,
    _parse_response,
    _format_chat_history,
)


class TestInteractivityParsing:
    """Tests for interactivity judge response parsing."""

    def test_parse_valid_json(self):
        raw = json.dumps({
            "thought": "The assistant asked clarifying questions",
            "interactivity": 0.8,
        })

        result = _parse_response(raw)

        assert result.score == 0.8
        assert result.thought == "The assistant asked clarifying questions"
        assert result.raw_output == raw

    def test_parse_clamps_score_above_1(self):
        raw = json.dumps({"thought": "Great", "interactivity": 1.5})

        result = _parse_response(raw)

        assert result.score == 1.0

    def test_parse_clamps_score_below_0(self):
        raw = json.dumps({"thought": "Bad", "interactivity": -0.3})

        result = _parse_response(raw)

        assert result.score == 0.0

    def test_parse_with_markdown_fences(self):
        raw = '```json\n{"thought": "OK", "interactivity": 0.6}\n```'

        result = _parse_response(raw)

        assert result.score == 0.6

    def test_parse_invalid_json_returns_zero(self):
        raw = "This is not JSON"

        result = _parse_response(raw)

        assert result.score == 0.0
        assert result.raw_output == raw

    def test_parse_missing_score_defaults_zero(self):
        raw = json.dumps({"thought": "No score provided"})

        result = _parse_response(raw)

        assert result.score == 0.0


class TestInteractivityFormatting:
    """Tests for chat history formatting."""

    def test_format_excludes_system(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, what can I help with?"},
        ]

        history = _format_chat_history(messages)

        assert "System prompt" not in history
        assert "USER: Hello" in history
        assert "AI: Hi, what can I help with?" in history

    def test_format_empty_conversation(self):
        messages = [{"role": "system", "content": "System prompt"}]

        history = _format_chat_history(messages)

        assert history == "(empty)"


class TestEvaluateInteractivity:
    """Tests for the evaluate_interactivity function."""

    def test_evaluate_calls_api_and_parses(self):
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps({
                "thought": "Assistant proactively asked for clarification",
                "interactivity": 0.85,
            })
            mock_client.chat.completions.create.return_value = mock_response

            messages = [
                {"role": "user", "content": "Help me write code"},
                {"role": "assistant", "content": "What language and what should it do?"},
                {"role": "user", "content": "Python, a web scraper"},
                {"role": "assistant", "content": "What sites? Any rate limits?"},
            ]

            result = evaluate_interactivity(messages, api_key="test-key")

            assert isinstance(result, InteractivityResult)
            assert result.score == 0.85
            assert "proactively" in result.thought

            call_args = mock_client.chat.completions.create.call_args
            sent_messages = call_args.kwargs["messages"]
            assert len(sent_messages) == 1
            assert "Help me write code" in sent_messages[0]["content"]
            assert "interactivity" in sent_messages[0]["content"].lower()
