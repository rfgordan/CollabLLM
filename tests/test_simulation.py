"""Tests for the simulation module."""

import json
import pytest
from typing import List, Dict
from unittest.mock import MagicMock, patch

from collabllm.simulation.user_models import (
    UserModel,
    UserTurnResult,
    DEFAULT_TERMINAL_SIGNAL,
)
from collabllm.simulation.simulator import ChatSimulator, RolloutResult


# -- Mock implementations --


class MockUserModel(UserModel):
    """Mock user model that returns pre-defined UserTurnResults."""

    def __init__(self, results: List[UserTurnResult]):
        super().__init__(
            task_desc="Test task",
            single_turn_prompt="Test prompt",
            terminal_signal=DEFAULT_TERMINAL_SIGNAL,
        )
        self.results = results
        self.call_count = 0
        self.received_messages = []

    def generate(self, messages: List[Dict[str, str]]) -> UserTurnResult:
        self.received_messages.append(messages)
        result = self.results[self.call_count % len(self.results)]
        self.call_count += 1
        return result


class MockAssistant:
    """Mock assistant for testing."""

    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
        self.received_messages = []

    def generate(self, messages: List[Dict[str, str]]) -> str:
        self.received_messages.append(messages)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


# -- UserModel tests --


class TestUserModelPromptFormatting:
    """Tests for prompt template formatting."""

    def test_format_chat_history_basic(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        history = model._format_chat_history(messages)

        assert "USER: Hello" in history
        assert "AI: Hi there" in history
        assert "System prompt" not in history

    def test_format_chat_history_empty(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        messages = [{"role": "system", "content": "System prompt"}]

        history = model._format_chat_history(messages)

        assert history == "(empty)"

    def test_format_prompt_fills_template(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        prompt = model._format_prompt(messages)

        assert "Test task" in prompt
        assert "Test prompt" in prompt
        assert "USER: Hello" in prompt
        assert "AI: Hi" in prompt


class TestUserModelResponseParsing:
    """Tests for JSON response parsing."""

    def test_parse_valid_json(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        raw = json.dumps({
            "current_answer": "The AI explained X",
            "thought": "I need more detail",
            "response": "Can you elaborate?",
        })

        result = model._parse_response(raw)

        assert result.response == "Can you elaborate?"
        assert result.thought == "I need more detail"
        assert result.current_answer == "The AI explained X"
        assert result.is_terminal is False
        assert result.raw_output == raw

    def test_parse_terminal_signal(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        raw = json.dumps({
            "current_answer": "Solved",
            "thought": "Done",
            "response": DEFAULT_TERMINAL_SIGNAL,
        })

        result = model._parse_response(raw)

        assert result.is_terminal is True

    def test_parse_json_with_markdown_fences(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        raw = '```json\n{"current_answer": "X", "thought": "Y", "response": "Z"}\n```'

        result = model._parse_response(raw)

        assert result.response == "Z"
        assert result.thought == "Y"

    def test_parse_invalid_json_falls_back(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        raw = "This is not JSON at all"

        result = model._parse_response(raw)

        assert result.response == raw
        assert result.is_terminal is False
        assert result.raw_output == raw


# -- ChatSimulator tests --


class TestChatSimulator:
    """Tests for the ChatSimulator class."""

    def test_rollout_basic(self):
        user_results = [
            UserTurnResult(response="User question 1"),
            UserTurnResult(response="User question 2"),
        ]
        user = MockUserModel(results=user_results)
        assistant = MockAssistant(responses=["Assistant response 1", "Assistant response 2", "Assistant response 3"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)

        prefix = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Initial question"},
        ]

        result = simulator.rollout(prefix, max_turns=2)

        assert isinstance(result, RolloutResult)
        assert len(result.messages) == 7
        assert result.messages[0]["role"] == "system"
        assert result.messages[1]["role"] == "user"
        assert result.messages[2]["role"] == "assistant"
        assert result.messages[-1]["role"] == "assistant"
        assert result.terminated_by_user is False

    def test_rollout_captures_user_turns(self):
        user_results = [
            UserTurnResult(response="Q1", thought="thinking about Q1", current_answer="none yet"),
        ]
        user = MockUserModel(results=user_results)
        assistant = MockAssistant(responses=["A1", "A2"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)
        prefix = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Initial"},
        ]

        result = simulator.rollout(prefix, max_turns=1)

        assert len(result.user_turns) == 1
        assert result.user_turns[0].thought == "thinking about Q1"
        assert result.user_turns[0].current_answer == "none yet"

    def test_rollout_terminal_signal_stops_early(self):
        user_results = [
            UserTurnResult(response="Q1"),
            UserTurnResult(response=DEFAULT_TERMINAL_SIGNAL, is_terminal=True),
        ]
        user = MockUserModel(results=user_results)
        assistant = MockAssistant(responses=["A1", "A2", "A3"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)
        prefix = [{"role": "user", "content": "Start"}]

        result = simulator.rollout(prefix, max_turns=5)

        assert result.terminated_by_user is True
        # Terminal message is not appended to messages
        assert all(m["content"] != DEFAULT_TERMINAL_SIGNAL for m in result.messages)
        # But it is captured in user_turns
        assert result.user_turns[-1].is_terminal is True

    def test_rollout_preserves_prefix(self):
        user = MockUserModel(results=[UserTurnResult(response="Follow up")])
        assistant = MockAssistant(responses=["Response", "Response 2"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)

        prefix = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Original message"},
        ]

        result = simulator.rollout(prefix, max_turns=1)

        assert result.messages[0]["content"] == "System prompt"
        assert result.messages[1]["content"] == "Original message"

    def test_rollout_does_not_modify_original_prefix(self):
        user = MockUserModel(results=[UserTurnResult(response="User msg")])
        assistant = MockAssistant(responses=["Assistant msg", "Assistant msg 2"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)

        prefix = [{"role": "user", "content": "Hello"}]
        original_len = len(prefix)

        simulator.rollout(prefix, max_turns=1)

        assert len(prefix) == original_len

    def test_should_assistant_go_next_after_user(self):
        user = MockUserModel(results=[UserTurnResult(response="test")])
        assistant = MockAssistant(responses=["test"])
        simulator = ChatSimulator(assistant=assistant, user_model=user)

        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "hello"},
        ]

        assert simulator._should_assistant_go_next(messages) is True

    def test_should_assistant_go_next_after_assistant(self):
        user = MockUserModel(results=[UserTurnResult(response="test")])
        assistant = MockAssistant(responses=["test"])
        simulator = ChatSimulator(assistant=assistant, user_model=user)

        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        assert simulator._should_assistant_go_next(messages) is False

    def test_should_user_go_first_with_only_system_prompt(self):
        user = MockUserModel(results=[UserTurnResult(response="test")])
        assistant = MockAssistant(responses=["test"])
        simulator = ChatSimulator(assistant=assistant, user_model=user)

        messages = [{"role": "system", "content": "prompt"}]

        assert simulator._should_assistant_go_next(messages) is False

    def test_rollout_ends_with_assistant(self):
        user_results = [
            UserTurnResult(response="Q1"),
            UserTurnResult(response="Q2"),
        ]
        user = MockUserModel(results=user_results)
        assistant = MockAssistant(responses=["A1", "A2", "A3"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)

        prefix = [{"role": "user", "content": "Start"}]
        result = simulator.rollout(prefix, max_turns=2)

        assert result.messages[-1]["role"] == "assistant"


# -- LocalAssistant tests --


class TestLocalAssistantWithTinyModel:
    """Tests for LocalAssistant using a minimal HuggingFace model."""

    @pytest.fixture
    def tiny_model_path(self):
        return "hf-internal-testing/tiny-random-LlamaForCausalLM"

    @pytest.mark.slow
    def test_local_assistant_initialization(self, tiny_model_path):
        """Test that LocalAssistant can load a tiny model."""
        from collabllm.simulation.assistant import LocalAssistant

        assistant = LocalAssistant(
            model_path=tiny_model_path,
            device_map="cpu",
            use_4bit=False,
        )

        assert assistant.model is not None
        assert assistant.tokenizer is not None

    @pytest.mark.slow
    def test_local_assistant_generate(self, tiny_model_path):
        """Test that LocalAssistant can generate a response."""
        from collabllm.simulation.assistant import LocalAssistant

        assistant = LocalAssistant(
            model_path=tiny_model_path,
            device_map="cpu",
            use_4bit=False,
            max_new_tokens=10,
        )

        messages = [
            {"role": "user", "content": "Hello"},
        ]

        response = assistant.generate(messages)

        assert isinstance(response, str)
        assert len(response) >= 0


# -- OpenAIUserModel tests --


class TestOpenAIUserModel:
    """Tests for OpenAIUserModel."""

    def test_openai_user_model_generate(self):
        """Test OpenAIUserModel.generate calls API and parses JSON response."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps({
                "current_answer": "AI said hello",
                "thought": "Seems friendly",
                "response": "Tell me more",
            })
            mock_client.chat.completions.create.return_value = mock_response

            from collabllm.simulation.user_models import OpenAIUserModel

            model = OpenAIUserModel(
                task_desc="Test task",
                single_turn_prompt="Test prompt",
                api_key="test-key",
            )

            messages = [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]

            result = model.generate(messages)

            assert isinstance(result, UserTurnResult)
            assert result.response == "Tell me more"
            assert result.thought == "Seems friendly"
            assert result.current_answer == "AI said hello"
            assert result.is_terminal is False

            # Verify the API was called with a single user message containing the template
            call_args = mock_client.chat.completions.create.call_args
            sent_messages = call_args.kwargs["messages"]
            assert len(sent_messages) == 1
            assert sent_messages[0]["role"] == "user"
            assert "Test task" in sent_messages[0]["content"]


# -- Extraction tests --


class TestExtraction:
    """Tests for final answer extraction."""

    def test_parse_valid_extraction(self):
        from collabllm.simulation.extraction import _parse_extraction

        raw = json.dumps({
            "thought": "The assistant revised the intro in turn 3",
            "final_completion": "Here is the final article...",
        })

        result = _parse_extraction(raw)

        assert result.final_completion == "Here is the final article..."
        assert result.thought == "The assistant revised the intro in turn 3"
        assert result.raw_output == raw

    def test_parse_extraction_with_markdown_fences(self):
        from collabllm.simulation.extraction import _parse_extraction

        raw = '```json\n{"thought": "T", "final_completion": "Final answer"}\n```'

        result = _parse_extraction(raw)

        assert result.final_completion == "Final answer"

    def test_parse_extraction_invalid_json(self):
        from collabllm.simulation.extraction import _parse_extraction

        raw = "The final answer is 42"

        result = _parse_extraction(raw)

        assert result.final_completion == raw
        assert result.raw_output == raw

    def test_format_chat_history(self):
        from collabllm.simulation.extraction import _format_chat_history

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Write an essay"},
            {"role": "assistant", "content": "Here is a draft..."},
            {"role": "user", "content": "Add more detail"},
            {"role": "assistant", "content": "Here is the revised version..."},
        ]

        history = _format_chat_history(messages)

        assert "System prompt" not in history
        assert "USER: Write an essay" in history
        assert "AI: Here is a draft..." in history
        assert "USER: Add more detail" in history
        assert "AI: Here is the revised version..." in history

    def test_extract_final_answer_api_call(self):
        """Test extract_final_answer calls OpenAI and parses response."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps({
                "thought": "Combined all revisions",
                "final_completion": "The complete article with all edits.",
            })
            mock_client.chat.completions.create.return_value = mock_response

            from collabllm.simulation.extraction import extract_final_answer

            messages = [
                {"role": "user", "content": "Write an article"},
                {"role": "assistant", "content": "Here is a draft"},
                {"role": "user", "content": "Revise it"},
                {"role": "assistant", "content": "Here is the revision"},
            ]

            result = extract_final_answer(
                messages=messages,
                extract_type="article",
                api_key="test-key",
            )

            assert result.final_completion == "The complete article with all edits."
            assert result.thought == "Combined all revisions"

            call_args = mock_client.chat.completions.create.call_args
            sent_messages = call_args.kwargs["messages"]
            assert len(sent_messages) == 1
            assert "article" in sent_messages[0]["content"]
            assert "Write an article" in sent_messages[0]["content"]

    def test_simulator_extract_convenience_method(self):
        """Test ChatSimulator.extract_final_answer wraps the function."""
        with patch("collabllm.simulation.simulator.extract_final_answer") as mock_extract:
            mock_extract.return_value = MagicMock(final_completion="result")

            user = MockUserModel(results=[UserTurnResult(response="test")])
            assistant = MockAssistant(responses=["test"])
            simulator = ChatSimulator(assistant=assistant, user_model=user)

            rollout_result = RolloutResult(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            )

            result = simulator.extract_final_answer(
                rollout_result, extract_type="code snippet"
            )

            mock_extract.assert_called_once_with(
                messages=rollout_result.messages,
                extract_type="code snippet",
                extraction_requirement="",
                model="gpt-4o-mini",
                api_key=None,
            )
            assert result.final_completion == "result"
