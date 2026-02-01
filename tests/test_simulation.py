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


# -- _sanitize_messages tests --


class TestSanitizeMessages:
    """Tests for LocalAssistant._sanitize_messages (TemplateError fix)."""

    def test_merges_consecutive_user_messages(self):
        from collabllm.simulation.assistant import LocalAssistant

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Are you there?"},
        ]

        result = LocalAssistant._sanitize_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Hello\nAre you there?" == result[0]["content"]

    def test_merges_consecutive_assistant_messages(self):
        from collabllm.simulation.assistant import LocalAssistant

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Part 1"},
            {"role": "assistant", "content": "Part 2"},
        ]

        result = LocalAssistant._sanitize_messages(messages)

        assert len(result) == 2
        assert result[1]["content"] == "Part 1\nPart 2"

    def test_preserves_already_alternating(self):
        from collabllm.simulation.assistant import LocalAssistant

        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ]

        result = LocalAssistant._sanitize_messages(messages)

        assert len(result) == 4

    def test_does_not_modify_original(self):
        from collabllm.simulation.assistant import LocalAssistant

        messages = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        original_len = len(messages)

        LocalAssistant._sanitize_messages(messages)

        assert len(messages) == original_len
        assert messages[0]["content"] == "A"

    def test_empty_messages(self):
        from collabllm.simulation.assistant import LocalAssistant

        assert LocalAssistant._sanitize_messages([]) == []

    def test_merges_three_consecutive(self):
        from collabllm.simulation.assistant import LocalAssistant

        messages = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
            {"role": "user", "content": "C"},
        ]

        result = LocalAssistant._sanitize_messages(messages)

        assert len(result) == 1
        assert result[0]["content"] == "A\nB\nC"


# -- LocalAssistant init validation tests --


class TestLocalAssistantInit:
    """Tests for LocalAssistant constructor validation."""

    def test_raises_without_model_or_path(self):
        from collabllm.simulation.assistant import LocalAssistant

        with pytest.raises(ValueError, match="Either model_path or model"):
            LocalAssistant()

    def test_raises_model_without_tokenizer(self):
        from collabllm.simulation.assistant import LocalAssistant

        mock_model = MagicMock()
        mock_model.get_memory_footprint.return_value = 1e9

        with pytest.raises(ValueError, match="tokenizer is required"):
            LocalAssistant(model=mock_model)

    def test_accepts_preloaded_model_and_tokenizer(self):
        from collabllm.simulation.assistant import LocalAssistant

        mock_model = MagicMock()
        mock_model.get_memory_footprint.return_value = 1e9
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        assistant = LocalAssistant(model=mock_model, tokenizer=mock_tokenizer)

        assert assistant.model is mock_model
        assert assistant.tokenizer is mock_tokenizer

    def test_sets_pad_token_when_none(self):
        from collabllm.simulation.assistant import LocalAssistant

        mock_model = MagicMock()
        mock_model.get_memory_footprint.return_value = 1e9
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        LocalAssistant(model=mock_model, tokenizer=mock_tokenizer)

        assert mock_tokenizer.pad_token == "<eos>"


# -- Rollout edge case tests --


class TestRolloutEdgeCases:
    """Tests for rollout edge cases."""

    def test_system_only_prefix_user_goes_first(self):
        """Rollout from system-only prefix should start with user."""
        user_results = [UserTurnResult(response="Hello")]
        user = MockUserModel(results=user_results)
        assistant = MockAssistant(responses=["Hi there"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)
        prefix = [{"role": "system", "content": "Be helpful."}]

        result = simulator.rollout(prefix, max_turns=1)

        # user goes first, then assistant
        assert result.messages[1]["role"] == "user"
        assert result.messages[1]["content"] == "Hello"
        assert result.messages[2]["role"] == "assistant"

    def test_empty_prefix_user_goes_first(self):
        """Empty prefix should start with user."""
        user_results = [UserTurnResult(response="Hey")]
        user = MockUserModel(results=user_results)
        assistant = MockAssistant(responses=["Yo"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)

        result = simulator.rollout([], max_turns=1)

        assert result.messages[0]["role"] == "user"

    def test_terminal_on_first_user_turn(self):
        """User terminates immediately on the first turn."""
        user_results = [UserTurnResult(response="[END]", is_terminal=True)]
        user = MockUserModel(results=user_results)
        assistant = MockAssistant(responses=["Should not be called"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)
        prefix = [{"role": "system", "content": "Sys"}]

        result = simulator.rollout(prefix, max_turns=5)

        assert result.terminated_by_user is True
        assert len(result.messages) == 1  # only system message
        assert assistant.call_count == 0

    def test_max_turns_zero(self):
        """max_turns=0 should return prefix unchanged (no turns executed)."""
        user = MockUserModel(results=[UserTurnResult(response="test")])
        assistant = MockAssistant(responses=["test"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)
        prefix = [{"role": "user", "content": "Hello"}]

        result = simulator.rollout(prefix, max_turns=0)

        # With 0 turns and prefix ending with user, the final-assistant logic fires
        assert result.messages[-1]["role"] == "assistant"
        assert user.call_count == 0

    def test_start_with_assistant_override(self):
        """Explicitly override start_with_assistant=True."""
        user_results = [UserTurnResult(response="Follow up")]
        user = MockUserModel(results=user_results)
        assistant = MockAssistant(responses=["First response", "Second response"])

        simulator = ChatSimulator(assistant=assistant, user_model=user)
        prefix = [{"role": "system", "content": "Sys"}]

        result = simulator.rollout(prefix, max_turns=1, start_with_assistant=True)

        # assistant goes first despite system-only prefix
        assert result.messages[1]["role"] == "assistant"
        assert result.messages[1]["content"] == "First response"


# -- Parse response edge cases --


class TestParseResponseEdgeCases:
    """Edge cases for user model JSON parsing."""

    def test_terminal_signal_with_whitespace(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        raw = json.dumps({
            "current_answer": "Done",
            "thought": "Finished",
            "response": "  [END]  ",
        })

        result = model._parse_response(raw)

        assert result.is_terminal is True

    def test_custom_terminal_signal(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        model.terminal_signal = "STOP"
        raw = json.dumps({
            "response": "STOP",
            "thought": "Done",
        })

        result = model._parse_response(raw)

        assert result.is_terminal is True

    def test_partial_terminal_signal_not_terminal(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        raw = json.dumps({
            "response": "[END] but I have more to say",
            "thought": "Not done",
        })

        result = model._parse_response(raw)

        assert result.is_terminal is False

    def test_markdown_fences_with_language_tag(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        raw = '```json\n{"response": "hello", "thought": "T"}\n```'

        result = model._parse_response(raw)

        assert result.response == "hello"

    def test_markdown_fences_without_language_tag(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])
        raw = '```\n{"response": "hello", "thought": "T"}\n```'

        result = model._parse_response(raw)

        assert result.response == "hello"

    def test_empty_string_input(self):
        model = MockUserModel(results=[UserTurnResult(response="test")])

        result = model._parse_response("")

        assert result.response == ""
        assert result.is_terminal is False
