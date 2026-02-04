from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging
from copy import deepcopy

from collabllm.simulation.user_models import UserModel, UserTurnResult
from collabllm.simulation.extraction import extract_final_answer, ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Result of a chat simulation rollout."""

    messages: List[Dict[str, str]]
    user_turns: List[UserTurnResult] = field(default_factory=list)
    terminated_by_user: bool = False


class ChatSimulator:
    """
    Simulates multi-turn conversations between a local assistant and a user model.

    The simulator alternates between assistant and user turns, starting from
    a conversation prefix and continuing until max_turns is reached.
    """

    def __init__(
        self,
        assistant,
        user_model: UserModel,
    ):
        """
        Initialize the chat simulator.

        Args:
            assistant: Any object with a generate(messages) -> str method
                (e.g. LocalAssistant, VLLMAssistant)
            user_model: API-based model to simulate the user
        """
        self.assistant = assistant
        self.user_model = user_model

    def rollout(
        self,
        conversation_prefix: List[Dict[str, str]],
        max_turns: int,
        start_with_assistant: Optional[bool] = None,
    ) -> RolloutResult:
        """
        Continue a conversation from a prefix until max_turns is reached
        or the user model signals termination.

        A "turn" is one exchange (user message + assistant response).

        Args:
            conversation_prefix: Initial messages to start from.
                Format: [{"role": "system"|"user"|"assistant", "content": str}, ...]
            max_turns: Maximum number of turns to simulate.
                Each turn consists of one user message and one assistant response.
            start_with_assistant: Override which role goes first.
                If None, automatically determined from the last message in prefix.

        Returns:
            RolloutResult with messages, per-turn user model outputs, and
            whether the user model terminated the conversation.
        """
        messages = deepcopy(conversation_prefix)
        user_turns: List[UserTurnResult] = []
        terminated_by_user = False

        if start_with_assistant is None:
            start_with_assistant = self._should_assistant_go_next(messages)

        turns_completed = 0
        assistant_turn = start_with_assistant

        while turns_completed < max_turns:
            if assistant_turn:
                response = self.assistant.generate(messages)
                messages.append({"role": "assistant", "content": response})
                logger.info(f"Turn {turns_completed + 1}: Assistant responded")
            else:
                result = self.user_model.generate(messages)
                user_turns.append(result)

                if result.is_terminal:
                    logger.info(f"Turn {turns_completed + 1}: User signaled termination")
                    terminated_by_user = True
                    break

                messages.append({"role": "user", "content": result.response})
                logger.info(f"Turn {turns_completed + 1}: User responded")

            if not assistant_turn:
                turns_completed += 1

            assistant_turn = not assistant_turn

        # Ensure conversation ends with assistant response
        if not terminated_by_user and assistant_turn:
            response = self.assistant.generate(messages)
            messages.append({"role": "assistant", "content": response})
            logger.info("Final assistant response added")

        return RolloutResult(
            messages=messages,
            user_turns=user_turns,
            terminated_by_user=terminated_by_user,
        )

    def _should_assistant_go_next(self, messages: List[Dict[str, str]]) -> bool:
        """Determine if assistant should respond next based on last non-system message."""
        for msg in reversed(messages):
            if msg["role"] == "user":
                return True
            elif msg["role"] == "assistant":
                return False
        return False

    def interactive_rollout(
        self,
        conversation_prefix: List[Dict[str, str]],
        max_turns: int = 10,
    ) -> RolloutResult:
        """
        Interactive chat session with option to override or auto-generate responses.

        At each turn, you can:
        - Press Enter to auto-generate the response
        - Type a custom response to use instead
        - Type '/quit' or '/q' to end the session early

        Args:
            conversation_prefix: Initial messages to start from.
            max_turns: Maximum turns before auto-stopping (default 10).

        Returns:
            RolloutResult with messages and user turn metadata
        """
        messages = deepcopy(conversation_prefix)
        user_turns: List[UserTurnResult] = []
        assistant_turn = self._should_assistant_go_next(messages)
        turns_completed = 0
        terminated_by_user = False

        print("\n" + "=" * 60)
        print("Interactive Chat Session")
        print("- Press Enter to auto-generate")
        print("- Type your own response to override")
        print("- Type '/quit' or '/q' to exit")
        print("=" * 60)

        # Display prefix
        for msg in messages:
            self._print_message(msg)

        while turns_completed < max_turns:
            role = "assistant" if assistant_turn else "user"
            print(f"\n[{role.upper()}] ", end="", flush=True)

            try:
                user_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                break

            if user_input.lower() in ("/quit", "/q"):
                print("Session ended.")
                break

            if assistant_turn:
                if user_input:
                    response = user_input
                else:
                    print("  (generating...)", end="\r")
                    response = self.assistant.generate(messages)
                    print(f"  {response}")
                messages.append({"role": "assistant", "content": response})
            else:
                if user_input:
                    result = UserTurnResult(response=user_input)
                else:
                    print("  (generating...)", end="\r")
                    result = self.user_model.generate(messages)
                    print(f"  {result.response}")
                    if result.thought:
                        print(f"  [thought: {result.thought}]")

                user_turns.append(result)

                if result.is_terminal:
                    print("  (user model signaled end of conversation)")
                    terminated_by_user = True
                    break

                messages.append({"role": "user", "content": result.response})

            if not assistant_turn:
                turns_completed += 1

            assistant_turn = not assistant_turn

        print("\n" + "=" * 60)
        print(f"Session complete. {len(messages)} messages total.")
        print("=" * 60)

        return RolloutResult(
            messages=messages,
            user_turns=user_turns,
            terminated_by_user=terminated_by_user,
        )

    def extract_final_answer(
        self,
        rollout_result: RolloutResult,
        extract_type: str = "response",
        extraction_requirement: str = "",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract the assistant's final answer from a completed rollout.

        Uses an LLM to analyze the full conversation and synthesize
        the assistant's final, complete response.

        Args:
            rollout_result: Output from rollout() or interactive_rollout().
            extract_type: Type of content (e.g., "article", "code snippet").
            extraction_requirement: Additional extraction instructions.
            model: OpenAI model to use for extraction.
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None).

        Returns:
            ExtractionResult with final_completion and thought fields.
        """
        return extract_final_answer(
            messages=rollout_result.messages,
            extract_type=extract_type,
            extraction_requirement=extraction_requirement,
            model=model,
            api_key=api_key,
        )

    def _print_message(self, msg: Dict[str, str]) -> None:
        """Print a formatted message."""
        role = msg["role"].upper()
        content = msg["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"\n[{role}] {content}")
