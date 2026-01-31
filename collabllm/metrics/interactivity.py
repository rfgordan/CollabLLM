from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

INTERACTIVITY_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "prompts" / "evaluate_interactivity.txt"
)


@dataclass
class InteractivityResult:
    """Result of an interactivity evaluation."""

    score: float
    thought: str = ""
    raw_output: str = ""


def evaluate_interactivity(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> InteractivityResult:
    """
    Evaluate the interactivity of an assistant in a conversation
    using an LLM judge.

    Args:
        messages: Conversation history (from RolloutResult.messages).
        model: OpenAI model to use as judge.
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None).
        temperature: Sampling temperature (0.0 for deterministic).
        max_tokens: Maximum tokens for the judge response.

    Returns:
        InteractivityResult with score (0-1) and thought.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: uv add openai")

    template = INTERACTIVITY_TEMPLATE_PATH.read_text()
    chat_history = _format_chat_history(messages)
    prompt = template.format(chat_history=chat_history)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_output = response.choices[0].message.content
    logger.debug(f"Interactivity judge raw output: {raw_output[:200]}...")

    return _parse_response(raw_output)


def _format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Format message list as text for the evaluation template."""
    lines = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        label = "USER" if msg["role"] == "user" else "AI"
        lines.append(f"{label}: {msg['content']}")
    return "\n".join(lines) if lines else "(empty)"


def _parse_response(raw_output: str) -> InteractivityResult:
    """Parse the judge's JSON response into an InteractivityResult."""
    try:
        text = raw_output.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        parsed = json.loads(text)

        score = float(parsed.get("interactivity", 0.0))
        score = max(0.0, min(1.0, score))

        return InteractivityResult(
            score=score,
            thought=parsed.get("thought", ""),
            raw_output=raw_output,
        )
    except (json.JSONDecodeError, AttributeError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse interactivity judge output: {e}")
        logger.debug(f"Raw output was: {raw_output}")
        return InteractivityResult(
            score=0.0,
            raw_output=raw_output,
        )
