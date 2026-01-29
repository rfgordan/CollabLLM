from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

EXTRACTION_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "prompts" / "extract_multiturn_completion.txt"
)


@dataclass
class ExtractionResult:
    """Result of extracting the final answer from a conversation."""

    final_completion: str
    thought: str = ""
    raw_output: str = ""


def extract_final_answer(
    messages: List[Dict[str, str]],
    extract_type: str = "response",
    extraction_requirement: str = "",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ExtractionResult:
    """
    Extract the assistant's final answer from a multi-turn conversation
    using the extract_multiturn_completion prompt template.

    Args:
        messages: Conversation history (from RolloutResult.messages)
        extract_type: Type of content to extract (e.g., "article", "code snippet",
            "response"). Fills {extract_type} in the template.
        extraction_requirement: Additional extraction instructions.
            Fills {extraction_requirement} in the template.
        model: OpenAI model to use for extraction.
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None).
        temperature: Sampling temperature (0.0 for deterministic extraction).
        max_tokens: Maximum tokens for the extraction response.

    Returns:
        ExtractionResult with final_completion and thought fields.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: uv add openai")

    template = EXTRACTION_TEMPLATE_PATH.read_text()

    chat_history = _format_chat_history(messages)
    prompt = template.format(
        extract_type=extract_type,
        chat_history=chat_history,
        extraction_requirement=extraction_requirement,
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_output = response.choices[0].message.content
    logger.debug(f"Extraction raw output: {raw_output[:200]}...")

    return _parse_extraction(raw_output)


def _format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Format message list as text for the extraction template."""
    lines = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        label = "USER" if msg["role"] == "user" else "AI"
        lines.append(f"{label}: {msg['content']}")
    return "\n".join(lines) if lines else "(empty)"


def _parse_extraction(raw_output: str) -> ExtractionResult:
    """Parse the extraction model's JSON response."""
    try:
        text = raw_output.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        parsed = json.loads(text)

        return ExtractionResult(
            final_completion=parsed.get("final_completion", ""),
            thought=parsed.get("thought", ""),
            raw_output=raw_output,
        )
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Failed to parse extraction JSON output: {e}")
        logger.debug(f"Raw output was: {raw_output}")
        return ExtractionResult(
            final_completion=raw_output,
            raw_output=raw_output,
        )
