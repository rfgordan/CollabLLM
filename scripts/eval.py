import os, logging
from nltk.translate import bleu_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from collabllm.simulation.simulator import ChatSimulator, RolloutResult
from collabllm.simulation.extraction import ExtractionResult, extract_final_answer
from collabllm.simulation.user_models import OpenAIUserModel
from collabllm.simulation.assistant import LocalAssistant


DOC_WRITING_TASK_DESC = "Write a medium article."
logger = logging.getLogger(__name__)


def _eval_model_doc_writing(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset,
    ):
    """Evaluate the model on a doc-writing dataset via simulated conversations."""

    # Validate dataset structure
    eval_dataset = dataset['eval']
    required_columns = {"messages", "single_turn_prompt", "single_turn_completion"}
    missing = required_columns - set(eval_dataset.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    total_samples = len(eval_dataset)

    # Tokenize references for BLEU (list of list of tokens per reference)
    references = [
        [stc.split()] for stc in eval_dataset['single_turn_completion']
    ]
    hypotheses = []

    # Create assistant once — stateless, reusable across conversations
    assistant = LocalAssistant(model=model, tokenizer=tokenizer)

    for i, row in enumerate(eval_dataset):
        user_model = OpenAIUserModel(
            task_desc=DOC_WRITING_TASK_DESC,
            single_turn_prompt=row["single_turn_prompt"],
        )

        simulator = ChatSimulator(
            assistant=assistant,
            user_model=user_model,
        )

        rollout_result: RolloutResult = simulator.rollout(
            conversation_prefix=row["messages"],
            max_turns=5,
        )

        extraction_result: ExtractionResult = simulator.extract_final_answer(
            rollout_result,
            extract_type="article",
        )

        # Debug logging for low BLEU diagnosis
        last_assistant_msg = next(
            (m["content"] for m in reversed(rollout_result.messages) if m["role"] == "assistant"),
            "(no assistant message)",
        )
        logger.debug(
            f"Sample {i + 1}/{total_samples} - "
            f"last assistant message:\n{last_assistant_msg}\n"
        )
        logger.debug(
            f"Sample {i + 1}/{total_samples} - "
            f"extraction raw output:\n{extraction_result.raw_output}\n"
        )
        logger.debug(
            f"Sample {i + 1}/{total_samples} - "
            f"parsed final_completion:\n{extraction_result.final_completion}\n"
        )

        hypotheses.append(extraction_result.final_completion.split())
        logger.info(
            f"Processed sample {i + 1}/{total_samples} - "
            f"final completion: {extraction_result.final_completion[:100]}..."
        )

    bleu = bleu_score.corpus_bleu(references, hypotheses)
    logger.info(f"BLEU score over {total_samples} samples: {bleu:.4f}")
    return bleu
