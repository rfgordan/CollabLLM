from dataclasses import dataclass
from typing import List, Optional
import os, logging, random, json
from nltk.translate import bleu_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from collabllm.simulation.simulator import ChatSimulator, RolloutResult
from collabllm.simulation.extraction import ExtractionResult, extract_final_answer
from collabllm.simulation.user_models import OpenAIUserModel
from collabllm.simulation.assistant import LocalAssistant
from collabllm.metrics import evaluate_interactivity


DOC_WRITING_TASK_DESC = "Write a medium article."
logger = logging.getLogger(__name__)


@dataclass
class SampledTrace:
    """A sampled trajectory from an eval run."""

    single_turn_prompt: str
    first_user_message: str
    last_assistant_message: str
    extracted_final_completion: str
    termination_reason: str


@dataclass
class EvalResult:
    avg_bleu: float
    avg_tokens: float
    avg_itr: float


def _eval_model_doc_writing(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset,
        num_samples: int = 0,
    ):
    """Evaluate the model on a doc-writing dataset via simulated conversations.

    Args:
        model: Pre-loaded HuggingFace model.
        tokenizer: Pre-loaded tokenizer.
        dataset: Dataset dict with 'eval' split.
        num_samples: Number of eval traces to sample and return (0 = none).
    """

    # Validate dataset structure
    eval_dataset = dataset['eval']
    required_columns = {"messages", "single_turn_prompt", "single_turn_completion"}
    missing = required_columns - set(eval_dataset.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    total_samples = len(eval_dataset)

    # Pre-select which indices to sample traces from
    sample_indices = set()
    if num_samples > 0:
        sample_indices = set(random.sample(
            range(total_samples), min(num_samples, total_samples)
        ))

    bleu_scores = []
    token_counts = []
    interactivity_scores = []
    sampled_traces: List[SampledTrace] = []

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

        reference = [row["single_turn_completion"].split()]
        hypothesis = extraction_result.final_completion.split()
        sample_bleu = bleu_score.sentence_bleu(reference, hypothesis)
        bleu_scores.append(sample_bleu)

        # Calculate # of tokens in conversation
        conversation_messages = rollout_result.messages[1:]
        total_tokens = sum(len(msg["content"].split()) for msg in conversation_messages)
        token_counts.append(total_tokens)

        # Evaluate interactivity
        itr_result = evaluate_interactivity(rollout_result.messages)
        interactivity_scores.append(itr_result.score)

        # Sample trace if selected
        if i in sample_indices:
            first_user_msg = next(
                (m["content"] for m in rollout_result.messages if m["role"] == "user"),
                "(no user message)",
            )

            if rollout_result.terminated_by_user:
                termination_reason = "user_signal"
            elif len([m for m in rollout_result.messages if m["role"] == "user"]) >= 5:
                termination_reason = "max_turns"
            else:
                termination_reason = "max_turns"

            sampled_traces.append(SampledTrace(
                single_turn_prompt=row["single_turn_prompt"],
                first_user_message=first_user_msg,
                last_assistant_message=last_assistant_msg,
                extracted_final_completion=extraction_result.final_completion,
                termination_reason=termination_reason,
            ))

        logger.info(
            f"Processed sample {i + 1}/{total_samples} - "
            f"BLEU: {sample_bleu:.4f} - "
            f"interactivity: {itr_result.score:.2f} - "
            f"tokens: {total_tokens} - "
            f"reference: {row['single_turn_completion'][:100]}... - "
            f"final completion: {extraction_result.final_completion[:100]}..."
        )

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
    avg_itr = sum(interactivity_scores) / len(interactivity_scores) if interactivity_scores else 0.0

    logger.info(f"Average sentence BLEU over {total_samples} samples: {avg_bleu:.4f}")
    logger.info(f"Average tokens over {total_samples} samples: {avg_tokens:.2f}")
    logger.info(f"Average interactivity over {total_samples} samples: {avg_itr:.4f}")

    if sampled_traces:
        _log_traces_to_wandb(sampled_traces)

    return EvalResult(avg_bleu=avg_bleu, avg_tokens=avg_tokens, avg_itr=avg_itr)


def _log_traces_to_wandb(traces: List[SampledTrace]) -> None:
    """Log sampled eval traces as a wandb artifact."""
    try:
        import wandb

        if wandb.run is None:
            logger.warning("No active wandb run, skipping trace logging")
            return

        traces_data = [
            {
                "single_turn_prompt": t.single_turn_prompt,
                "first_user_message": t.first_user_message,
                "last_assistant_message": t.last_assistant_message,
                "extracted_final_completion": t.extracted_final_completion,
                "termination_reason": t.termination_reason,
            }
            for t in traces
        ]

        artifact = wandb.Artifact(
            name=f"eval-traces-{wandb.run.id}",
            type="eval_traces",
            description="Sampled evaluation trajectories from doc writing eval",
        )
        with artifact.new_file("traces.json") as f:
            f.write(json.dumps(traces_data, indent=2))

        wandb.log_artifact(artifact)
        logger.info(f"Logged {len(traces)} eval traces to wandb artifact")

    except ImportError:
        logger.warning("wandb not installed, skipping trace logging")
    except Exception as e:
        logger.warning(f"Failed to log traces to wandb: {e}")
