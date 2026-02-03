"""Document writing evaluation via simulated conversations."""

from dataclasses import dataclass
from typing import List, Optional, Dict
import logging, random, json, time
from nltk.translate import bleu_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from collabllm.simulation.simulator import ChatSimulator, RolloutResult
from collabllm.simulation.extraction import ExtractionResult
from collabllm.simulation.user_models import OpenAIUserModel
from collabllm.simulation.assistant import LocalAssistant
from collabllm.metrics import evaluate_interactivity

logger = logging.getLogger(__name__)

DOC_WRITING_TASK_DESC = "Write a medium article."


@dataclass
class TimingStats:
    """Timing statistics for an eval run."""
    total_time: float = 0.0
    assistant_time: float = 0.0
    user_time: float = 0.0
    num_samples: int = 0

    def as_dict(self) -> Dict:
        return {
            "total_time_s": self.total_time,
            "assistant_time_s": self.assistant_time,
            "user_time_s": self.user_time,
            "avg_assistant_time_s": self.assistant_time / max(1, self.num_samples),
            "avg_user_time_s": self.user_time / max(1, self.num_samples),
            "avg_sample_time_s": self.total_time / max(1, self.num_samples),
            "num_samples": self.num_samples,
        }


class _TimedAssistant:
    """Wrapper that tracks cumulative generation time."""
    def __init__(self, assistant):
        self._assistant = assistant
        self.total_time = 0.0

    def generate(self, messages):
        start = time.perf_counter()
        result = self._assistant.generate(messages)
        self.total_time += time.perf_counter() - start
        return result


class _TimedUserModel:
    """Wrapper that tracks cumulative generation time."""
    def __init__(self, user_model):
        self._user_model = user_model
        self.total_time = 0.0

    def generate(self, messages):
        start = time.perf_counter()
        result = self._user_model.generate(messages)
        self.total_time += time.perf_counter() - start
        return result


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
    """Result of a doc writing evaluation."""
    avg_bleu: float
    avg_tokens: float
    avg_itr: float


def _log_traces_to_wandb(traces: List[SampledTrace]) -> None:
    """Log sampled eval traces as a wandb artifact."""
    try:
        import wandb

        if wandb.run is None:
            wandb.init(project="collabllm", job_type="eval")
            logger.info("Initialized new wandb run for eval trace logging")

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


def _log_timing_to_wandb(timing: TimingStats) -> None:
    """Log timing statistics to wandb."""
    try:
        import wandb

        if wandb.run is None:
            wandb.init(project="collabllm", job_type="eval")

        wandb.log(timing.as_dict())
        logger.info("Logged timing stats to wandb")

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to log timing to wandb: {e}")


def _run_eval(
    assistant: LocalAssistant,
    dataset,
    num_samples: int = 0,
    max_turns: int = 5,
) -> EvalResult:
    """Core eval logic operating on a LocalAssistant and preprocessed dataset.

    Args:
        assistant: LocalAssistant instance (pre-loaded or from path).
        dataset: DatasetDict with 'eval' split (output of multiturn_dataset_to_sft).
        num_samples: Number of eval traces to sample and log to wandb (0 = none).
        max_turns: Maximum conversation turns per rollout.
    """
    eval_dataset = dataset['eval']
    required_columns = {"messages", "single_turn_prompt", "single_turn_completion"}
    missing = required_columns - set(eval_dataset.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    total_samples = len(eval_dataset)

    sample_indices = set()
    if num_samples > 0:
        sample_indices = set(random.sample(
            range(total_samples), min(num_samples, total_samples)
        ))

    bleu_scores = []
    token_counts = []
    interactivity_scores = []
    sampled_traces: List[SampledTrace] = []

    timed_assistant = _TimedAssistant(assistant)
    cumulative_user_time = 0.0
    eval_start = time.perf_counter()

    for i, row in enumerate(eval_dataset):
        user_model = OpenAIUserModel(
            task_desc=DOC_WRITING_TASK_DESC,
            single_turn_prompt=row["single_turn_prompt"],
        )
        timed_user = _TimedUserModel(user_model)

        simulator = ChatSimulator(
            assistant=timed_assistant,
            user_model=timed_user,
        )

        rollout_result: RolloutResult = simulator.rollout(
            conversation_prefix=row["messages"],
            max_turns=max_turns,
        )

        extraction_result: ExtractionResult = simulator.extract_final_answer(
            rollout_result,
            extract_type="article",
        )

        last_assistant_msg = next(
            (m["content"] for m in reversed(rollout_result.messages) if m["role"] == "assistant"),
            "(no assistant message)",
        )
        logger.debug(f"Sample {i + 1}/{total_samples} - last assistant message:\n{last_assistant_msg}\n")
        logger.debug(f"Sample {i + 1}/{total_samples} - extraction raw output:\n{extraction_result.raw_output}\n")
        logger.debug(f"Sample {i + 1}/{total_samples} - parsed final_completion:\n{extraction_result.final_completion}\n")

        reference = [row["single_turn_completion"].split()]
        hypothesis = extraction_result.final_completion.split()
        sample_bleu = bleu_score.sentence_bleu(reference, hypothesis)
        bleu_scores.append(sample_bleu)

        conversation_messages = rollout_result.messages[1:]
        total_tokens = sum(len(msg["content"].split()) for msg in conversation_messages)
        token_counts.append(total_tokens)

        itr_result = evaluate_interactivity(rollout_result.messages)
        interactivity_scores.append(itr_result.score)

        if i in sample_indices:
            first_user_msg = next(
                (m["content"] for m in rollout_result.messages if m["role"] == "user"),
                "(no user message)",
            )
            termination_reason = "user_signal" if rollout_result.terminated_by_user else "max_turns"

            sampled_traces.append(SampledTrace(
                single_turn_prompt=row["single_turn_prompt"],
                first_user_message=first_user_msg,
                last_assistant_message=last_assistant_msg,
                extracted_final_completion=extraction_result.final_completion,
                termination_reason=termination_reason,
            ))

        cumulative_user_time += timed_user.total_time

        logger.info(
            f"Processed sample {i + 1}/{total_samples} - "
            f"BLEU: {sample_bleu:.4f} - "
            f"interactivity: {itr_result.score:.2f} - "
            f"tokens: {total_tokens} - "
            f"reference: {row['single_turn_completion'][:100]}... - "
            f"final completion: {extraction_result.final_completion[:100]}..."
        )

    timing = TimingStats(
        total_time=time.perf_counter() - eval_start,
        assistant_time=timed_assistant.total_time,
        user_time=cumulative_user_time,
        num_samples=total_samples,
    )
    logger.info(f"Timing: {timing.as_dict()}")
    _log_timing_to_wandb(timing)

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
    avg_itr = sum(interactivity_scores) / len(interactivity_scores) if interactivity_scores else 0.0

    logger.info(f"Average sentence BLEU over {total_samples} samples: {avg_bleu:.4f}")
    logger.info(f"Average tokens over {total_samples} samples: {avg_tokens:.2f}")
    logger.info(f"Average interactivity over {total_samples} samples: {avg_itr:.4f}")

    if sampled_traces:
        _log_traces_to_wandb(sampled_traces)

    return EvalResult(avg_bleu=avg_bleu, avg_tokens=avg_tokens, avg_itr=avg_itr)


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset,
    num_samples: int = 0,
    max_turns: int = 5,
) -> EvalResult:
    """Evaluate a pre-loaded model on doc writing.

    Args:
        model: Pre-loaded HuggingFace model.
        tokenizer: Pre-loaded tokenizer.
        dataset: DatasetDict with 'eval' split.
        num_samples: Number of eval traces to sample and log to wandb.
        max_turns: Maximum conversation turns per rollout.
    """
    assistant = LocalAssistant(model=model, tokenizer=tokenizer)
    return _run_eval(assistant, dataset, num_samples=num_samples, max_turns=max_turns)


def evaluate_checkpoint(
    model_path: str,
    dataset,
    lora_path: Optional[str] = None,
    use_4bit: bool = False,
    num_samples: int = 0,
    max_turns: int = 5,
) -> EvalResult:
    """Evaluate a model checkpoint on doc writing.

    Args:
        model_path: HuggingFace model path or local path.
        dataset: DatasetDict with 'eval' split.
        lora_path: Optional path to LoRA adapter weights.
        use_4bit: Whether to use 4-bit quantization.
        num_samples: Number of eval traces to sample and log to wandb.
        max_turns: Maximum conversation turns per rollout.
    """
    assistant = LocalAssistant(
        model_path=model_path,
        lora_path=lora_path,
        use_4bit=use_4bit,
    )
    return _run_eval(assistant, dataset, num_samples=num_samples, max_turns=max_turns)
