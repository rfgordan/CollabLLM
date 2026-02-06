"""CLI for doc writing evaluation."""

import argparse
import logging
from datasets import load_dataset

from collabllm.evaluation import evaluate_checkpoint, evaluate_checkpoint_vllm, EvalResult
from collabllm.data_processing.dataset_utils import multiturn_dataset_to_sft

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on doc writing")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="HuggingFace model path or local path to evaluate.",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="HuggingFace dataset path to evaluate on.",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None,
        help="Optional path to LoRA adapter weights.",
    )
    parser.add_argument(
        "--use_4bit", action="store_true",
        help="Use 4-bit quantization when loading model.",
    )
    parser.add_argument(
        "--eval_ratio", type=float, default=1.0,
        help="Fraction of dataset to use for eval (default: all).",
    )
    parser.add_argument(
        "--max_turns", type=int, default=5,
        help="Maximum conversation turns per rollout.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=0,
        help="Number of eval traces to sample and log to wandb.",
    )
    parser.add_argument(
        "--use_vllm", action="store_true",
        help="Use vLLM for inference (faster, requires vllm package).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    logger.info(f"Loading dataset from {args.dataset_path}")
    raw_dataset = load_dataset(args.dataset_path, cache_dir="./data_cache", split="train")
    dataset = multiturn_dataset_to_sft(raw_dataset, eval_ratio=args.eval_ratio)
    logger.info(f"Dataset preprocessed: {dataset}")

    if args.use_vllm:
        result = evaluate_checkpoint_vllm(
            model_path=args.model_path,
            dataset=dataset,
            lora_path=args.lora_path,
            num_samples=args.num_samples,
            max_turns=args.max_turns,
        )
    else:
        result = evaluate_checkpoint(
            model_path=args.model_path,
            dataset=dataset,
            lora_path=args.lora_path,
            use_4bit=args.use_4bit,
            num_samples=args.num_samples,
            max_turns=args.max_turns,
        )

    logger.info(f"Eval complete: {result}")


if __name__ == "__main__":
    main()
