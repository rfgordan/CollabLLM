# general python imports
from typing import Any
import logging
import argparse, os, json

# HF / TRL / PyTorch stack
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from transformers import AutoTokenizer
import transformers
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset
import torch
import wandb

# project code
from collabllm.data_processing.dataset_utils import multiturn_dataset_to_sft
from collabllm.training.train_utils import get_timebased_filename
from collabllm.evaluation import evaluate_checkpoint_vllm, EvalResult

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to the Hugging Face model to be fine-tuned.",
    )
    parser.add_argument(
        "--hf_dataset_path",
        type=str,
        required=True,
        help="Path to the Hugging Face dataset for fine-tuning.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--parse_secrets_runpod",
        action="store_true",
        help="Whether to parse secrets from RunPod environment variables.",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Whether to run evaluation after training.",
    )
    parser.add_argument(
        "--output_name_tag",
        type=str,
        default="default",
        help="Tag to append to the output model name.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Maximum conversation turns per eval rollout.",
    )
    parser.add_argument(
        "--eval_ratio", type=float, default=0.05,
        help="Fraction of dataset to use for eval (default: 5%).",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank parameter.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate.",
    )
    return parser.parse_args()

def load_and_train_dpo(
        hf_model_path: str,
        hf_dataset_path: str,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        parse_secrets_runpod: bool = False,
        run_eval: bool = False,
        output_name_tag: str = "default",
        max_turns: int = 5,
        eval_ratio: float = 0.05,
        num_train_epochs: int = 3,
        lora_alpha: int = 16,
        lora_r: int = 16,
        lora_dropout: float = 0.1,
):
    
    logger.info(f"Running DPO on model: {hf_model_path} with dataset: {hf_dataset_path}")
    logger.info(f"cuda current device: f{torch.cuda.current_device()}")
