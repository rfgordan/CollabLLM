# general python imports
from typing import Any
import logging
import argparse, os

# HF / TRL / PyTorch stack
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from transformers import AutoTokenizer
import transformers
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
from datasets import load_dataset
import torch
import wandb

# project code
from collabllm.data_processing.dataset_utils import multiturn_dataset_to_dpo
from collabllm.training.train_utils import get_timebased_filename
from collabllm.evaluation import evaluate_checkpoint_vllm, EvalResult

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline DPO Fine-Tuning Script")
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
        "--eval_ratio",
        type=float,
        default=0.05,
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
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta (KL penalty coefficient).",
    )
    parser.add_argument(
        "--min_score_gap",
        type=float,
        default=0.0,
        help="Minimum score gap between chosen and rejected responses.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum total sequence length (prompt + response).",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Maximum prompt length; prompt is truncated before response if exceeded.",
    )
    return parser.parse_args()

def _get_param_counts(model: torch.nn.Module) -> dict:
    """Get trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0
    logger.info(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")
    return {"trainable_params": trainable, "total_params": total}

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
        beta: float = 0.1,
        min_score_gap: float = 0.0,
        max_length: int = 4096,
        max_prompt_length: int = 2048,
):
    """Load a Hugging Face model and perform offline DPO fine-tuning on the provided dataset."""

    if parse_secrets_runpod:
        os.environ["HF_TOKEN"] = os.environ.get("RUNPOD_SECRET_HF_TOKEN", "") or os.environ.get("HF_TOKEN", "")
        os.environ["WANDB_API_KEY"] = os.environ.get("RUNPOD_SECRET_WANDB_API_KEY", "") or os.environ.get("WANDB_API_KEY", "")

    logger.info(f"Running DPO on model: {hf_model_path} with dataset: {hf_dataset_path}")
    logger.info(f"cuda current device: {torch.cuda.current_device()}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    dataset = load_dataset(
        hf_dataset_path,
        cache_dir="./data_cache",
        split="train",
    )

    logger.info(f"Dataset {hf_dataset_path} loaded.\n {dataset}")

    dataset_clean = multiturn_dataset_to_dpo(
        dataset,
        eval_ratio=eval_ratio,
        min_score_gap=min_score_gap,
    )

    logger.info(f"Dataset {hf_dataset_path} preprocessed.\n {dataset_clean}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )

    is_flash_attn_2_available = transformers.utils.is_flash_attn_2_available()
    if is_flash_attn_2_available:
        logger.info("✅ FlashAttention-2 is installed and hardware-compatible.")
    else:
        logger.warning("❌ FlashAttention-2 is NOT available.")

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        cache_dir="./model_cache",
        quantization_config=bnb_config,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available else "sdpa",
        device_map={"": 0},
    )

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_path,
        cache_dir="./model_cache",
        use_fast=True,
    )

    logger.info(f"Model {hf_model_path} loaded with memory footprint: {model.get_memory_footprint()/(1024**3):.2f} GB")

    # LoRA setup — ref_model=None lets DPOTrainer use the base model with adapters
    # disabled as the implicit reference, saving GPU memory
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    run_name: str = f"dpo-{output_name_tag}-{get_timebased_filename()}"

    training_arguments = DPOConfig(
        output_dir="./results",
        hub_model_id=run_name,
        push_to_hub=True,
        run_name=run_name,
        report_to="wandb",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        logging_steps=True,
        learning_rate=learning_rate,
        gradient_checkpointing=True,
        num_train_epochs=num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # uses base model with adapters disabled as implicit reference
        args=training_arguments,
        train_dataset=dataset_clean["train"],
        eval_dataset=dataset_clean["eval"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("DPOTrainer initialized.")
    param_counts = _get_param_counts(trainer.model)

    trainer.train()

    trainer.model.save_pretrained(f"./{run_name}")
    tokenizer.save_pretrained(f"./{run_name}")

    del trainer
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Cleared training model from GPU memory")

    eval_result = evaluate_checkpoint_vllm(
        model_path=hf_model_path,
        dataset=dataset_clean,
        lora_path=f"./{run_name}",
        num_samples=4,
        max_turns=max_turns,
        gpu_memory_utilization=0.8,
    ) if run_eval else None

    logger.info(f"Eval result: {eval_result}")
    logger.info(f"DPO training completed for model: {hf_model_path}. Eval result: {eval_result}")

    if wandb.run is not None:
        summary = {
            "model_path": hf_model_path,
            "dataset_path": hf_dataset_path,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": training_arguments.num_train_epochs,
            "lora_r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "beta": beta,
            "min_score_gap": min_score_gap,
            **param_counts,
        }
        if eval_result is not None:
            summary.update(eval_result.as_dict())
        wandb.run.summary.update(summary)
        wandb.run.finish()


def main() -> None:
    args = parse_args()
    load_and_train_dpo(
        hf_model_path=args.hf_model_path,
        hf_dataset_path=args.hf_dataset_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        parse_secrets_runpod=args.parse_secrets_runpod,
        run_eval=args.run_eval,
        output_name_tag=args.output_name_tag,
        max_turns=args.max_turns,
        eval_ratio=args.eval_ratio,
        num_train_epochs=args.num_train_epochs,
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        beta=args.beta,
        min_score_gap=args.min_score_gap,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )


if __name__ == "__main__":
    main()
