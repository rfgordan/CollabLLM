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
from collabllm.evaluation import evaluate_model, EvalResult

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
    return parser.parse_args()

def _log_trainable_params(model: torch.nn.Module) -> None:
    """ Log the number of trainable parameters in the model. """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total

    s = (
        f"trainable params: {trainable:,} || "
        f"all params: {total:,} || "
        f"trainable%: {pct:.4f}"
    )
    logger.info(s)

def load_and_train_sft(
        hf_model_path: str, 
        hf_dataset_path: str,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        parse_secrets_runpod: bool = False,
        run_eval: bool = False,
        output_name_tag: str = "default"):
    
    """ Load a Hugging Face model and perform supervised fine-tuning (SFT) on the provided dataset. """

    if parse_secrets_runpod:
        os.environ["HF_TOKEN"] = os.environ.get("RUNPOD_SECRET_HF_TOKEN", "") or os.environ.get("HF_TOKEN", "")
        os.environ["WANDB_API_KEY"] = os.environ.get("RUNPOD_SECRET_WANDB_API_KEY", "") or os.environ.get("WANDB_API_KEY", "")

    logger.info(f"Running SFT on model: {hf_model_path} with dataset: {hf_dataset_path}")
    logger.info(f"cuda current device: f{torch.cuda.current_device()}")

    # # use bf16 if supported by GPU
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # dtype = torch.float16

    dataset = load_dataset(
        hf_dataset_path, 
        cache_dir="./data_cache",
        split="train"
        )
    
    logger.info(f"Dataset {hf_dataset_path} loaded.\n {dataset}")

    dataset_clean = multiturn_dataset_to_sft(dataset, eval_ratio=0.01, lower_bound_metric=0.1)

    logger.info(f"Dataset {hf_dataset_path} preprocessed.\n {dataset_clean}")


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        # bnb_4bit_quant_storage=dtype,
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
        device_map={"":0}
    )

    model.config.use_cache = False # TODO: remove if no VRAM bottleneck?

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_path,
        cache_dir="./model_cache",
        use_fast=True,
    )

    logger.info(f"Model {hf_model_path} loaded with memory footprint: {model.get_memory_footprint()/(1024**3):.2f} GB")

    # LORA setup
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=16,
        bias="none",
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    run_name: str = f"sft-{output_name_tag}-{get_timebased_filename()}"

    # training
    training_arguments = SFTConfig(
        output_dir="./results",
        hub_model_id=run_name,
        push_to_hub=True,
        run_name=run_name,  # This names your W&B run
        report_to="wandb",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        logging_steps=True,
        learning_rate=learning_rate,
        gradient_checkpointing=True,
        num_train_epochs=2,
        eval_strategy="epoch",
        # eval_steps=500,
        save_strategy="epoch",
        # packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_clean["train"],
        eval_dataset=dataset_clean["eval"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
        # callbacks=[CustomMetricsCallback()],
    )

    logger.info("SFTTrainer initialized. Trainable parameters:")
    _log_trainable_params(trainer.model)
    trainer.model.print_trainable_parameters()
    
    trainer.train()

    # save trained model to hf?
    model.save_pretrained(f"./{run_name}")
    tokenizer.save_pretrained(f"./{run_name}")

    # custom eval on data?
    
    eval_result = evaluate_model(
        model=trainer.model,
        tokenizer=tokenizer,
        dataset=dataset_clean,
        num_samples=4,
    ) if run_eval else None

    logger.info(f"Eval result: {eval_result}")
    logger.info(f"SFT training completed for model: {hf_model_path}. Eval result: {eval_result}")

    # trainer.push_to_hub(
    #     f"sft-model-{output_name_tag}-{get_timebased_filename()}",
    #     organization="boreasg",  # replace with your HF org or username
    #     private=True,
    # )

    # wandb.finish()

    if wandb.run is not None:
        if eval_result is not None:
            wandb.run.summary.update(eval_result.as_dict())
            wandb.run.log(eval_result.as_dict())
        wandb.run.finish()

def main() -> None:
    args = parse_args()

    # # Distributed setup
    # local_rank = int(os.environ['LOCAL_RANK'])
    # dist.init_process_group(backend='nccl', init_method=None)
    # torch.cuda.set_device(local_rank)
    # dist.barrier()

    # # DeepSpeed zero
    # ds_cfg = {
    #     "zero_optimization": {
    #         "stage": 2,
    #         "overlap_comm": False,
    #         "reduce_bucket_size": "auto",
    #         "contiguous_gradients": True,
    #         "offload_optimizer": {"device": "none"},
    #         "offload_param": {"device": "none"},
    #     },
    #     "gradient_clipping": "auto",
    #     "train_batch_size": "auto",
    #     "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
    #     "gradient_accumulation_steps": args.gradient_accumulation_steps,
    #     "steps_per_print": 200,
    # }

    load_and_train_sft(
        hf_model_path=args.hf_model_path,
        hf_dataset_path=args.hf_dataset_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        parse_secrets_runpod=args.parse_secrets_runpod,
        run_eval=args.run_eval,
        output_name_tag=args.output_name_tag,
    )

if __name__ == "__main__":
    main()