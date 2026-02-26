# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CollabLLM is a Python project for supervised fine-tuning (SFT) of large language models using QLoRA with the HuggingFace/TRL stack. It processes multi-turn conversation datasets and fine-tunes causal language models with 4-bit quantization.

## Commands

### Environment Setup
```bash
uv sync                    # Install dependencies
source .venv/bin/activate  # Activate virtual environment
```

### Training
```bash
python scripts/train_sft.py \
  --hf_model_path <model> \
  --hf_dataset_path <dataset> \
  --learning_rate 2e-5 \
  --batch_size 4 \
  --output_name_tag <tag>
```

Add `--parse_secrets_runpod` when running on RunPod to load HF_TOKEN and WANDB_API_KEY from RunPod secrets.

## Architecture

### Core Package (`collabllm/`)
- **data_processing/dataset_utils.py**: Converts multi-turn HF datasets to SFT format. Key function `multiturn_dataset_to_sft()` selects the highest-scoring response per conversation turn and formats as chat messages with system prompt.
- **training/train_utils.py**: Utility functions for training (timestamp-based filenames).
- **simulation/**: Chat simulation environment for generating multi-turn conversations.
  - `ChatSimulator`: Orchestrates conversations between assistant and user models
  - `LocalAssistant`: Loads local HF models with optional LoRA/QLoRA support
  - `UserModel`: Abstract base for user simulation (extend for new providers)
  - `OpenAIUserModel`: OpenAI API implementation for user simulation

### Training Pipeline (`scripts/train_sft.py`)
Uses QLoRA (4-bit NF4 quantization) with LoRA adapters targeting attention projection layers (q/k/v/o_proj). Integrates with Weights & Biases for logging and pushes models to HuggingFace Hub.

### Known Optimization: VLLMAssistant
The current `LocalAssistant` uses HuggingFace `generate()` which re-encodes the full conversation from scratch each turn (no KV cache reuse across turns). The original CollabLLM paper uses vLLM with `LoRARequest` for efficient inference: vLLM holds the base model with `enable_lora=True`, and LoRA adapters are hot-swapped via `LoRARequest` after saving with `model.save_pretrained()`. This would give KV cache reuse, PagedAttention, and continuous batching. See `Wuyxin/collabllm/collabllm/simulation.py` for reference.

### Data Flow
1. Load multi-turn dataset from HF Hub
2. Filter by score threshold (`lower_bound_metric`) and select best response per turn
3. Format as chat messages with system prompt
4. Split into train/eval sets
5. Fine-tune with SFTTrainer using QLoRA

## Experiment Tracking

All experiment results and takeaways must be documented in `doc/` notes. After any experiment run, update or create the relevant file in `doc/` with results, observations, and conclusions.
