# Agent Notes

> **AGENTS: This is a living document. Update it after every training/eval run and whenever you learn something useful about the codebase or environment. Keep it concise to avoid wasting tokens.**

---

## Environment

- **GPU:** NVIDIA A40, 46 GB VRAM
- **CPU:** Intel Xeon Gold 6342 @ 2.80GHz, 503 GB RAM
- **Platform:** RunPod
- **Key packages:** torch 2.9.1+cu128, transformers 4.57.6, trl 0.27.2, peft 0.18.1, vllm 0.15.1
- **Venv:** `.venv/` — activate with `source .venv/bin/activate`, install with `uv sync`
- **Secrets:** On RunPod use `--parse_secrets_runpod` to load HF_TOKEN/WANDB_API_KEY. On current pod they're already in env.

## Codebase Quick Reference

| Path | Purpose |
|---|---|
| `scripts/train_sft.py` | SFT training entrypoint |
| `scripts/eval.py` | Eval entrypoint (HF or vLLM backend) |
| `collabllm/evaluation/doc_writing.py` | Eval logic: multi-turn rollouts, BLEU + interactivity scoring |
| `collabllm/simulation/simulator.py` | ChatSimulator: orchestrates assistant/user turns |
| `collabllm/simulation/assistant.py` | LocalAssistant (HF) and VLLMAssistant |
| `collabllm/data_processing/dataset_utils.py` | `multiturn_dataset_to_sft()` — data prep |

## Gotchas & Learnings

- **GPU memory is tight.** vLLM + 4-bit Llama-8B uses ~44/46 GB. Don't load multiple models at once. Training and eval cannot overlap.
- **vLLM speed degrades with context length.** ~60 tok/s early in conversation, drops as turns accumulate.
- **`--use_4bit` only affects the HF eval path**, not vLLM (vLLM handles its own quantization).
- **`--eval_ratio 0.01`** on `collabllm-multiturn-medium-large` yields ~5 samples.
- **Training output dirs** are named `sft-{model}_{dataset}_{timestamp}/`.
- **Adapters auto-push to HF Hub** and log to wandb (project: `huggingface`).

## Run Log

### Run 3: Eval (vLLM) — 2026-02-06 ~00:31 UTC — COMPLETE
- **Command:** `python scripts/eval.py --dataset_path collabllm/collabllm-multiturn-medium-large --model_path meta-llama/Llama-3.1-8B-Instruct --lora_path boreasg/sft-llama8b_test-test_20260206_000112 --use_4bit --eval_ratio 0.01 --max_turns 14 --num_samples 4 --use_vllm`
- **Aggregated results:** avg_bleu=0.2727, avg_interactivity=0.52, avg_tokens=6097
- **Per-sample:**
  - Sample 1: BLEU 0.2563, itr 0.00, 10024 tok, 14 turns (hit max)
  - Sample 2: BLEU 0.3141, itr 1.00, 2776 tok, 6 turns (user terminated)
  - Sample 3: BLEU 0.3304, itr 0.70, 5344 tok
  - Sample 4: BLEU 0.1666, itr 0.00, 9562 tok
  - Sample 5: BLEU 0.2961, itr 0.90, 2777 tok
- **Notes:** Low interactivity on samples 1 & 4 (model monologued to max turns without engaging user). Higher BLEU correlated with shorter, more interactive conversations.

### Run 2: SFT Training — 2026-02-06 00:01 UTC — COMPLETE
- **Output:** `sft-llama8b_test-test_20260206_000112/`, `results/`
- **wandb:** `le3qriu2` | **Hub:** `boreasg/sft-llama8b_test-test_20260206_000112`
- **Config:** Llama-3.1-8B-Instruct, QLoRA 4-bit NF4, LoRA r=16 a=16 dropout=0.1, targets q/k/v/o_proj, batch=4, lr=2e-5, 1 epoch, bf16, gradient_checkpointing
- **Dataset:** `collabllm/collabllm-multiturn-medium-large`

### Run 1: SFT Training — 2026-02-05 23:44 UTC — COMPLETE
- **Output:** `sft-llama8b_test-test_20260205_234358/`
- **wandb:** `ee6ursav`
- **Notes:** First training run. Completed successfully, produced LoRA adapter.
