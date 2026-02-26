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
- **vLLM without `enforce_eager` is ~3.1x faster than HF** (32s vs 100s avg_assistant_time_s, sequential single-request eval). With `enforce_eager=True` it is ~1.74x *slower* — CUDA graphs are the dominant factor. Use vLLM by default; only fall back to `enforce_eager` if the triton ptxas permission error appears.
- **vLLM speed degrades with context length.** ~60 tok/s early in conversation, drops as turns accumulate.
- **`--use_4bit` only affects the HF eval path**, not vLLM (vLLM handles its own quantization).
- **`--eval_ratio 0.01`** on `collabllm-multiturn-medium-large` yields ~5 samples.
- **Training output dirs** are named `sft-{model}_{dataset}_{timestamp}/`.
- **Adapters auto-push to HF Hub** and log to wandb (project: `huggingface`).

## Run Log

### Run 6: vLLM without enforce_eager — 2026-02-26 ~23:00 UTC

**Parameters:** same as Runs 4 & 5. `enforce_eager` removed from `vllm_assistant.py`.

| Metric | HF (Run 4) | vLLM + eager (Run 5) | vLLM no eager (Run 6) |
|---|---|---|---|
| `avg_assistant_time_s` | 99.67s | 173.27s | **32.08s** |
| `avg_sample_time_s` | 140.10s | 220.58s | 69.12s |
| `avg_bleu` | 0.2490 | 0.3086 | 0.3110 |
| `avg_tokens` | 1547.2 | 2225.6 | 1717.0 |
| `avg_itr` | 0.60 | 0.82 | 0.60 |
| `words/sec` (approx) | ~15.5 | ~12.8 | **~53.5** |
| **Speedup vs HF** | 1.0x | 0.58x | **3.11x** |

**Notes:** Removing `enforce_eager` unlocked CUDA graphs (~63 tok/s output vs ~13 tok/s with `enforce_eager`). vLLM without `enforce_eager` is **3.1x faster than HF** and **5.4x faster than vLLM+eager** on sequential single-request eval. The triton ptxas permission error that originally motivated `enforce_eager` did not appear on this run. Recommend vLLM without `enforce_eager` as the default; only add it back if the ptxas error resurfaces.

---

### Runs 4 & 5: Speed Comparison (HF vs vLLM) — 2026-02-26 ~21:21–21:49 UTC

**Parameters:** Llama-3.1-8B-Instruct + LoRA `boreasg/sft-llama8b_test-test_20260206_000112`,
`collabllm/collabllm-multiturn-medium-large`, eval_ratio=0.01 (~5 samples), max_turns=5
**Same samples:** guaranteed by hardcoded seed=42 in dataset split.

| Metric | HF (`--use_4bit`) | vLLM (`--use_vllm`) |
|---|---|---|
| `avg_assistant_time_s` | 99.67s | 173.27s |
| `avg_sample_time_s` | 140.10s | 220.58s |
| `avg_bleu` | 0.2490 | 0.3086 |
| `avg_tokens` | 1547.2 | 2225.6 |
| `avg_itr` | 0.60 | 0.82 |
| `words/sec` (approx) | ~15.5 | ~12.8 |
| **Speedup** | **1.0x (baseline)** | **0.58x (1.74x slower)** |

**Notes:** vLLM was unexpectedly ~1.74x **slower** than HF in this configuration. Three likely causes:
1. **`enforce_eager=True`** — disables CUDA graphs (required to avoid triton ptxas permission error on RunPod). This removes the primary compile-time optimization vLLM relies on.
2. **No batching benefit** — eval runs one request at a time sequentially; vLLM's PagedAttention / continuous batching advantages only manifest with concurrent requests.
3. **BitsAndBytes in vLLM is less mature** — bnb quantization support in vLLM adds per-request LoRA loading overhead via `LoRARequest`.

The quality difference (avg_itr 0.82 vs 0.60, avg_bleu 0.31 vs 0.25) reflects stochasticity in generation (different sampling seeds / decoding paths), not a real quality gap. vLLM also generated longer responses on average (2225 vs 1547 tokens).

**Conclusion:** vLLM provides no speed benefit here and is slower due to `enforce_eager`. For production use with many concurrent requests and CUDA graphs enabled, vLLM would likely win. For sequential single-request eval on RunPod, HF is faster.

---

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
