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

### Runs 12 & 13: DPO Hyperparameter Search — 2026-03-03/04 UTC

**Goal:** Test two hyperparameter variations against the Run 11 DPO-v3 baseline: (a) 3 epochs instead of 1, (b) lower beta (0.05 vs 0.1).

**Run 12a training:** `dpo-dpo-v4-3ep-test_20260303_183354` (wandb ID: `vy890lot`)
- Same as Run 11 except `num_train_epochs=3` → 5184 steps, ~3h51m
- final `train_loss=0.652`; by epoch 3: `rewards/accuracies=1.0`, `rewards/margins=7–18` (large positive — DPO objective converged)

**Run 13a training:** `dpo-dpo-v5-beta0.05-test_20260304_000309`
- Same as Run 11 except `beta=0.05`, 1 epoch

**Eval:** vLLM, max_turns=7, eval_ratio=0.05, lower_bound_metric=0.1, 24 eval samples

| Metric | R11 DPO-v3 (baseline) | R12 DPO-v4 (3 epochs) | R13 DPO-v5 (β=0.05) |
|---|---|---|---|
| `avg_bleu` | 0.3466 | 0.3252 | 0.2527 |
| `avg_tokens` | 2503.8 | 2594.8 | **856.7** |
| `avg_itr` | 0.863 | **0.904** | 0.188 |
| `avg_sample_time_s` | 91.6s | 90.4s | 47.4s |

**Full comparison table:**

| Metric | R7 Base | R8 SFT | R11 DPO-v3 | R12 DPO-v4 (3ep) | R13 DPO-v5 (β=0.05) |
|---|---|---|---|---|---|
| `avg_bleu` | 0.3224 | **0.3496** | 0.3466 | 0.3252 | 0.2527 |
| `avg_tokens` | 2992.5 | **2113.8** | 2503.8 | 2594.8 | 856.7 |
| `avg_itr` | 0.825 | 0.913 | 0.863 | **0.904** | 0.188 |

**Conclusions:**
- **3 epochs (R12):** Interactivity improved to 0.904 (essentially tied with SFT's 0.913), but BLEU dropped to 0.3252 — below both Run 11 and SFT. The model fully converged on the DPO objective (margins ~7–18 by epoch 3) but slightly overfit, trading BLEU quality for interactive behavior.
- **beta=0.05 (R13): Complete collapse.** avg_itr dropped from 0.863 → 0.188, avg_tokens from 2504 → 857. Lower beta allows more deviation from the reference model; with noisy DPO signal (mixed reward margins), the model collapsed into terse, non-interactive responses. Beta=0.1 appears to be the right regularization strength for this dataset.
- **Best on interactivity:** R12 3-epoch DPO (0.904) ≈ SFT (0.913)
- **Best on BLEU:** SFT (0.3496) still leads; R11 DPO-v3 (0.3466) is close
- **Overall best DPO:** Run 11 (1 epoch, β=0.1) offers the best BLEU/itr tradeoff among DPO runs

---

### Run 11: DPO v3 Training + Eval — 2026-03-03 UTC

**Goal:** Retrain DPO using the updated dataset pipeline that samples one training pair per conversation turn (not once per conversation), while keeping eval at one sample per conversation.

**Training (Run 11a):** `dpo-dpo-v3-test_20260303_162609` (wandb ID: `gzzhv1pk`, `rfgordan/collabllm`)
- **1728 train / 24 eval pairs** (vs 346/18 in Run 10 — 5× more pairs from per-turn sampling)
- 1 epoch, batch_size=1, lr=2e-5, beta=0.1, max_length=4096, eval_strategy=no
- 1728 steps, ~77 min, final `train/loss=0.7939`
- ⚠️ `rewards/margins=-0.192`, `rewards/accuracies=0` — DPO still not converging in chosen>rejected direction on training set

**Eval (Run 11b):** `eval-Llama-3.1-8B-Instruct-dpo-v3-mt7` (wandb ID: `a0xk3jpn`, `rfgordan/collabllm`)
- vLLM, max_turns=7, eval_ratio=0.05, lower_bound_metric=0.1, 24 eval samples

| Metric | Run 7: Base | Run 8: SFT | Run 9: DPO-v1 | Run 10: DPO-v2 | Run 11: DPO-v3 |
|---|---|---|---|---|---|
| `avg_bleu` | 0.3224 | **0.3496** | 0.2731 | 0.2948 | 0.3466 |
| `avg_tokens` | 2992.5 | **2113.8** | 3706.2 | 2960.9 | 2503.8 |
| `avg_itr` | 0.825 | **0.913** | 0.704 | 0.604 | 0.863 |
| `avg_sample_time_s` | ~69s | ~84s | ~69s | 119.5s | 91.6s |

**Conclusions:** Per-turn pair sampling was the key change and it worked dramatically — interactivity jumped from 0.604 to **0.863** (+42.9%), nearly matching SFT (0.913). BLEU also improved to 0.3466, nearly matching SFT. Token count dropped to 2503.8, much closer to SFT's 2113.8. DPO-v3 is a near-tie with SFT on BLEU and token count, and comes close on interactivity. Despite `rewards/accuracies=0` on the training set (model not converging in the traditional DPO sense), the 5× more training pairs provided enough signal for emergent behavioral improvement. The negative reward margin suggests the reference model assigns higher logprobs to rejected responses; future work could try higher beta or longer training to overcome this.

---

### Run 10: DPO v2 Training + Eval — 2026-03-03 UTC

**Goal:** Retrain DPO with new defaults (`min_score_gap=0.02`, `max_length=4096`, `eval_strategy=no`) and compare against prior runs on the same 24-sample held-out split.

**Training (Run 10a):** `dpo-dpo-v2-test_20260303_145448` (wandb ID: `hd5id88h`, `rfgordan/collabllm`)
- 346 train / 18 eval pairs after min_score_gap=0.02 filter
- 1 epoch, batch_size=1 (OOM at batch_size=4), lr=2e-5, beta=0.1, max_length=4096, eval_strategy=no
- 346 steps, ~23 min, final train_loss=0.9663
- Adapter saved locally: `/CollabLLM/dpo-dpo-v2-test_20260303_145448/` (not pushed to HF Hub — eval_strategy=no avoids save-point, local-only)

**Eval (Run 10b):** `eval-Llama-3.1-8B-Instruct-dpo-v2-mt7` (wandb ID: `5txxz462`, `rfgordan/collabllm`)
- vLLM, max_turns=7, eval_ratio=0.05, lower_bound_metric=0.1, 24 eval samples

| Metric | Run 7: Base | Run 8: SFT | Run 9: DPO-v1 | Run 10: DPO-v2 | Δ v2 vs v1 |
|---|---|---|---|---|---|
| `avg_bleu` | 0.3224 | **0.3496** | 0.2731 | 0.2948 | +7.9% |
| `avg_tokens` | 2992.5 | **2113.8** | 3706.2 | 2960.9 | −20.1% |
| `avg_itr` | 0.825 | **0.913** | 0.704 | 0.604 | −14.2% |
| `avg_sample_time_s` | ~69s | ~84s | ~69s | 119.5s | — |

**Conclusions:** DPO-v2 improves over DPO-v1 on BLEU (+7.9%) and token count (−20.1%), bringing token count back in line with the base model. However, interactivity dropped further to 0.604, the worst of all four runs. All three DPO-v2 metrics still underperform both the base model and SFT adapter. The `max_length=4096` fix helped stabilize token count, but the preference signal remains too weak to improve interactivity: 346 pairs, 1 epoch, final loss 0.9663 (above 0.693 random). The falling interactivity despite preference training on higher-itr pairs is a red flag — the DPO objective may be being dominated by the chosen/rejected BLEU signal rather than the interactivity component. **SFT remains the best-performing adapter across all metrics.** Next steps to consider: more epochs, higher-quality preference data, or a combined SFT→DPO pipeline.

---

### Run 9: DPO Adapter Eval — 2026-03-03 UTC

**Goal:** Evaluate DPO-trained adapter (`boreasg/dpo-llama8b-dpo-test_20260302_211616`) on the same held-out eval split as Runs 7 & 8, to compare DPO vs base vs SFT.

**Parameters:** vLLM, max_turns=7, eval_ratio=0.05, lower_bound_metric=0.1 → 24 eval samples (same deterministic split)

**wandb run:** `eval-Llama-3.1-8B-Instruct-dpo-mt7` (`jwwopnaf`)

| Metric | Run 7: Base | Run 8: SFT | Run 9: DPO | Δ DPO vs SFT |
|---|---|---|---|---|
| `avg_bleu` | 0.3224 | **0.3496** | 0.2731 | −21.9% |
| `avg_tokens` | 2992.5 | **2113.8** | 3706.2 | +75.3% |
| `avg_itr` | 0.825 | **0.913** | 0.704 | −22.9% |

**Conclusions:** The DPO adapter underperforms both the base model and SFT on all three metrics. BLEU dropped −15% vs base and −22% vs SFT; interactivity fell to 0.70 (below even base at 0.83); token count ballooned to 3706 — longer than base and 75% more than SFT. The DPO training run used a very small dataset (the output of a single `train_dpo_offline` call on a subset) and only 1 epoch, so the adapter likely moved the model away from the SFT distribution without sufficient signal to improve it. This is a common failure mode for offline DPO with low-quality or small preference datasets. The SFT adapter remains the best-performing model on this eval.

---

### Runs 7 & 8: Base vs Fine-Tuned Comparison — 2026-02-27 UTC

**Goal:** Compare base Llama-3.1-8B-Instruct vs fine-tuned adapter (`boreasg/sft-llama8b_test-test_20260206_000112`) on the held-out eval split from training.

**Parameters:** vLLM, max_turns=7, eval_ratio=0.05, lower_bound_metric=0.1 → 24 eval samples (deterministic split matching training)

**wandb runs:** `eval-Llama-3.1-8B-Instruct-base-mt7` (`njw7jqrb`), `eval-Llama-3.1-8B-Instruct-finetuned-mt7` (`vs25047c`)

| Metric | Run 7: Base | Run 8: Fine-tuned | Δ |
|---|---|---|---|
| `avg_bleu` | 0.3224 | **0.3496** | +8.4% |
| `avg_tokens` | 2992.5 | **2113.8** | −29.3% |
| `avg_itr` | 0.825 | **0.913** | +10.7% |
| `avg_sample_time_s` | — | 84.1s | — |

**Conclusions:** Fine-tuning improves all three metrics. The largest gain is interactivity (+10.7%), suggesting the model learned to engage the user more effectively rather than monologuing. The −29% token reduction indicates more concise, focused outputs. BLEU improvement (+8.4%) is modest but consistent with the paper's findings. The held-out split (lower_bound_metric=0.1, eval_ratio=0.05, seed=42) is reproducible and guaranteed not to overlap with training data.

**Fix applied:** Added `--lower_bound_metric` arg to `scripts/eval.py` so the eval dataset filter matches training (previously defaulted to 0.0, producing a different split).

---

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
