from typing import List, Dict, Any, Optional, Optional
import datasets
from datasets import Dataset, DatasetDict, load_dataset

import logging
import random
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Provide detailed and accurate responses to the user's queries."

def _uniform_split(dataset: Dataset, eval_ratio: float = 0, seed: int = 42) -> Dataset:

    if eval_ratio >= 1.0:
        logger.warning("eval_ratio >= 1.0, the entire dataset will be used for evaluation.")

    k = int(len(dataset) * eval_ratio)
    k = min(k, len(dataset))

    random.seed(seed)
    eval_idx = set(random.sample(range(len(dataset)), k=k))
    train_idx = set(range(len(dataset))) - eval_idx

    return DatasetDict({
        "train": dataset.select(train_idx),
        "eval": dataset.select(sorted(eval_idx))
    })

def multiturn_dataset_to_dpo(
        dataset: Dataset,
        eval_ratio: Optional[float] = 0.0,
        min_score_gap: Optional[float] = 0.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> DatasetDict:
    """Convert a multiturn dataset to DPO format.

    For every (conv_id, turn_id) that has at least two responses with a score
    gap >= min_score_gap, emits a DPO pair: highest-scoring response as chosen,
    lowest-scoring as rejected. Multiple turns per conversation each produce a
    separate training row.

    Output columns: prompt, chosen, rejected, chosen_score, rejected_score,
    messages, single_turn_prompt, single_turn_completion, where
    prompt/chosen/rejected are lists of message dicts (TRL conversational format).
    """
    # Group rows by (conv_id, turn_id) — each group may have multiple responses
    groups: Dict[Any, List[Any]] = {}
    for row in dataset:
        key = (row["conv_id"], row["turn_id"])
        groups.setdefault(key, []).append(row)

    out_rows = []
    for rows in groups.values():
        if len(rows) < 2:
            continue

        rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
        chosen_row = rows_sorted[0]
        rejected_row = rows_sorted[-1]

        gap = chosen_row["score"] - rejected_row["score"]
        if gap <= 0 or gap < min_score_gap:
            continue

        prompt = [{"role": "system", "content": system_prompt}] + chosen_row["prompt"]
        chosen = [{"role": "assistant", "content": chosen_row["completion"]}]
        out_rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": [{"role": "assistant", "content": rejected_row["completion"]}],
            "messages": prompt + chosen,
            "chosen_score": chosen_row["score"],
            "rejected_score": rejected_row["score"],
            "single_turn_prompt": chosen_row["single_turn_prompt"],
            "single_turn_completion": chosen_row["single_turn_completion"],
        })

    out_dataset = datasets.Dataset.from_list(out_rows)
    return _uniform_split(out_dataset, eval_ratio=eval_ratio)


# given a multiturn dataset, from HF, map to SFT format according to choice logic
def multiturn_dataset_to_sft(
        dataset: Dataset,
        eval_ratio : Optional[float] = 0.0,
        lower_bound_metric : Optional[float] = 0.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Dataset:
    
    out_rows = []
    conv_id_to_final_row = {}
    
    for row in dataset:
        try:
            prev = conv_id_to_final_row.get(row["conv_id"])
            if prev is None or row["turn_id"] > prev['turn_id'] or (row['turn_id'] == prev['turn_id'] and row['score'] > prev['score']):
                conv_id_to_final_row[row["conv_id"]] = row
        except Exception as e:
            logger.error(f"Failed processing dataset row: {row} with error: {e}")
            continue


    for row in conv_id_to_final_row.values():
        if row['score'] < lower_bound_metric:
            continue

        combined_conversation = {
            "messages": [{"role": "system", "content": system_prompt}] + row["prompt"] + [{"role": "assistant", "content": row["completion"]} ],
            "single_turn_prompt": row["single_turn_prompt"],
            "single_turn_completion": row["single_turn_completion"],
        }

        out_rows.append(combined_conversation)

    out_dataset = datasets.Dataset.from_list(out_rows)
    return _uniform_split(out_dataset, eval_ratio=eval_ratio)
