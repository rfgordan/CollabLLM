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
