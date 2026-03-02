"""Tests for multiturn_dataset_to_dpo."""

import pytest
from datasets import Dataset

from collabllm.data_processing.dataset_utils import (
    multiturn_dataset_to_dpo,
    DEFAULT_SYSTEM_PROMPT,
)


def make_dataset(rows):
    return Dataset.from_list(rows)


def make_row(conv_id, turn_id, score, completion, prompt=None, single_turn_prompt="q", single_turn_completion="a"):
    return {
        "conv_id": conv_id,
        "turn_id": turn_id,
        "score": score,
        "completion": completion,
        "prompt": prompt or [{"role": "user", "content": "Hello"}],
        "single_turn_prompt": single_turn_prompt,
        "single_turn_completion": single_turn_completion,
    }


class TestBasicStructure:

    def test_output_columns(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "good answer"),
            make_row("c1", 0, 0.3, "bad answer"),
        ])
        result = multiturn_dataset_to_dpo(ds)
        row = result["train"][0]
        assert set(row.keys()) >= {"prompt", "chosen", "rejected", "chosen_score", "rejected_score",
                                    "messages", "single_turn_prompt", "single_turn_completion"}

    def test_chosen_is_highest_score(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "good answer"),
            make_row("c1", 0, 0.3, "bad answer"),
        ])
        result = multiturn_dataset_to_dpo(ds)
        row = result["train"][0]
        assert row["chosen"][0]["content"] == "good answer"
        assert row["rejected"][0]["content"] == "bad answer"

    def test_chosen_rejected_scores(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "good"),
            make_row("c1", 0, 0.3, "bad"),
        ])
        result = multiturn_dataset_to_dpo(ds)
        row = result["train"][0]
        assert row["chosen_score"] == pytest.approx(0.9)
        assert row["rejected_score"] == pytest.approx(0.3)

    def test_prompt_has_system_message_prepended(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "good"),
            make_row("c1", 0, 0.3, "bad"),
        ])
        result = multiturn_dataset_to_dpo(ds)
        prompt = result["train"][0]["prompt"]
        assert prompt[0]["role"] == "system"
        assert prompt[0]["content"] == DEFAULT_SYSTEM_PROMPT

    def test_custom_system_prompt(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "good"),
            make_row("c1", 0, 0.3, "bad"),
        ])
        result = multiturn_dataset_to_dpo(ds, system_prompt="Custom prompt")
        assert result["train"][0]["prompt"][0]["content"] == "Custom prompt"

    def test_chosen_rejected_are_assistant_messages(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "good"),
            make_row("c1", 0, 0.3, "bad"),
        ])
        result = multiturn_dataset_to_dpo(ds)
        row = result["train"][0]
        assert row["chosen"][0]["role"] == "assistant"
        assert row["rejected"][0]["role"] == "assistant"


class TestTurnSelection:

    def test_selects_highest_turn_id(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "turn0 good"),
            make_row("c1", 0, 0.3, "turn0 bad"),
            make_row("c1", 1, 0.8, "turn1 good"),
            make_row("c1", 1, 0.2, "turn1 bad"),
        ])
        result = multiturn_dataset_to_dpo(ds)
        assert len(result["train"]) == 1
        row = result["train"][0]
        assert row["chosen"][0]["content"] == "turn1 good"
        assert row["rejected"][0]["content"] == "turn1 bad"

    def test_one_row_per_conv_id(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "c1 good"),
            make_row("c1", 0, 0.3, "c1 bad"),
            make_row("c2", 0, 0.8, "c2 good"),
            make_row("c2", 0, 0.1, "c2 bad"),
        ])
        result = multiturn_dataset_to_dpo(ds)
        assert len(result["train"]) == 2


class TestFiltering:

    def test_min_score_gap_filters_close_pairs(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "good"),
            make_row("c1", 0, 0.85, "almost as good"),  # gap = 0.05
        ])
        result = multiturn_dataset_to_dpo(ds, min_score_gap=0.1)
        assert len(result["train"]) == 0

    def test_min_score_gap_keeps_wide_pairs(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "good"),
            make_row("c1", 0, 0.3, "bad"),  # gap = 0.6
        ])
        result = multiturn_dataset_to_dpo(ds, min_score_gap=0.5)
        assert len(result["train"]) == 1

    def test_skips_single_response_turns(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "only response"),
        ])
        result = multiturn_dataset_to_dpo(ds)
        assert len(result["train"]) == 0

    def test_skips_identical_score_turns(self):
        ds = make_dataset([
            make_row("c1", 0, 0.7, "response A"),
            make_row("c1", 0, 0.7, "response B"),
        ])
        result = multiturn_dataset_to_dpo(ds)
        assert len(result["train"]) == 0

    def test_empty_dataset(self):
        ds = make_dataset([])
        result = multiturn_dataset_to_dpo(ds)
        assert len(result["train"]) == 0


class TestEvalSplit:

    def test_eval_split(self):
        rows = []
        for i in range(10):
            rows += [
                make_row(f"c{i}", 0, 0.9, "good"),
                make_row(f"c{i}", 0, 0.1, "bad"),
            ]
        ds = make_dataset(rows)
        result = multiturn_dataset_to_dpo(ds, eval_ratio=0.2)
        assert len(result["eval"]) == 2
        assert len(result["train"]) == 8

    def test_no_eval_split_by_default(self):
        ds = make_dataset([
            make_row("c1", 0, 0.9, "good"),
            make_row("c1", 0, 0.3, "bad"),
        ])
        result = multiturn_dataset_to_dpo(ds)
        assert len(result["eval"]) == 0
        assert len(result["train"]) == 1
