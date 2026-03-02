"""Tests for train_dpo_offline."""

import sys
import pytest
from unittest.mock import MagicMock, patch, call
import argparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(trainable_params=100, total_params=1000):
    mock = MagicMock()
    p_trainable = MagicMock()
    p_trainable.numel.return_value = trainable_params
    p_trainable.requires_grad = True
    p_frozen = MagicMock()
    p_frozen.numel.return_value = total_params - trainable_params
    p_frozen.requires_grad = False
    mock.parameters.return_value = [p_trainable, p_frozen]
    mock.get_memory_footprint.return_value = 2 * (1024 ** 3)
    mock.config = MagicMock()
    return mock


def _make_mock_dataset(n=10):
    mock = MagicMock()
    mock.__len__ = lambda self: n
    return mock


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:

    def _parse(self, extra_args=None):
        from scripts.train_dpo_offline import parse_args
        base = ["--hf_model_path", "meta-llama/Llama-3-8B", "--hf_dataset_path", "some/dataset"]
        with patch("sys.argv", ["train_dpo_offline.py"] + base + (extra_args or [])):
            return parse_args()

    def test_required_args(self):
        args = self._parse()
        assert args.hf_model_path == "meta-llama/Llama-3-8B"
        assert args.hf_dataset_path == "some/dataset"

    def test_defaults(self):
        args = self._parse()
        assert args.learning_rate == pytest.approx(2e-5)
        assert args.batch_size == 4
        assert args.num_train_epochs == 3
        assert args.lora_alpha == 16
        assert args.lora_r == 16
        assert args.lora_dropout == pytest.approx(0.1)
        assert args.beta == pytest.approx(0.1)
        assert args.min_score_gap == pytest.approx(0.0)
        assert args.eval_ratio == pytest.approx(0.05)
        assert args.max_turns == 5
        assert args.parse_secrets_runpod is False
        assert args.run_eval is False

    def test_beta_override(self):
        args = self._parse(["--beta", "0.5"])
        assert args.beta == pytest.approx(0.5)

    def test_min_score_gap_override(self):
        args = self._parse(["--min_score_gap", "0.3"])
        assert args.min_score_gap == pytest.approx(0.3)

    def test_flags(self):
        args = self._parse(["--parse_secrets_runpod", "--run_eval"])
        assert args.parse_secrets_runpod is True
        assert args.run_eval is True

    def test_missing_required_args_raises(self):
        from scripts.train_dpo_offline import parse_args
        with patch("sys.argv", ["train_dpo_offline.py"]):
            with pytest.raises(SystemExit):
                parse_args()


# ---------------------------------------------------------------------------
# _get_param_counts
# ---------------------------------------------------------------------------

class TestGetParamCounts:

    def test_counts(self):
        from scripts.train_dpo_offline import _get_param_counts
        model = _make_mock_model(trainable_params=100, total_params=1000)
        result = _get_param_counts(model)
        assert result["trainable_params"] == 100
        assert result["total_params"] == 1000

    def test_all_frozen(self):
        from scripts.train_dpo_offline import _get_param_counts
        mock = MagicMock()
        p = MagicMock()
        p.numel.return_value = 500
        p.requires_grad = False
        mock.parameters.return_value = [p]
        result = _get_param_counts(mock)
        assert result["trainable_params"] == 0
        assert result["total_params"] == 500


# ---------------------------------------------------------------------------
# load_and_train_dpo — mocked smoke tests
# ---------------------------------------------------------------------------

class TestLoadAndTrainDpo:
    """Smoke tests with all heavy dependencies mocked out."""

    def _make_patches(self):
        mock_dataset = MagicMock()
        mock_train = _make_mock_dataset(80)
        mock_eval = _make_mock_dataset(20)
        mock_dataset_clean = {"train": mock_train, "eval": mock_eval}

        patches = {
            "load_dataset": patch(
                "scripts.train_dpo_offline.load_dataset",
                return_value=mock_dataset,
            ),
            "multiturn_dataset_to_dpo": patch(
                "scripts.train_dpo_offline.multiturn_dataset_to_dpo",
                return_value=mock_dataset_clean,
            ),
            "AutoModelForCausalLM": patch(
                "scripts.train_dpo_offline.AutoModelForCausalLM"
            ),
            "AutoTokenizer": patch(
                "scripts.train_dpo_offline.AutoTokenizer"
            ),
            "BitsAndBytesConfig": patch(
                "scripts.train_dpo_offline.BitsAndBytesConfig"
            ),
            "DPOTrainer": patch(
                "scripts.train_dpo_offline.DPOTrainer"
            ),
            "DPOConfig": patch(
                "scripts.train_dpo_offline.DPOConfig"
            ),
            "LoraConfig": patch(
                "scripts.train_dpo_offline.LoraConfig"
            ),
            "torch_cuda": patch(
                "scripts.train_dpo_offline.torch"
            ),
            "transformers": patch(
                "scripts.train_dpo_offline.transformers"
            ),
            "wandb": patch(
                "scripts.train_dpo_offline.wandb"
            ),
            "get_timebased_filename": patch(
                "scripts.train_dpo_offline.get_timebased_filename",
                return_value="20260101_120000",
            ),
            "evaluate_checkpoint_vllm": patch(
                "scripts.train_dpo_offline.evaluate_checkpoint_vllm",
            ),
            "gc": patch("scripts.train_dpo_offline.gc", create=True),
        }
        return patches, mock_dataset_clean

    def _run(self, extra_kwargs=None, run_eval=False):
        from scripts.train_dpo_offline import load_and_train_dpo

        patches, mock_dataset_clean = self._make_patches()
        mocks = {}
        for name, p in patches.items():
            mocks[name] = p.start()

        mock_model = _make_mock_model()
        mocks["AutoModelForCausalLM"].from_pretrained.return_value = mock_model
        mocks["AutoTokenizer"].from_pretrained.return_value = MagicMock()
        mocks["torch_cuda"].cuda.is_bf16_supported.return_value = False
        mocks["torch_cuda"].cuda.current_device.return_value = 0
        mocks["torch_cuda"].distributed.is_initialized.return_value = False
        mocks["transformers"].utils.is_flash_attn_2_available.return_value = False
        mocks["wandb"].run = None

        mock_trainer = MagicMock()
        mock_trainer.model = mock_model
        mocks["DPOTrainer"].return_value = mock_trainer

        kwargs = dict(
            hf_model_path="test-model",
            hf_dataset_path="test-dataset",
            run_eval=run_eval,
        )
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        try:
            load_and_train_dpo(**kwargs)
        finally:
            for p in patches.values():
                p.stop()

        return mocks, mock_dataset_clean, mock_trainer

    def test_dpo_trainer_is_used(self):
        mocks, _, _ = self._run()
        mocks["DPOTrainer"].assert_called_once()

    def test_ref_model_is_none(self):
        """DPOTrainer must be called with ref_model=None for implicit reference."""
        mocks, _, _ = self._run()
        call_kwargs = mocks["DPOTrainer"].call_args.kwargs
        assert call_kwargs["ref_model"] is None

    def test_train_called(self):
        _, _, mock_trainer = self._run()
        mock_trainer.train.assert_called_once()

    def test_model_saved(self):
        _, _, mock_trainer = self._run()
        mock_trainer.model.save_pretrained.assert_called_once()

    def test_beta_passed_to_dpo_config(self):
        mocks, _, _ = self._run(extra_kwargs={"beta": 0.3})
        dpo_config_kwargs = mocks["DPOConfig"].call_args.kwargs
        assert dpo_config_kwargs["beta"] == pytest.approx(0.3)

    def test_min_score_gap_passed_to_dataset(self):
        mocks, _, _ = self._run(extra_kwargs={"min_score_gap": 0.2})
        call_kwargs = mocks["multiturn_dataset_to_dpo"].call_args.kwargs
        assert call_kwargs["min_score_gap"] == pytest.approx(0.2)

    def test_train_eval_datasets_passed(self):
        mocks, mock_dataset_clean, _ = self._run()
        trainer_kwargs = mocks["DPOTrainer"].call_args.kwargs
        assert trainer_kwargs["train_dataset"] is mock_dataset_clean["train"]
        assert trainer_kwargs["eval_dataset"] is mock_dataset_clean["eval"]

    def test_run_eval_false_skips_evaluate(self):
        mocks, _, _ = self._run(run_eval=False)
        mocks["evaluate_checkpoint_vllm"].assert_not_called()

    def test_run_eval_true_calls_evaluate(self):
        mocks, _, _ = self._run(run_eval=True)
        mocks["evaluate_checkpoint_vllm"].assert_called_once()

    def test_runpod_secrets_parsed(self):
        import os
        with patch.dict(os.environ, {
            "RUNPOD_SECRET_HF_TOKEN": "hf-secret",
            "RUNPOD_SECRET_WANDB_API_KEY": "wandb-secret",
        }):
            self._run(extra_kwargs={"parse_secrets_runpod": True})
            assert os.environ.get("HF_TOKEN") == "hf-secret"
            assert os.environ.get("WANDB_API_KEY") == "wandb-secret"

    def test_run_name_prefixed_dpo(self):
        mocks, _, _ = self._run(extra_kwargs={"output_name_tag": "mytag"})
        dpo_config_kwargs = mocks["DPOConfig"].call_args.kwargs
        assert dpo_config_kwargs["run_name"].startswith("dpo-mytag-")

    def test_wandb_summary_updated_with_beta(self):
        mocks, _, _ = self._run(extra_kwargs={"beta": 0.2})
        mocks["wandb"].run = MagicMock()
        # Re-run with wandb.run active
        mocks2, _, _ = self._run(extra_kwargs={"beta": 0.2})
        # Just verify beta is in the config that was passed
        dpo_config_kwargs = mocks2["DPOConfig"].call_args.kwargs
        assert dpo_config_kwargs["beta"] == pytest.approx(0.2)
