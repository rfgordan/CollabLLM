from typing import List, Dict, Optional, Union
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class LocalAssistant:
    """Local LLM assistant for chat simulation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        lora_path: Optional[str] = None,
        use_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: str = "./model_cache",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        """
        Initialize local assistant model.

        Provide either model_path to load from disk, or model+tokenizer
        to wrap an already-loaded model (e.g., from a training run).

        Args:
            model_path: HuggingFace model path or local path (loads from disk)
            model: Pre-loaded HuggingFace model instance
            tokenizer: Pre-loaded tokenizer instance (required if model is provided)
            lora_path: Optional path to LoRA adapter weights
            use_4bit: Whether to use 4-bit quantization (QLoRA style)
            device_map: Device placement strategy
            torch_dtype: Model dtype (auto-detected if None)
            cache_dir: Directory for model cache
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (False = greedy)
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        if model is not None:
            if tokenizer is None:
                raise ValueError("tokenizer is required when passing a pre-loaded model")
            self.model = model
            self.tokenizer = tokenizer
            logger.info("Using pre-loaded model and tokenizer")
        elif model_path is not None:
            self.model, self.tokenizer = self._load_from_path(
                model_path, use_4bit, device_map, torch_dtype, cache_dir
            )
        else:
            raise ValueError("Either model_path or model+tokenizer must be provided")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if lora_path:
            self._load_lora(lora_path)

        if torch.cuda.is_available():
            self.model = torch.compile(self.model)
            logger.info("Model compiled with torch.compile()")

        logger.info(
            f"Model ready with memory footprint: "
            f"{self.model.get_memory_footprint() / (1024**3):.2f} GB"
        )

    @staticmethod
    def _load_from_path(
        model_path: str,
        use_4bit: bool,
        device_map: str,
        torch_dtype: Optional[torch.dtype],
        cache_dir: str,
    ):
        if torch_dtype is None:
            torch_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )

        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )

        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            use_fast=True,
        )

        return model, tokenizer

    @staticmethod
    def _sanitize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Ensure messages alternate user/assistant after any system message.
        Merges consecutive same-role messages with newlines.
        """
        result = []
        for msg in messages:
            if result and msg["role"] == result[-1]["role"]:
                result[-1] = {
                    "role": msg["role"],
                    "content": result[-1]["content"] + "\n" + msg["content"],
                }
            else:
                result.append(dict(msg))
        return result

    def _load_lora(self, lora_path: str) -> None:
        """Load LoRA adapter weights."""
        from peft import PeftModel

        logger.info(f"Loading LoRA adapter from {lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate assistant response given conversation history.

        Args:
            messages: Conversation history as list of {"role": str, "content": str}

        Returns:
            Generated assistant message content
        """
        sanitized = self._sanitize_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            sanitized, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else None,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        logger.debug(f"Assistant generated: {response[:100]}...")
        return response
