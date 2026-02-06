"""vLLM-based assistant with LoRA hot-swapping support."""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VLLMAssistant:
    """Assistant using vLLM for efficient multi-turn inference."""

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        enable_lora: bool = True,
        quantization: Optional[str] = "bitsandbytes",
        **vllm_kwargs,
    ):
        from vllm import LLM, SamplingParams

        self.sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens
        )
        self.llm = LLM(
            model=model_path,
            enable_lora=enable_lora,
            quantization=quantization,
            **vllm_kwargs,
        )
        self.lora_request = None

        if lora_path:
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("adapter", 1, lora_path)
            logger.info(f"LoRA adapter loaded from {lora_path}")

        logger.info(f"VLLMAssistant ready: {model_path}")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        outputs = self.llm.chat(
            messages, self.sampling_params, lora_request=self.lora_request
        )
        return outputs[0].outputs[0].text
