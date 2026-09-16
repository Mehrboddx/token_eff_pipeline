from typing import Any

from core.tokenWise import TokenWise
from replicate.cpc_compressor import CPCCompressor as _CPCCompressor

# TokenWise.PRETRAINED_PRESETS carries the tokenizer + LoRA adapter for each
# preset but not the base model id, since TokenWise's own pretrained loader
# takes base_model_name_or_path as a separate argument (the base model isn't
# part of what the CPC LoRA/tokenizer publish "own" -- it's whatever they
# were fine-tuned on top of). Each entry here was read directly out of the
# corresponding LoRA adapter's adapter_config.json (base_model_name_or_path)
# on the Hub, not guessed.
BASE_MODELS = {
    "llama": "unsloth/Llama-3.2-1B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}


class CPCCompressor:
    """Adapts replicate/cpc_compressor.py's CPCCompressor (context-aware,
    embedding-based sentence scoring via a bidirectional-attention forward
    pass) to the `model.compress(query, token_budget, context)` interface
    TokenWise delegates to -- the same interface GeminiCompressor implements,
    so this can be dropped in as `TokenWise(model=CPCCompressor())`.

    Reuses the replicate/ implementation directly instead of reimplementing
    the chunking/embedding pipeline a second time, so there's one copy of
    that logic instead of two that can drift apart."""

    def __init__(self, preset: str = "llama", **kwargs: Any) -> None:
        if preset not in TokenWise.PRETRAINED_PRESETS or preset not in BASE_MODELS:
            supported = ", ".join(sorted(set(TokenWise.PRETRAINED_PRESETS) & set(BASE_MODELS)))
            raise ValueError(f"Unsupported CPC preset '{preset}'. Supported presets: {supported}")

        preset_spec = TokenWise.PRETRAINED_PRESETS[preset]
        kwargs.setdefault("base_model", BASE_MODELS[preset])
        kwargs.setdefault("lora_id", preset_spec["lora_name_or_path"])
        kwargs.setdefault("tokenizer_id", preset_spec["tokenizer_name_or_path"])

        # Loads the base model + LoRA adapter onto GPU/CPU -- expensive, so
        # only pay for it when the CPC backend is actually selected (this
        # class shouldn't be instantiated otherwise). Mistral-7B in
        # particular needs far more VRAM than the 1B Llama preset -- not
        # realistic on an 8GB laptop GPU, see eval/deploy.sh for running it
        # on a Vertex AI GPU instead.
        self._compressor = _CPCCompressor(**kwargs)

    def compress(self, query: str, token_budget: int, context: str) -> str:
        if not context or not context.strip():
            return ""

        return self._compressor.compress(
            context=context,
            question=query,
            compression_target_tokens=token_budget,
        )
