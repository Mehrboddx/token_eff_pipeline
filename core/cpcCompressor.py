from typing import Any, Dict, List, Sequence

import numpy as np

from core.tokenWise import TokenWise
from core.units import Unit, role_label

# TokenWise.PRETRAINED_PRESETS carries the tokenizer + LoRA adapter for each
# preset but not the base model id. Each entry here was read directly out of
# the corresponding LoRA adapter's adapter_config.json (base_model_name_or_path)
# on the Hub, not guessed.
BASE_MODELS = {
    "llama": "unsloth/Llama-3.2-1B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}

# Bump when the way units are laid out into chunks changes, so cached
# scores from an older layout are never reused.
LAYOUT_VERSION = "session-chunks-v1"


def cpc_cache_key(preset: str, max_seq_length: int, attention: str = "bidirectional") -> str:
    return f"cpc-{preset}-{max_seq_length}-{attention}-{LAYOUT_VERSION}"


class CPCCompressor:
    """Scores history units with replicate/cpc_compressor.py's context-aware
    sentence embeddings, behind the `score_units(units, queries)` interface
    TokenWise uses for extractive scorers. Reuses the replicate/
    implementation directly so there's one copy of the encoding logic."""

    def __init__(self, preset: str = "llama", **kwargs: Any) -> None:
        if preset not in TokenWise.PRETRAINED_PRESETS or preset not in BASE_MODELS:
            supported = ", ".join(sorted(set(TokenWise.PRETRAINED_PRESETS) & set(BASE_MODELS)))
            raise ValueError(f"Unsupported CPC preset '{preset}'. Supported presets: {supported}")

        preset_spec = TokenWise.PRETRAINED_PRESETS[preset]
        kwargs.setdefault("base_model", BASE_MODELS[preset])
        kwargs.setdefault("lora_id", preset_spec["lora_name_or_path"])
        kwargs.setdefault("tokenizer_id", preset_spec["tokenizer_name_or_path"])

        # Imported here, not at module level, so code that only needs this
        # module's constants (e.g. a CPU-only ablation over cached scores)
        # doesn't pull in torch. Loading the base model + LoRA is expensive;
        # Mistral-7B needs far more VRAM than an 8GB laptop GPU -- see
        # eval/deploy.sh for running it on a Vertex AI GPU instead.
        from replicate.cpc_compressor import CPCCompressor as _CPCCompressor

        self._compressor = _CPCCompressor(**kwargs)

    @staticmethod
    def unit_prefixes(units: Sequence[Unit]) -> List[str]:
        """A turn break + role label before the first sentence of each turn,
        a space otherwise -- conversational structure for the encoder."""
        prefixes = []
        previous_entry = None
        for unit in units:
            prefixes.append(f"\n{role_label(unit)} " if unit.entry_index != previous_entry else " ")
            previous_entry = unit.entry_index
        return prefixes

    def score_units(self, units: Sequence[Unit], queries: Sequence[str]) -> List[List[float]]:
        return self._compressor.score_texts(
            [unit.text for unit in units],
            self.unit_prefixes(units),
            [unit.session for unit in units],
            list(queries),
        )

    def unit_embeddings(self, units: Sequence[Unit], ids: Sequence[int]) -> Dict[int, np.ndarray]:
        """Context-aware sentence embeddings for the given unit ids, for
        meaning-based MMR. Chunks just scored are served from the
        compressor's embedding cache, so this costs no extra encoding."""
        embeddings = self._compressor.embed_units(
            [unit.text for unit in units],
            self.unit_prefixes(units),
            [unit.session for unit in units],
        )
        return {index: embeddings[index].numpy() for index in ids}
