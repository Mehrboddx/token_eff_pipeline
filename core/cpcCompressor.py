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


def cpc_cache_key(preset: str, max_seq_length: int, attention: str = "bidirectional",
                  speaker_labels: bool = True, session_chunks: bool = True) -> str:
    ablated = ("" if speaker_labels else "-nolabels") + ("" if session_chunks else "-flatchunks")
    return f"cpc-{preset}-{max_seq_length}-{attention}-{LAYOUT_VERSION}{ablated}"


class CPCCompressor:
    """Scores history units with replicate/cpc_compressor.py's context-aware
    sentence embeddings, behind the `score_units(units, queries)` interface
    TokenWise uses for extractive scorers. Reuses the replicate/
    implementation directly so there's one copy of the encoding logic."""

    def __init__(self, preset: str = "llama", speaker_labels: bool = True, session_chunks: bool = True,
                 **kwargs: Any) -> None:
        """`speaker_labels` / `session_chunks` False are ablations: no role
        label before each turn, and chunks packed across sessions."""
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
        self.speaker_labels = speaker_labels
        self.session_chunks = session_chunks

    def unit_prefixes(self, units: Sequence[Unit]) -> List[str]:
        """A turn break + role label before the first sentence of each turn,
        a space otherwise -- conversational structure for the encoder."""
        prefixes = []
        previous_entry = None
        for unit in units:
            label = f"{role_label(unit)} " if self.speaker_labels else ""
            prefixes.append(f"\n{label}" if unit.entry_index != previous_entry else " ")
            previous_entry = unit.entry_index
        return prefixes

    def unit_groups(self, units: Sequence[Unit]) -> list:
        """Chunks never cross a change of group: one group per session, or a
        single group (the whole history) when session chunks are off."""
        return [unit.session if self.session_chunks else None for unit in units]

    def score_units(self, units: Sequence[Unit], queries: Sequence[str]) -> List[List[float]]:
        return self._compressor.score_texts(
            [unit.text for unit in units],
            self.unit_prefixes(units),
            self.unit_groups(units),
            list(queries),
        )

    def compress_document(self, context: str, question: str, token_budget: int, count_tokens=None) -> str:
        """CPC exactly as the paper runs it: one plain document, split and
        chunked as a whole, top sentences joined in their original order."""
        return self._compressor.compress(context, question, token_budget, count_tokens=count_tokens)

    def unit_embeddings(self, units: Sequence[Unit], ids: Sequence[int]) -> Dict[int, np.ndarray]:
        """Context-aware sentence embeddings for the given unit ids, for
        meaning-based MMR. Chunks just scored are served from the
        compressor's embedding cache, so this costs no extra encoding."""
        embeddings = self._compressor.embed_units(
            [unit.text for unit in units],
            self.unit_prefixes(units),
            self.unit_groups(units),
        )
        return {index: embeddings[index].numpy() for index in ids}
