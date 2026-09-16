import re
from typing import Any, Callable, Dict, Iterable, List, Optional


class TokenWise:
    PRETRAINED_PRESETS: Dict[str, Dict[str, str]] = {
        "mistral": {
            "config_path": "configs/cpc-1.0-mistral.json",
            "tokenizer_name_or_path": "deadcode99/cpc-1.0-mistral-7b-tokenizer",
            "lora_name_or_path": "deadcode99/cpc-1.0-mistral-7b-ds-v5-iter66-lora-bidirectional-attn",
        },
        "llama": {
            "config_path": "configs/cpc-1.0-llama.json",
            "tokenizer_name_or_path": "deadcode99/cpc-1.0-llama-1b-tokenizer",
            "lora_name_or_path": "deadcode99/cpc-1.0-llama-1b-ds-v5-iter66-lora-bidirectional-attn",
        },
    }

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        preset: str = "mistral",
        base_model_name_or_path: Optional[str] = None,
        load_pretrained: bool = False,
        use_openai_tokenizer: bool = False,
        recency_tiebreak: bool = True,
    ) -> None:
        self.preset = preset
        self.model = model
        self.tokenizer = tokenizer
        self.base_model_name_or_path = base_model_name_or_path
        self.sentences: List[str] = []
        self.model_spec = self._resolve_preset(preset)
        self.openai_tokenizer = self._load_openai_tokenizer() if use_openai_tokenizer else None
        # When two sentences tie on relevance (most often both score 0 —
        # true of most sentences in a large history), prefer the more
        # recent one. This only breaks ties; it never outweighs an actual
        # relevance difference. See compress() for why that distinction
        # matters at scale.
        self.recency_tiebreak = recency_tiebreak

        if load_pretrained and self.model is None and self.tokenizer is None:
            self.load_pretrained()

    @classmethod
    def available_presets(cls) -> List[str]:
        return sorted(cls.PRETRAINED_PRESETS)

    @classmethod
    def from_preset(
        cls,
        preset: str = "mistral",
        **kwargs: Any,
    ) -> "TokenWise":
        return cls(preset=preset, load_pretrained=True, **kwargs)

    def _resolve_preset(self, preset: str) -> Dict[str, str]:
        try:
            return self.PRETRAINED_PRESETS[preset].copy()
        except KeyError as exc:
            supported = ", ".join(self.available_presets())
            raise ValueError(f"Unsupported preset '{preset}'. Supported presets: {supported}") from exc

    def _load_openai_tokenizer(self) -> Any:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError("Install tiktoken to use OpenAI-style token counting.") from exc

        return tiktoken.encoding_for_model("gpt-4")

    def load_pretrained(
        self,
        model_loader: Optional[Callable[..., Any]] = None,
        tokenizer_loader: Optional[Callable[..., Any]] = None,
    ) -> "TokenWise":
        if self.model is not None and self.tokenizer is not None:
            return self

        if tokenizer_loader is None:
            tokenizer_loader = self._default_tokenizer_loader
        if model_loader is None:
            model_loader = self._default_model_loader

        self.tokenizer = tokenizer_loader(self.model_spec)
        self.model = model_loader(
            self.model_spec,
            self.tokenizer,
            base_model_name_or_path=self.base_model_name_or_path,
        )

        return self

    def _default_tokenizer_loader(self, model_spec: Dict[str, str]) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install transformers to load the CPC tokenizer presets."
            ) from exc

        return AutoTokenizer.from_pretrained(model_spec["tokenizer_name_or_path"])

    def _default_model_loader(
        self,
        model_spec: Dict[str, str],
        tokenizer: Any,
        base_model_name_or_path: Optional[str] = None,
    ) -> Any:
        try:
            import torch
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "Install torch, transformers, and peft to load the pretrained CPC model."
            ) from exc

        if not base_model_name_or_path:
            raise ValueError(
                "base_model_name_or_path is required to load a pretrained CPC model."
            )

        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_spec["lora_name_or_path"])
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        return model

    def context_sentences(self, context: str) -> List[str]:
        return self.set_sentences(re.split(r"(?<=[.!?])\s+", context.strip()))

    def set_sentences(self, sentences: Iterable[str]) -> List[str]:
        self.sentences = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]
        return self.sentences

    def add_sentences(self, sentences: Iterable[str]) -> List[str]:
        self.sentences.extend(sentence.strip() for sentence in sentences if sentence and sentence.strip())
        return self.sentences

    def _token_count(self, text: str) -> int:
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text, add_special_tokens=False))

        if self.openai_tokenizer is not None:
            return len(self.openai_tokenizer.encode(text))

        return len(text.split())

    def _truncate_to_budget(self, sentence: str, token_budget: int) -> str:
        """A single sentence can outweigh the whole budget on its own; trim it
        instead of either blowing past the budget or dropping it entirely."""
        if token_budget <= 0:
            return ""

        if self.tokenizer is not None and hasattr(self.tokenizer, "encode") and hasattr(self.tokenizer, "decode"):
            token_ids = self.tokenizer.encode(sentence, add_special_tokens=False)[:token_budget]
            return self.tokenizer.decode(token_ids).strip() + "…"

        if self.openai_tokenizer is not None:
            token_ids = self.openai_tokenizer.encode(sentence)[:token_budget]
            return self.openai_tokenizer.decode(token_ids).strip() + "…"

        return " ".join(sentence.split()[:token_budget]) + "…"

    def _query_tokens(self, query: str) -> set[str]:
        return {token.lower() for token in re.findall(r"\w+", query)}

    def _sentence_score(self, sentence: str, query_tokens: set[str]) -> float:
        sentence_tokens = {token.lower() for token in re.findall(r"\w+", sentence)}
        if not sentence_tokens:
            return 0.0

        overlap = len(sentence_tokens & query_tokens)
        return overlap / len(sentence_tokens)

    def compress(
        self,
        query: str,
        token_budget: int,
        context: Optional[str] = None,
        sentences: Optional[Iterable[str]] = None,
    ) -> str:
        if context is not None:
            self.context_sentences(context)
        elif sentences is not None:
            self.set_sentences(sentences)

        if not self.sentences:
            return ""

        if self.model is not None and hasattr(self.model, "compress"):
            return self.model.compress(query=query, token_budget=token_budget, context=" ".join(self.sentences))

        query_tokens = self._query_tokens(query)
        total = len(self.sentences)

        def sort_key(index: int):
            relevance = self._sentence_score(self.sentences[index], query_tokens)
            recency = (index / (total - 1)) if total > 1 else 1.0
            recency_tiebreak = recency if self.recency_tiebreak else 0.0
            return (relevance, recency_tiebreak, -self._token_count(self.sentences[index]))

        # Rank by relevance first; recency only breaks ties (it's a
        # secondary tuple element, not blended into relevance). Blending it
        # additively used to let recency dominate on large histories: with
        # thousands of candidate sentences, enough near-zero-relevance but
        # recent ones accumulate a recency bonus that outranks genuinely
        # relevant older sentences, burying exactly the fact being asked
        # about. As a pure tie-break, recency still resolves the anaphoric
        # case ("that"/"it" sentences usually share no words with the query,
        # so among equally-irrelevant candidates the most recent one wins),
        # without ever letting it override an actual relevance gap.
        #
        # Selection is greedy over this ranking, but each candidate's
        # original position is kept so the excerpt below can be reassembled
        # in its original order instead of rank order — scrambled order
        # breaks pronouns/references across sentences.
        ranked_indices = sorted(range(total), key=sort_key, reverse=True)

        overrides: Dict[int, str] = {}
        selected_indices: List[int] = []
        used_tokens = 0
        for index in ranked_indices:
            sentence = self.sentences[index]
            sentence_tokens = self._token_count(sentence)

            if not selected_indices and sentence_tokens > token_budget:
                sentence = self._truncate_to_budget(sentence, token_budget)
                overrides[index] = sentence
                sentence_tokens = self._token_count(sentence)

            if selected_indices and used_tokens + sentence_tokens > token_budget:
                continue

            selected_indices.append(index)
            used_tokens += sentence_tokens
            if used_tokens >= token_budget:
                break

        if not selected_indices:
            selected_indices = [0]

        selected_indices.sort()
        return " ".join(overrides.get(index, self.sentences[index]) for index in selected_indices)