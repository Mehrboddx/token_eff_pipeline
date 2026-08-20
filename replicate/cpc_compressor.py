"""
Context-aware Prompt Compression (CPC) — inference-time implementation.

Reimplements the inference pipeline described in Section 3.4 of
"Prompt Compression with Context-Aware Sentence Encoding for Fast and
Improved LLM Inference" (Liskavets et al., AAAI 2025), using a
transformers-native bidirectional-attention load (config.is_causal = False)
instead of the paper's custom LlamaBiForMNTPandSentEmbeddingsV2 class.

Pipeline:
  1. Split context into sentences (with character spans).
  2. Any single sentence longer than the per-chunk token budget is itself
     split into token-bounded sub-spans (see "oversized sentences" below)
     so nothing is silently truncated.
  3. Pack sentences into token-BALANCED chunks (see "chunking" below), and
     run one forward pass per chunk (bidirectional attention), average-
     pooling each sentence's token embeddings from its own chunk's pass ->
     this is what makes the embedding "context-aware": the same sentence
     embeds differently depending on what surrounds it within its chunk.
  4. Embed the question the same way (average-pool its own token embeddings).
  5. Score every sentence (across all chunks) by cosine similarity to the
     question embedding.
  6. Greedily keep highest-scoring sentences until the token budget is hit.
  7. Restore original sentence order (keeps output human-readable).

Chunking:
  The model's effective context window is `max_seq_length` tokens. A
  document longer than that cannot be embedded in a single forward pass.
  Truncating at `max_seq_length` would silently discard everything after
  it — for long documents (the whole point of long-context benchmarks
  like LongBench) that means real, potentially answer-bearing content
  never gets scored at all. So instead we chunk.

  This follows the same two-phase approach as the official Workday/cpc
  implementation (util/preprocessing.py: SamplePreprocessor.chunkify):
  first find the smallest number of chunks `num_chunks` such that packing
  sentences into `num_chunks` roughly-equal-token buckets keeps every
  bucket under budget, then verify. One difference from the official
  version: their rearrange step trusts the average-size split without
  re-checking that each rebalanced bucket is actually under budget, which
  can in principle overflow for unevenly-sized sentences. Here, every
  candidate bucketing is re-measured against the real token budget before
  accepting it, and `num_chunks` is incremented and retried if it doesn't
  fit — so a fit is always verified, not assumed.

Oversized sentences:
  A sentence longer than the whole per-chunk budget (e.g. one giant
  unbroken line of code) can't fit in any chunk by itself. The official
  implementation's `_ensure_sents_not_too_long` tries to split these into
  smaller pieces but has a slicing bug (`tokens[i*chunk_size:i*(chunk_size+1)]`
  instead of `tokens[i*chunk_size:(i+1)*chunk_size]`), which produces an
  empty first piece and increasingly overlapping/duplicated later pieces
  rather than a clean partition of the sentence. `_split_oversized_sentence`
  below instead uses the tokenizer's own offset mapping to cut the
  sentence into correctly non-overlapping, budget-sized pieces with
  accurate character spans, so nothing is dropped or duplicated.
"""

import math
from dataclasses import dataclass

import pysbd
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass
class ScoredSentence:
    text: str
    start: int          # char offset in original context
    end: int
    token_count: int
    score: float


class CPCCompressor:
    def __init__(
        self,
        base_model: str = "unsloth/Llama-3.2-1B-Instruct",
        lora_id: str = "deadcode99/cpc-1.0-llama-1b-ds-v5-iter66-lora-bidirectional-attn",
        tokenizer_id: str = "deadcode99/cpc-1.0-llama-1b-tokenizer",
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_seq_length: int = 6144,  # matches cpc-1.0-llama.json
        attn_implementation: str = "sdpa",  # avoids eager attention materializing
                                             # a full (seq_len x seq_len x heads) matrix
        chunk_safety_margin: int = 16,  # tokens reserved per chunk for special tokens
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.max_seq_length = max_seq_length
        self.chunk_safety_margin = chunk_safety_margin

        # Load the tokenizer FIRST: the CPC tokenizer has extra special tokens
        # (e.g. a mask token for MNTP training) beyond the base model's vocab,
        # so the base model's embedding matrix must be resized to match before
        # the LoRA adapter (which carries its own resized embed_tokens weight)
        # is attached. Loading the adapter onto an unresized base model raises
        # a shape mismatch, e.g.:
        #   size mismatch for base_model.model.embed_tokens.weight:
        #   checkpoint has [128258, 2048], model has [128256, 2048]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if not self.tokenizer.is_fast:
            raise ValueError(
                "This implementation requires a fast tokenizer "
                "(needed for return_offsets_mapping to map sentences -> tokens)."
            )

        config = AutoConfig.from_pretrained(base_model)
        config.is_causal = False  # bidirectional attention, see prior discussion

        # We never call .generate() / do autoregressive decoding here — every
        # forward pass is a single one-shot encode. The KV cache buys nothing
        # in that setting but still gets allocated by default, so turn it off.
        config.use_cache = False

        base = AutoModel.from_pretrained(
            base_model,
            config=config,
            dtype=dtype,
            # Pin the attention backend explicitly. Without this, the
            # bidirectional (is_causal=False) config can fall back to eager
            # attention, which materializes the full (seq_len x seq_len x
            # num_heads) score matrix — at max_seq_length=6144 that's several
            # GB by itself. SDPA (or "flash_attention_2" if installed) keeps
            # memory roughly linear in sequence length instead of quadratic.
            attn_implementation=attn_implementation,
        )

        # Resize base model embeddings to match the CPC tokenizer's vocab size
        # BEFORE attaching the LoRA adapter, so the adapter's embed_tokens
        # weight has a matching shape to load into.
        if len(self.tokenizer) != base.get_input_embeddings().weight.shape[0]:
            base.resize_token_embeddings(len(self.tokenizer))

        self.model = PeftModel.from_pretrained(base, lora_id)
        self.model.eval().to(self.device)

        self._segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)

    # ---------- sentence splitting ----------

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Return list of (sentence_text, char_start, char_end)."""
        spans = self._segmenter.segment(text)
        out = []
        for s in spans:
            stripped = s.sent.strip()
            if not stripped:
                continue
            # re-derive tight char bounds after stripping leading/trailing whitespace
            local_start = s.start + (len(s.sent) - len(s.sent.lstrip()))
            local_end = local_start + len(stripped)
            out.append((stripped, local_start, local_end))
        return out

    # ---------- oversized-sentence splitting ----------

    def _split_oversized_sentence(
        self, sent_text: str, start: int, end: int, budget: int
    ) -> list[tuple[str, int, int]]:
        """
        Split a single sentence whose token count exceeds `budget` into
        consecutive, non-overlapping, budget-sized pieces, using the
        tokenizer's own offset mapping so each piece keeps an accurate
        char span back into the original context.
        """
        enc = self.tokenizer(sent_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        pieces = []
        for i in range(0, len(offsets), budget):
            piece_offsets = offsets[i : i + budget]
            if not piece_offsets:
                continue
            local_start = piece_offsets[0][0]
            local_end = piece_offsets[-1][1]
            piece_text = sent_text[local_start:local_end]
            if not piece_text.strip():
                continue
            pieces.append((piece_text, start + local_start, start + local_end))
        return pieces

    def _expand_oversized(
        self, sentences: list[tuple[str, int, int]], budget: int
    ) -> list[tuple[str, int, int]]:
        """Replace any sentence longer than `budget` tokens with its split pieces."""
        expanded = []
        for sent_text, start, end in sentences:
            if self._count_tokens(sent_text) > budget:
                expanded.extend(self._split_oversized_sentence(sent_text, start, end, budget))
            else:
                expanded.append((sent_text, start, end))
        return expanded

    # ---------- chunking ----------

    def _pack_by_target_size(
        self, sentences: list[tuple[str, int, int]], token_counts: list[int], num_chunks: int
    ) -> list[list[tuple[str, int, int]]]:
        """
        Walk sentences in order, accumulating token count, and start a new
        bucket whenever the running total reaches the average target size
        (total_tokens / num_chunks). Mirrors the official chunkify's
        rebalance step.
        """
        total_tokens = sum(token_counts)
        target = total_tokens / num_chunks if num_chunks > 0 else total_tokens

        buckets: list[list[tuple[str, int, int]]] = []
        current: list[tuple[str, int, int]] = []
        current_len = 0
        for sent, tc in zip(sentences, token_counts):
            current.append(sent)
            current_len += tc + 1  # +1 approximates the join separator
            if current_len >= target:
                buckets.append(current)
                current = []
                current_len = 0
        if current:
            buckets.append(current)
        return buckets

    def _build_chunks(
        self, context: str, sentences: list[tuple[str, int, int]]
    ) -> list[tuple[str, int, list[tuple[str, int, int]]]]:
        """
        Pack sentences into token-balanced chunks that each fit under
        (max_seq_length - chunk_safety_margin) tokens.

        Two-phase approach (same idea as the official Workday/cpc
        `chunkify`): find the smallest `num_chunks` for which rebalancing
        sentences into that many token-averaged buckets keeps every bucket
        under budget — but unlike the official version, every candidate
        bucketing here is re-measured against the real token budget before
        being accepted, rather than assumed to fit.

        Returns a list of (chunk_text, chunk_char_start, chunk_sentences).
        """
        budget = self.max_seq_length - self.chunk_safety_margin
        # Any individual sentence longer than budget must already have been
        # split by _expand_oversized before this is called.
        token_counts = [self._count_tokens(s[0]) for s in sentences]

        num_chunks = 1
        buckets = self._pack_by_target_size(sentences, token_counts, num_chunks)
        while True:
            fits = True
            for b in buckets:
                chunk_start = b[0][1]
                chunk_end = b[-1][2]
                chunk_text = context[chunk_start:chunk_end]
                if self._count_tokens(chunk_text) > budget:
                    fits = False
                    break
            if fits:
                break
            num_chunks += 1
            buckets = self._pack_by_target_size(sentences, token_counts, num_chunks)
            if num_chunks > len(sentences):
                # Safety net: shouldn't happen since every individual
                # sentence already fits budget after _expand_oversized.
                break

        chunks = []
        for b in buckets:
            chunk_start = b[0][1]
            chunk_end = b[-1][2]
            chunks.append((context[chunk_start:chunk_end], chunk_start, b))
        return chunks

    # ---------- embedding ----------

    @torch.no_grad()
    def _embed_with_spans(self, text: str) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """
        Single forward pass over `text`. Returns:
          - token_embeddings: (seq_len, hidden_dim) on CPU
          - offset_mapping: list of (char_start, char_end) per token
        Truncates to max_seq_length tokens if needed (shouldn't trigger in
        normal operation since chunks and oversized-sentence pieces are
        already kept under budget).
        """
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        offset_mapping = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**enc)
        token_embeddings = outputs.last_hidden_state[0].to("cpu", dtype=torch.float32)

        # Free the GPU-resident activations for this pass immediately rather
        # than waiting for the next allocation to trigger a cache reuse.
        del outputs, enc
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return token_embeddings, offset_mapping

    @staticmethod
    def _span_average(
        token_embeddings: torch.Tensor,
        offset_mapping: list[tuple[int, int]],
        char_start: int,
        char_end: int,
    ) -> torch.Tensor | None:
        """Average-pool token embeddings whose char span overlaps [char_start, char_end)."""
        idxs = [
            i
            for i, (s, e) in enumerate(offset_mapping)
            if e > char_start and s < char_end and (s, e) != (0, 0)
        ]
        if not idxs:
            return None
        pooled = token_embeddings[idxs].mean(dim=0)
        return F.normalize(pooled, dim=0)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # ---------- public API ----------

    @torch.no_grad()
    def compress(
        self,
        context: str,
        question: str,
        compression_target_tokens: int,
    ) -> str:
        sentences = self._split_sentences(context)
        if not sentences:
            return context

        budget = self.max_seq_length - self.chunk_safety_margin
        sentences = self._expand_oversized(sentences, budget)

        chunks = self._build_chunks(context, sentences)
        if len(chunks) > 1:
            print(f"  context split into {len(chunks)} balanced chunks (no truncation)")

        # one forward pass over the question alone
        question_token_embeddings, question_offsets = self._embed_with_spans(question)
        question_embedding = self._span_average(
            question_token_embeddings, question_offsets, 0, len(question)
        )
        if question_embedding is None:
            raise ValueError("Could not embed question (empty after tokenization?).")

        scored: list[ScoredSentence] = []
        for chunk_text, chunk_start, chunk_sentences in chunks:
            # one forward pass per chunk -> context-aware token embeddings
            # (context is limited to this chunk, not the whole document)
            chunk_token_embeddings, chunk_offsets = self._embed_with_spans(chunk_text)

            for sent_text, start, end in chunk_sentences:
                local_start = start - chunk_start
                local_end = end - chunk_start
                sent_embedding = self._span_average(
                    chunk_token_embeddings, chunk_offsets, local_start, local_end
                )
                if sent_embedding is None:
                    continue
                score = torch.dot(sent_embedding, question_embedding).item()
                scored.append(
                    ScoredSentence(
                        text=sent_text,
                        start=start,
                        end=end,
                        token_count=self._count_tokens(sent_text),
                        score=score,
                    )
                )

        # greedy selection under the token budget, highest relevance first
        ranked = sorted(scored, key=lambda s: s.score, reverse=True)
        selected: list[ScoredSentence] = []
        running_tokens = 0
        for s in ranked:
            if running_tokens + s.token_count > compression_target_tokens:
                continue
            selected.append(s)
            running_tokens += s.token_count
            if running_tokens >= compression_target_tokens:
                break

        # restore original chronological order
        selected.sort(key=lambda s: s.start)
        return " ".join(s.text for s in selected)


if __name__ == "__main__":
    compressor = CPCCompressor()
    context = """
    Lorem ipsum is a dummy or placeholder text commonly used in graphic design,
    publishing, and web development to fill empty spaces in a layout that does
    not yet have content. Lorem ipsum is typically a corrupted version of De
    finibus bonorum et malorum, a 1st-century BC text by the Roman statesman
    and philosopher Cicero. Versions of the Lorem ipsum text have been used in
    typesetting at least since the 1960s.
    """
    question = "What is Lorem ipsum?"
    print(compressor.compress(context, question, compression_target_tokens=30))