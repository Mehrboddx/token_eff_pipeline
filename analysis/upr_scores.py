"""A third relevance signal for CPC-Mem: question likelihood (UPR, Sachan et al., EMNLP 2022).

CPC scores a sentence by embedding similarity to the question; Provence by a cross-encoder's
keep probability. UPR asks a language model how likely the question is *given* the sentence:
mean log p(question tokens | "a line from an earlier conversation: <role>: <sentence> ...").
It rewards sentences that make the question predictable, which is what implicit questions need
("recommend video-editing resources" given "I've been using Premiere Pro"), and it is a
different inductive bias from both existing scorers.

Re-scores only the top `--top` units of each question under the fused CPC + Provence score
(from analysis.score_dump), so it is a re-ranker, not a first-stage scorer. GPU, one model.

    python -m analysis.upr_scores --dump logs/analysis/scores_longmemeval_s.jsonl.gz \
        --model Qwen/Qwen2.5-1.5B-Instruct --top 300 --out logs/analysis/upr_qwen1.5b.jsonl.gz [--limit 5]
"""

import argparse
import gzip
import json
import time
from pathlib import Path

ROLE = {"user": "User", "model": "Assistant", "assistant": "Assistant"}
PROMPT = (
    "Here is a line from an earlier conversation between a user and an AI assistant.\n"
    "{role}: {text}\n\nA question the user could ask later that this line helps answer:\n"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from analysis.rerank_lab import fused

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)["question_id"] for line in gzip.open(args.out, "rt", encoding="utf-8")}
    out = gzip.open(args.out, "at", encoding="utf-8")
    start = time.time()
    count = 0
    for line in gzip.open(args.dump, "rt", encoding="utf-8"):
        row = json.loads(line)
        if row["question_id"] in done:
            continue
        if args.limit is not None and count >= args.limit:
            break
        scores = fused(row)
        top = [int(i) for i in np.argsort(-scores)[: args.top] if str(int(i)) in row["text"]]
        question_ids = tokenizer(row["question"], add_special_tokens=False)["input_ids"]
        result = {}
        for b in range(0, len(top), args.batch):
            ids = top[b : b + args.batch]
            seqs, starts = [], []
            for i in ids:
                role = ROLE.get(row["units"][i][2], row["units"][i][2])
                prefix = tokenizer(
                    PROMPT.format(role=role, text=row["text"][str(i)].strip()[:1500]), add_special_tokens=False
                )["input_ids"]
                seqs.append(prefix + question_ids)
                starts.append(len(prefix))
            width = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for k, s in enumerate(seqs):
                input_ids[k, : len(s)] = torch.tensor(s)
                mask[k, : len(s)] = 1
            input_ids, mask = input_ids.cuda(), mask.cuda()
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=mask).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, input_ids[:, 1:, None])[..., 0]
            for k, i in enumerate(ids):
                s, e = starts[k] - 1, len(seqs[k]) - 1  # positions predicting the question tokens
                result[str(i)] = round(float(logp[k, s:e].mean()), 5)
        out.write(json.dumps({"question_id": row["question_id"], "upr": result}) + "\n")
        out.flush()
        count += 1
        if count % 25 == 0:
            print(
                f"{count} questions, {time.time() - start:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                flush=True,
            )
    out.close()
    print(
        f"done: {count} questions in {time.time() - start:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
