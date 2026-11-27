"""Compress-stage log for CPC-Mem with UPR re-ranking (analysis.upr_scores), so the reader can
be tested on it with the usual --stage answer / --stage judge. No model is loaded: CPC and
Provence scores come from the score cache, UPR scores from the upr_scores output (keyed by
the same unit indices, since units are rebuilt by the same code path).

Selection is the pipeline's own: fused z-scores (CPC + Provence W) + gamma * z(UPR) on the
units UPR re-scored, then select_units under the budget and render_units, exactly as
TokenWise.compress_units does.

    python -m analysis.upr_render --upr logs/analysis/upr_qwen1.5b.jsonl.gz --budget 1000 --gamma 0.5 \
        --score-cache logs/score_cache --out logs/eval_longmemeval_run0061_compress_cpc-mistral_b1000_fuse0.25_upr.jsonl
"""

import argparse
import gzip
import json
from datetime import datetime

from core.console import utf8_console
from core.selection import SelectionConfig, fuse_scores, select_units, zscore
from core.token_wise import CompressionResult
from core.units import render_units
from eval.longmemeval import CompressionSettings, build_tokenwise, compress_question, load_items

utf8_console()


def refuse_to_load():
    raise RuntimeError("score cache miss: this script never loads a model")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upr", required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--score-cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    upr = {}
    for line in gzip.open(args.upr, "rt", encoding="utf-8"):
        row = json.loads(line)
        upr[row["question_id"]] = row["upr"]
    tokenwise = build_tokenwise(compressor="cpc", cpc_preset="mistral",
                                selection=SelectionConfig(provence_weight=args.weight), score_cache=args.score_cache)
    assert tokenwise is not None
    tokenwise.model._factory = refuse_to_load
    tokenwise.secondary._factory = refuse_to_load
    current: dict = {}

    def compress_units(query, token_budget, units, extra_queries=()):
        units = list(units)
        queries = [query, *dict.fromkeys(q for q in extra_queries if q and q != query)]
        primary = tokenwise.model.score_units(units, queries)
        secondary = tokenwise.secondary.score_units(units, queries)
        scores = fuse_scores(primary, tokenwise.selection, None, secondary)
        boost = current.get("upr") or {}
        if boost:
            ids = [int(i) for i in boost]
            for i, z in zip(ids, zscore([boost[str(i)] for i in ids])):
                scores[i] += args.gamma * z
        selected_ids = select_units(units, scores, token_budget, tokenwise.count_tokens, tokenwise.selection)
        selected = [units[i] for i in selected_ids]
        text = render_units(selected)
        return CompressionResult(text=text, selected=selected, tokens=tokenwise.count_tokens(text))

    tokenwise.compress_units = compress_units
    settings = CompressionSettings(token_budget=args.budget)
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps({"timestamp": datetime.now().isoformat(), "event": "run_start", "run": 61, "stage": "compress",
                              "dataset": "xiaowu0162/LongMemEval:longmemeval_s", "compressor": "cpc", "cpc_preset": "mistral",
                              "token_budget": args.budget, "selection": {"provence_weight": args.weight},
                              "upr": {"scores": args.upr, "gamma": args.gamma}, "configs": ["compressed_cpc"]}) + "\n")
        for index, item in enumerate(load_items("longmemeval_s"), 1):
            current["upr"] = upr.get(item["question_id"])
            described, mode, question, info = compress_question(item, tokenwise, settings)
            out.write(json.dumps({"timestamp": datetime.now().isoformat(), "event": "result", "config": "compressed_cpc",
                                  "question_id": item["question_id"], "question_type": item["question_type"],
                                  "question": question, "reference": item["answer"], "compression_mode": mode,
                                  "history": described, **info}) + "\n")
            if index % 50 == 0:
                print(f"{index} questions", flush=True)
        out.write(json.dumps({"timestamp": datetime.now().isoformat(), "event": "run_end"}) + "\n")
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
