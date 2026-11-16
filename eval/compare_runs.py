import json, sys
from math import comb
def load(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("event") == "result" and r.get("llm_judge_correct") is not None:
            out[r["question_id"]] = r
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
ca = sum(a[q]["llm_judge_correct"] for q in common); cb = sum(b[q]["llm_judge_correct"] for q in common)
better = sum(b[q]["llm_judge_correct"] and not a[q]["llm_judge_correct"] for q in common)
worse = sum(a[q]["llm_judge_correct"] and not b[q]["llm_judge_correct"] for q in common)
n = better + worse; k = min(better, worse)
p = min(1, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1
print(f"n={len(common)}  base {100*ca/len(common):.1f}  variant {100*cb/len(common):.1f}  better {better} / worse {worse}  McNemar p {p:.3f}")
by = {}
for q in common:
    t = a[q]["question_type"]; by.setdefault(t, [0, 0, 0]); by[t][0] += 1; by[t][1] += a[q]["llm_judge_correct"]; by[t][2] += b[q]["llm_judge_correct"]
for t, (n_, x, y) in sorted(by.items()):
    print(f"  {t:<28} n={n_:<4} {100*x/n_:5.1f} -> {100*y/n_:5.1f}")
