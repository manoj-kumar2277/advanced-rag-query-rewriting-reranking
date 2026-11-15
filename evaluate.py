"""Compare all four modes on eval/questions.json.
Usage:  python evaluate.py            (retrieval metrics only, fast)
        python evaluate.py --answers  (also generate answers for manual 1-5 scoring)
"""
import json
import sys

import pandas as pd

import config
import rag

with_answers = "--answers" in sys.argv
questions = json.load(open("eval/questions.json", encoding="utf-8"))
rows = []

for q in questions:
    phrases = [p.lower() for p in q["expected_phrases"]]
    for mode in rag.MODES:
        res = rag.retrieve(q["question"], mode)
        rank = None
        for i, c in enumerate(res["chunks"], 1):
            if any(p in c["text"].lower() for p in phrases):
                rank = i
                break
        row = {"question": q["question"], "type": q.get("type", ""), "mode": mode,
               "search_query": res["search_query"], "hit": rank is not None,
               "rank": rank, "rr": 1 / rank if rank else 0.0,
               "retrieval_s": round(res["t_retrieval"], 3)}
        if with_answers:
            text, t = rag.answer(q["question"], res["chunks"])
            row.update(answer=text, answer_s=round(t, 2), manual_score="")
        rows.append(row)
        print(f"{mode:8s} hit={row['hit']!s:5s} rank={rank}  {q['question'][:60]}")

df = pd.DataFrame(rows)
df.to_csv("results/detailed_results.csv", index=False)

summary = (df.groupby("mode")
             .agg(hit_rate=("hit", "mean"), mrr=("rr", "mean"),
                  avg_retrieval_s=("retrieval_s", "mean"))
             .reindex(rag.MODES).round(3))
by_type = df.pivot_table(index="mode", columns="type", values="hit", aggfunc="mean").round(3)
summary.to_csv("results/summary.csv")
print(f"\nSummary (hit rate @{config.TOP_K}, MRR, latency):\n", summary)
print("\nHit rate by question type:\n", by_type)
