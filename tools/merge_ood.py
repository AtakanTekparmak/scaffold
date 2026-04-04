#!/usr/bin/env python3
"""Merge all OOD datasets into a single JSONL for stress-testing."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OOD_DIR = BASE / "examples" / "datasets" / "ood"
OUT = BASE / "examples" / "datasets" / "ood_merged.jsonl"

datasets = [
    "ag_news", "amazon_reviews", "banking77", "financial_phrasebank",
    "finer139", "go_emotions", "scicite", "scitail", "tweet_eval_hate",
]

all_rows = []
for name in datasets:
    path = OOD_DIR / "{}.jsonl".format(name)
    if not path.exists():
        print("  WARNING: {} not found, skipping".format(path))
        continue
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["id"] = "{}_{}".format(name, row["id"])
            all_rows.append(row)
            count += 1
    print("{}: {} cases".format(name, count))

with OUT.open("w", encoding="utf-8") as f:
    for row in all_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

train = sum(1 for r in all_rows if "_train-" in r["id"])
val = sum(1 for r in all_rows if "_val-" in r["id"])
test = sum(1 for r in all_rows if "_test-" in r["id"])
print("\nMerged: {} total ({} train / {} val / {} test)".format(
    len(all_rows), train, val, test))
print("Output: {}".format(OUT))
