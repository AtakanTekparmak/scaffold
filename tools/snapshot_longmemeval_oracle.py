#!/usr/bin/env python3
"""Create Scaffold-friendly LongMemEval oracle dataset snapshots.

This script downloads or reads the official LongMemEval oracle release, strips
oracle-only answer-span labels from the history, converts the sessions into the
typed memory benchmark schema used by Scaffold, and writes both a full dataset
and a smaller fixed subset for faster optimize/evaluate runs.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


OFFICIAL_ORACLE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_oracle.json"
)


PRESETS = {
    "full": {
        "output": "examples/datasets/longmemeval_oracle.jsonl",
        "metadata": "examples/datasets/longmemeval_oracle.meta.json",
        "total": None,
        "train": 300,
        "val": 100,
        "test": 100,
        "minimum_per_bucket": False,
    },
    "mini": {
        "output": "examples/datasets/longmemeval_oracle_mini.jsonl",
        "metadata": "examples/datasets/longmemeval_oracle_mini.meta.json",
        "total": 12,
        "train": 6,
        "val": 3,
        "test": 3,
        "minimum_per_bucket": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-file",
        default="examples/datasets/longmemeval_oracle.raw.json",
        help=(
            "Local official oracle JSON path. If it does not exist, the official release "
            "is downloaded to this path."
        ),
    )
    parser.add_argument(
        "--url",
        default=OFFICIAL_ORACLE_URL,
        help=f"Source URL used when --source-file is missing (default: {OFFICIAL_ORACLE_URL})",
    )
    parser.add_argument(
        "--preset",
        choices=["all", *sorted(PRESETS.keys())],
        default="all",
        help="Which snapshot preset(s) to build",
    )
    parser.add_argument("--seed", type=int, default=23, help="Deterministic sampling seed")
    return parser.parse_args()


def load_source(path: Path, url: str) -> list[dict[str, Any]]:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response, path.open("wb") as out:
            out.write(response.read())
    return json.loads(path.read_text(encoding="utf-8"))


def bucket_for(item: dict[str, Any]) -> str:
    if str(item["question_id"]).endswith("_abs"):
        return "abstention"
    return str(item["question_type"])


def normalize_turn(turn: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(turn["role"]),
        "content": str(turn["content"]),
    }


def normalize_case(item: dict[str, Any]) -> dict[str, Any]:
    dates = item["haystack_dates"]
    session_ids = item["haystack_session_ids"]
    sessions_raw = item["haystack_sessions"]
    if not (len(dates) == len(session_ids) == len(sessions_raw)):
        raise SystemExit(
            "LongMemEval oracle entry has mismatched haystack_dates / haystack_session_ids / "
            "haystack_sessions lengths"
        )
    sessions = list(zip(dates, session_ids, sessions_raw))
    # The oracle release does not guarantee chronological ordering. Sort by the
    # timestamp string so the memory pipeline receives coherent history order.
    sessions.sort(key=lambda entry: entry[0])

    return {
        "bucket": bucket_for(item),
        "question_id": str(item["question_id"]),
        "input": {
            "question_id": str(item["question_id"]),
            "question_type": str(item["question_type"]),
            "is_abstention": str(item["question_id"]).endswith("_abs"),
            "question_date": str(item["question_date"]),
            "question": str(item["question"]),
            "history": [
                {
                    "session_id": str(session_id),
                    "date": str(date),
                    "turns": [normalize_turn(turn) for turn in turns],
                }
                for date, session_id, turns in sessions
            ],
        },
        "expected": {
            "answer": str(item["answer"]),
            "evidence_session_ids": [str(value) for value in item["answer_session_ids"]],
            "abstain": str(item["question_id"]).endswith("_abs"),
        },
    }


def allocate_bucket_counts(
    counts_by_bucket: dict[str, int],
    total: int,
    minimum_per_bucket: bool,
) -> dict[str, int]:
    if total > sum(counts_by_bucket.values()):
        raise SystemExit(
            f"Requested {total} items but only {sum(counts_by_bucket.values())} are available"
        )

    buckets = sorted(counts_by_bucket)
    minima = {bucket: 0 for bucket in buckets}
    if minimum_per_bucket:
        if total < len(buckets):
            raise SystemExit(
                f"Need at least {len(buckets)} items to guarantee one example per bucket"
            )
        minima = {bucket: 1 for bucket in buckets}

    allocated = minima.copy()
    remaining = total - sum(allocated.values())
    remaining_capacity = {
        bucket: counts_by_bucket[bucket] - allocated[bucket] for bucket in buckets
    }
    total_capacity = sum(remaining_capacity.values())
    if remaining > total_capacity:
        raise SystemExit("Sampling request exceeds available capacity after minima")

    if remaining == 0:
        return allocated

    extras = {bucket: 0 for bucket in buckets}
    remainders: list[tuple[float, str]] = []
    for bucket in buckets:
        quota = remaining * (remaining_capacity[bucket] / total_capacity)
        whole = int(quota)
        extras[bucket] = min(whole, remaining_capacity[bucket])
        remainders.append((quota - whole, bucket))

    allocated = {bucket: allocated[bucket] + extras[bucket] for bucket in buckets}
    left = total - sum(allocated.values())
    for _, bucket in sorted(remainders, reverse=True):
        if left <= 0:
            break
        if allocated[bucket] < counts_by_bucket[bucket]:
            allocated[bucket] += 1
            left -= 1

    if left != 0:
        raise SystemExit("Failed to allocate the requested number of bucketed samples")

    return allocated


def interleave_round_robin(items_by_bucket: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    queues = {bucket: deque(items) for bucket, items in items_by_bucket.items() if items}
    ordered: list[dict[str, Any]] = []
    while queues:
        for bucket in list(sorted(queues)):
            queue = queues[bucket]
            if not queue:
                del queues[bucket]
                continue
            ordered.append(queue.popleft())
            if not queue:
                del queues[bucket]
    return ordered


def build_rows(
    items: list[dict[str, Any]],
    total: int | None,
    train_count: int,
    val_count: int,
    test_count: int,
    minimum_per_bucket: bool,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if train_count + val_count + test_count != (total or len(items)):
        raise SystemExit("split counts must add up to the requested total")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["bucket"]].append(item)
    for bucket_items in grouped.values():
        rng.shuffle(bucket_items)

    bucket_counts = {bucket: len(bucket_items) for bucket, bucket_items in grouped.items()}
    requested_total = total or len(items)
    sample_counts = allocate_bucket_counts(bucket_counts, requested_total, minimum_per_bucket)
    sampled_by_bucket = {
        bucket: grouped[bucket][: sample_counts[bucket]] for bucket in sorted(grouped)
    }
    ordered = interleave_round_robin(sampled_by_bucket)

    train = ordered[:train_count]
    val = ordered[train_count : train_count + val_count]
    test = ordered[train_count + val_count : train_count + val_count + test_count]

    def row(split: str, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": f"{split}-{item['question_id']}",
            "input": item["input"],
            "expected": item["expected"],
        }

    rows = [
        *[row("train", item) for item in train],
        *[row("val", item) for item in val],
        *[row("test", item) for item in test],
    ]

    split_bucket_counts = {
        "train": dict(sorted(Counter(item["bucket"] for item in train).items())),
        "val": dict(sorted(Counter(item["bucket"] for item in val).items())),
        "test": dict(sorted(Counter(item["bucket"] for item in test).items())),
    }

    metadata = {
        "counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "total": len(rows),
        },
        "sample_bucket_counts": dict(sorted(Counter(item["bucket"] for item in ordered).items())),
        "split_bucket_counts": split_bucket_counts,
        "train_question_ids": [item["question_id"] for item in train],
        "val_question_ids": [item["question_id"] for item in val],
        "test_question_ids": [item["question_id"] for item in test],
    }
    return rows, metadata


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_file)
    raw_items = load_source(source_path, args.url)
    normalized = [normalize_case(item) for item in raw_items]

    presets = PRESETS.items() if args.preset == "all" else [(args.preset, PRESETS[args.preset])]
    for index, (name, preset) in enumerate(presets):
        rng = random.Random(args.seed + index)
        rows, sample_meta = build_rows(
            normalized,
            total=preset["total"],
            train_count=preset["train"],
            val_count=preset["val"],
            test_count=preset["test"],
            minimum_per_bucket=preset["minimum_per_bucket"],
            rng=rng,
        )

        output_path = Path(preset["output"])
        metadata_path = Path(preset["metadata"])
        write_jsonl(output_path, rows)
        write_json(
            metadata_path,
            {
                "source": "official LongMemEval oracle release",
                "source_url": args.url,
                "source_file": str(source_path),
                "preset": name,
                "seed": args.seed + index,
                "output": str(output_path),
                "recommended_objective_split": {
                    "train": preset["train"] / len(rows),
                    "val": preset["val"] / len(rows),
                    "test": preset["test"] / len(rows),
                },
                **sample_meta,
            },
        )
        print(f"Wrote {len(rows)} cases to {output_path}")
        print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
