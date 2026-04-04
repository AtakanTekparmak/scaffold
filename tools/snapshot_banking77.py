#!/usr/bin/env python3
"""Create a small local Banking77 snapshot for Scaffold experiments.

Example:
    uv run --python 3.11 --with datasets python tools/snapshot_banking77.py \
      --output examples/datasets/banking77_subset.jsonl \
      --metadata examples/datasets/banking77_subset.meta.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="gtfintechlab/banking77",
        help="Hugging Face dataset id to load (default: gtfintechlab/banking77)",
    )
    parser.add_argument(
        "--output",
        default="examples/datasets/banking77_subset.jsonl",
        help="Output JSONL path for the Scaffold dataset snapshot",
    )
    parser.add_argument(
        "--metadata",
        default="examples/datasets/banking77_subset.meta.json",
        help="Output metadata JSON path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for label selection and per-label sampling",
    )
    parser.add_argument(
        "--label-count",
        type=int,
        default=8,
        help="How many Banking77 labels to include when --labels is not given",
    )
    parser.add_argument(
        "--train-per-label",
        type=int,
        default=2,
        help="Examples per selected label taken from the original train split for optimization train",
    )
    parser.add_argument(
        "--val-per-label",
        type=int,
        default=1,
        help="Examples per selected label taken from the original train split for validation",
    )
    parser.add_argument(
        "--test-per-label",
        type=int,
        default=1,
        help="Examples per selected label taken from the original test split for holdout test",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated explicit label list to use instead of random selection",
    )
    return parser.parse_args()


def load_split(dataset_name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in user environment
        raise SystemExit(
            "This script requires the 'datasets' package. "
            "Run it with: uv run --python 3.11 --with datasets python tools/snapshot_banking77.py"
        ) from exc

    return load_dataset(dataset_name, split=split)


def normalize_label(label: str) -> str:
    return label.strip()


def humanize_label(label: str) -> str:
    return label.replace("_", " ")


def rows_by_label(split_dataset, label_names: list[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_dataset:
        label = row["label"]
        if isinstance(label, int):
            label = label_names[label]
        label = normalize_label(label)
        grouped[label].append({"text": row["text"], "label": label})
    return grouped


def choose_labels(
    available_labels: list[str],
    explicit_labels: list[str],
    label_count: int,
    rng: random.Random,
) -> list[str]:
    if explicit_labels:
        missing = [label for label in explicit_labels if label not in available_labels]
        if missing:
            raise SystemExit(f"Unknown requested labels: {missing}")
        return explicit_labels

    if label_count > len(available_labels):
        raise SystemExit(
            f"Requested {label_count} labels but only {len(available_labels)} labels are available"
        )
    return sorted(rng.sample(sorted(available_labels), label_count))


def sample_rows(
    rows: dict[str, list[dict[str, Any]]],
    labels: list[str],
    per_label: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if per_label <= 0:
        return []

    sampled: list[dict[str, Any]] = []
    for label in labels:
        candidates = list(rows[label])
        rng.shuffle(candidates)
        if len(candidates) < per_label:
            raise SystemExit(
                f"Label '{label}' only has {len(candidates)} rows, need {per_label}"
            )
        sampled.extend(candidates[:per_label])
    rng.shuffle(sampled)
    return sampled


def partition_train_rows(
    train_rows: dict[str, list[dict[str, Any]]],
    labels: list[str],
    train_per_label: int,
    val_per_label: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_cases: list[dict[str, Any]] = []
    val_cases: list[dict[str, Any]] = []
    needed = train_per_label + val_per_label

    for label in labels:
        candidates = list(train_rows[label])
        rng.shuffle(candidates)
        if len(candidates) < needed:
            raise SystemExit(
                f"Label '{label}' only has {len(candidates)} train rows, need {needed}"
            )
        train_cases.extend(candidates[:train_per_label])
        val_cases.extend(candidates[train_per_label:needed])

    rng.shuffle(train_cases)
    rng.shuffle(val_cases)
    return train_cases, val_cases


def build_label_guide(labels: list[str]) -> str:
    return "\n".join(f"- {label}: {humanize_label(label)}" for label in labels)


def case_record(
    split: str,
    index: int,
    row: dict[str, Any],
    labels: list[str],
    label_guide: str,
) -> dict[str, Any]:
    label = normalize_label(row["label"])
    return {
        "id": f"{split}-{index:03d}-{label}",
        "input": {
            "text": row["text"],
            "label_guide": label_guide,
            "labels": labels,
        },
        "expected": {
            "label": label,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_split = load_split(args.dataset, "train")
    test_split = load_split(args.dataset, "test")
    label_feature = train_split.features["label"]
    label_names = list(getattr(label_feature, "names", []))

    train_rows = rows_by_label(train_split, label_names)
    test_rows = rows_by_label(test_split, label_names)

    explicit_labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    available_labels = sorted(
        label
        for label in train_rows
        if len(train_rows[label]) >= args.train_per_label + args.val_per_label
        and len(test_rows.get(label, [])) >= args.test_per_label
    )
    selected_labels = choose_labels(
        available_labels, explicit_labels, args.label_count, rng
    )

    train_rows_selected, val_rows_selected = partition_train_rows(
        train_rows,
        selected_labels,
        args.train_per_label,
        args.val_per_label,
        rng,
    )
    test_rows_selected = sample_rows(
        test_rows,
        selected_labels,
        args.test_per_label,
        rng,
    )

    label_guide = build_label_guide(selected_labels)
    dataset_rows = [
        *[
            case_record("train", index, row, selected_labels, label_guide)
            for index, row in enumerate(train_rows_selected, start=1)
        ],
        *[
            case_record("val", index, row, selected_labels, label_guide)
            for index, row in enumerate(val_rows_selected, start=1)
        ],
        *[
            case_record("test", index, row, selected_labels, label_guide)
            for index, row in enumerate(test_rows_selected, start=1)
        ],
    ]

    output_path = Path(args.output)
    metadata_path = Path(args.metadata)
    write_jsonl(output_path, dataset_rows)

    total = len(dataset_rows)
    metadata = {
        "source_dataset": args.dataset,
        "seed": args.seed,
        "selected_labels": selected_labels,
        "label_guide": label_guide,
        "counts": {
            "train": len(train_rows_selected),
            "val": len(val_rows_selected),
            "test": len(test_rows_selected),
            "total": total,
        },
        "recommended_objective_split": {
            "train": len(train_rows_selected) / total if total else 0.0,
            "val": len(val_rows_selected) / total if total else 0.0,
            "test": len(test_rows_selected) / total if total else 0.0,
        },
        "output": str(output_path),
    }
    write_metadata(metadata_path, metadata)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "metadata": str(metadata_path),
                "selected_labels": selected_labels,
                "counts": metadata["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
