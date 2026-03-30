#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["datasets"]
# ///
"""Prepare text classification datasets for Scaffold experiments.

Stratified sampling from HuggingFace datasets, outputs JSONL with label_guide.

Usage:
    uv run tools/prepare_classification.py \
        --dataset symptom2disease \
        --output examples/datasets/symptom2disease.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Dataset configurations: HuggingFace ID, text column, label column
DATASET_CONFIGS = {
    "symptom2disease": {
        "hf_id": "gretelai/symptom_to_diagnosis",
        "text_col": "text",
        "label_col": "label",
    },
    "uspto50k": {
        "hf_id": "pingzhili/uspto-50k",
        "text_col": "rxn_smiles",
        "label_col": "class",
    },
}


def parse_args():
    # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for the Scaffold dataset",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Output metadata JSON path (default: <output>.meta.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--train-per-label",
        type=int,
        default=None,
        help="Train examples per label (default: auto from Meta-Harness split sizes)",
    )
    parser.add_argument(
        "--val-per-label",
        type=int,
        default=None,
        help="Val examples per label (default: auto)",
    )
    parser.add_argument(
        "--test-per-label",
        type=int,
        default=None,
        help="Test examples per label (default: auto)",
    )
    return parser.parse_args()


def load_split(hf_id, split):
    # type: (str, str) -> Any
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "This script requires the 'datasets' package. "
            "Run it with: uv run tools/prepare_classification.py"
        ) from exc
    return load_dataset(hf_id, split=split)


def rows_by_label(dataset, text_col, label_col):
    # type: (Any, str, str) -> Tuple[Dict[str, List[str]], List[str]]
    """Group rows by label. Returns (grouped dict, sorted label names)."""
    grouped = defaultdict(list)  # type: Dict[str, List[str]]
    for row in dataset:
        label = str(row[label_col]).strip()
        text = str(row[text_col]).strip()
        grouped[label].append(text)
    labels = sorted(grouped.keys())
    return grouped, labels


def build_label_guide(labels):
    # type: (List[str]) -> str
    return "\n".join("- {}".format(label) for label in labels)


def sample_and_partition(grouped, labels, train_per, val_per, test_per, rng):
    # type: (Dict[str, List[str]], List[str], int, int, int, random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]
    """Stratified sampling: partition each label's examples into train/val/test."""
    needed = train_per + val_per + test_per
    train_rows = []  # type: List[Dict[str, Any]]
    val_rows = []  # type: List[Dict[str, Any]]
    test_rows = []  # type: List[Dict[str, Any]]

    for label in labels:
        candidates = list(grouped[label])
        rng.shuffle(candidates)
        if len(candidates) < needed:
            raise SystemExit(
                "Label '{}' has only {} examples, need {} ({}+{}+{})".format(
                    label, len(candidates), needed, train_per, val_per, test_per
                )
            )
        for text in candidates[:train_per]:
            train_rows.append({"text": text, "label": label})
        for text in candidates[train_per : train_per + val_per]:
            val_rows.append({"text": text, "label": label})
        for text in candidates[train_per + val_per : needed]:
            test_rows.append({"text": text, "label": label})

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def case_record(split, index, row, label_guide):
    # type: (str, int, Dict[str, Any], str) -> Dict[str, Any]
    label = row["label"]
    return {
        "id": "{}-{:03d}-{}".format(split, index, label.replace(" ", "_")),
        "input": {
            "task_text": row["text"],
            "label_guide": label_guide,
            "expected_label": label,
        },
        "expected": {
            "label": label,
        },
    }


def write_jsonl(path, rows):
    # type: (Path, List[Dict[str, Any]]) -> None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metadata(path, payload):
    # type: (Path, Dict[str, Any]) -> None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    # type: () -> None
    args = parse_args()
    rng = random.Random(args.seed)
    config = DATASET_CONFIGS[args.dataset]

    # Load full dataset — most classification datasets have train+test splits
    # For simplicity, load train split and do our own partitioning
    print("Loading dataset: {} ({})".format(args.dataset, config["hf_id"]), flush=True)
    dataset = load_split(config["hf_id"], "train")

    grouped, labels = rows_by_label(dataset, config["text_col"], config["label_col"])
    n_labels = len(labels)
    print("Found {} labels, {} total examples".format(n_labels, sum(len(v) for v in grouped.values())), flush=True)

    # Meta-Harness split sizes: 50 train / 30 val / 100 test (total 180)
    # Per-label: divide by number of classes
    train_per = args.train_per_label if args.train_per_label is not None else max(1, 50 // n_labels)
    val_per = args.val_per_label if args.val_per_label is not None else max(1, 30 // n_labels)
    test_per = args.test_per_label if args.test_per_label is not None else max(1, 100 // n_labels)

    print("Per-label counts: train={}, val={}, test={} (total={})".format(
        train_per, val_per, test_per,
        (train_per + val_per + test_per) * n_labels
    ), flush=True)

    train_rows, val_rows, test_rows = sample_and_partition(
        grouped, labels, train_per, val_per, test_per, rng
    )

    label_guide = build_label_guide(labels)
    dataset_rows = (
        [case_record("train", i, row, label_guide) for i, row in enumerate(train_rows, 1)]
        + [case_record("val", i, row, label_guide) for i, row in enumerate(val_rows, 1)]
        + [case_record("test", i, row, label_guide) for i, row in enumerate(test_rows, 1)]
    )

    output_path = Path(args.output)
    metadata_path = Path(args.metadata) if args.metadata else output_path.with_suffix(".meta.json")

    write_jsonl(output_path, dataset_rows)

    total = len(dataset_rows)
    metadata = {
        "source_dataset": config["hf_id"],
        "dataset_name": args.dataset,
        "seed": args.seed,
        "labels": labels,
        "label_guide": label_guide,
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "total": total,
        },
        "recommended_objective_split": {
            "train": len(train_rows) / total if total else 0.0,
            "val": len(val_rows) / total if total else 0.0,
            "test": len(test_rows) / total if total else 0.0,
        },
        "output": str(output_path),
    }
    write_metadata(metadata_path, metadata)

    print(json.dumps({
        "output": str(output_path),
        "metadata": str(metadata_path),
        "labels": labels,
        "counts": metadata["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
