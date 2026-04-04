#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["datasets>=2.14,<3.0", "requests"]
# ///
"""Prepare the 3 Meta-Harness training datasets and merge them.

Downloads LawBench (task 3-3), USPTO-50k, and uses existing S2D,
then merges into a single JSONL for joint Scaffold optimization.

Usage:
    uv run tools/prepare_training_merge.py
"""

from __future__ import annotations

import json
import random
import re
import requests
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


BASE = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE / "examples" / "datasets"
SEED = 42

# ── Helpers ─────────────────────────────────────────────────────────


def build_label_guide(labels):
    # type: (List[str]) -> str
    return "\n".join("- {}".format(l) for l in labels)


def case_record(split_name, index, text, label, label_guide):
    # type: (str, int, str, str, str) -> Dict[str, Any]
    return {
        "id": "{}-{:03d}-{}".format(split_name, index, label.replace(" ", "_").replace(";", "_")),
        "input": {
            "task_text": text,
            "label_guide": label_guide,
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


def write_metadata(path, meta):
    # type: (Path, Dict[str, Any]) -> None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sample_stratified(examples, total, rng):
    # type: (List[Tuple[str, str]], int, random.Random) -> List[Tuple[str, str]]
    by_label = defaultdict(list)  # type: Dict[str, List[str]]
    for text, label in examples:
        by_label[label].append(text)
    labels = sorted(by_label.keys())
    n_labels = len(labels)
    if n_labels == 0:
        return []
    for label in labels:
        rng.shuffle(by_label[label])
    per_label = total // n_labels
    extra = total % n_labels
    sampled = []  # type: List[Tuple[str, str]]
    overflow = []  # type: List[Tuple[str, str]]
    for i, label in enumerate(labels):
        target = per_label + (1 if i < extra else 0)
        available = by_label[label]
        take = min(target, len(available))
        for text in available[:take]:
            sampled.append((text, label))
        for text in available[take:]:
            overflow.append((text, label))
    if len(sampled) < total and overflow:
        rng.shuffle(overflow)
        needed = total - len(sampled)
        sampled.extend(overflow[:needed])
    rng.shuffle(sampled)
    return sampled[:total]


def partition_and_write(name, examples, all_labels, output_dir, total, rng):
    # type: (str, List[Tuple[str, str]], List[str], Path, int, random.Random) -> Dict[str, Any]
    sampled = sample_stratified(examples, total, rng)
    n_train = min(50, len(sampled))
    n_val = min(30, max(0, len(sampled) - n_train))
    n_test = max(0, len(sampled) - n_train - n_val)
    train = sampled[:n_train]
    val = sampled[n_train:n_train + n_val]
    test = sampled[n_train + n_val:]
    label_guide = build_label_guide(all_labels)
    rows = []  # type: List[Dict[str, Any]]
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        for i, (text, label) in enumerate(split_data, 1):
            rows.append(case_record(split_name, i, text, label, label_guide))
    jsonl_path = output_dir / "{}.jsonl".format(name)
    write_jsonl(jsonl_path, rows)
    meta = {
        "dataset_name": name,
        "seed": SEED,
        "labels": all_labels,
        "label_count": len(all_labels),
        "counts": {"train": len(train), "val": len(val), "test": len(test), "total": len(rows)},
        "recommended_objective_split": {
            "train": round(len(train) / len(rows), 4) if rows else 0,
            "val": round(len(val) / len(rows), 4) if rows else 0,
            "test": round(len(test) / len(rows), 4) if rows else 0,
        },
    }
    meta_path = output_dir / "{}.meta.json".format(name)
    write_metadata(meta_path, meta)
    return meta


# ── LawBench ────────────────────────────────────────────────────────


def prepare_lawbench(output_dir, rng):
    # type: (Path, random.Random) -> Dict[str, Any]
    print("=== LawBench (task 3-3: charge prediction) ===")
    url = "https://raw.githubusercontent.com/open-compass/LawBench/main/data/zero_shot/3-3.json"
    print("  Downloading from GitHub...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print("  Downloaded {} cases".format(len(data)))

    # Parse: answer format is "罪名:charge1" or "罪名:charge1;charge2"
    examples = []  # type: List[Tuple[str, str]]
    all_labels_set = set()  # type: set
    multi_label_count = 0
    for item in data:
        text = item["question"].strip()
        answer = item["answer"].strip()
        # Extract charge(s) after "罪名:"
        match = re.match(r"罪名[:：](.+)", answer)
        if not match:
            continue
        charges_str = match.group(1).strip()
        charges = [c.strip() for c in charges_str.split(";") if c.strip()]
        all_labels_set.update(charges)
        if len(charges) > 1:
            multi_label_count += 1
            # For multi-label cases, use the first charge as primary label
            # (consistent with classification framing)
        label = charges[0]
        examples.append((text, label))

    all_labels = sorted(all_labels_set)
    print("  {} examples, {} unique charges, {} multi-label cases".format(
        len(examples), len(all_labels), multi_label_count
    ))

    meta = partition_and_write("lawbench", examples, all_labels, output_dir, 180, rng)
    print("  Wrote {} cases ({} train / {} val / {} test)".format(
        meta["counts"]["total"], meta["counts"]["train"], meta["counts"]["val"], meta["counts"]["test"]
    ))
    return meta


# ── USPTO-50k ──────────────────────────────────────────────────────


# Reaction superclass names (Schneider et al. 2016)
USPTO_CLASS_NAMES = {
    1: "Heteroatom alkylation/arylation",
    2: "Acylation and related",
    3: "C-C bond formation",
    4: "Heterocycle formation",
    5: "Protections",
    6: "Deprotections",
    7: "Reductions",
    8: "Oxidations",
    9: "Functional group interconversion",
    10: "Functional group addition",
}


def prepare_uspto(output_dir, rng):
    # type: (Path, random.Random) -> Dict[str, Any]
    print("\n=== USPTO-50k (reaction classification) ===")
    from datasets import load_dataset
    ds = load_dataset("pingzhili/uspto-50k", split="train")
    print("  Loaded {} reactions".format(len(ds)))

    examples = []  # type: List[Tuple[str, str]]
    for row in ds:
        # Use full reaction SMILES (reactants>>product) so the model can see the transformation
        text = str(row["rxn_smiles"]).strip()
        if not text:
            continue
        class_id = int(row["class"])
        label = USPTO_CLASS_NAMES.get(class_id, "Class {}".format(class_id))
        examples.append((text, label))

    all_labels = sorted(USPTO_CLASS_NAMES.values())
    print("  {} examples, {} classes".format(len(examples), len(all_labels)))

    meta = partition_and_write("uspto50k", examples, all_labels, output_dir, 180, rng)
    print("  Wrote {} cases ({} train / {} val / {} test)".format(
        meta["counts"]["total"], meta["counts"]["train"], meta["counts"]["val"], meta["counts"]["test"]
    ))
    return meta


# ── Merge ───────────────────────────────────────────────────────────


def merge_datasets(output_dir):
    # type: (Path) -> None
    """Merge S2D + LawBench + USPTO-50k into one JSONL for joint training."""
    print("\n=== Merging training datasets ===")
    s2d_path = DATASETS_DIR / "symptom2disease.jsonl"
    lawbench_path = output_dir / "lawbench.jsonl"
    uspto_path = output_dir / "uspto50k.jsonl"

    all_rows = []  # type: List[Dict[str, Any]]
    sources = [
        ("s2d", s2d_path),
        ("lawbench", lawbench_path),
        ("uspto50k", uspto_path),
    ]

    for tag, path in sources:
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
                # Prefix the case ID with the dataset tag for disambiguation
                row["id"] = "{}_{}".format(tag, row["id"])
                all_rows.append(row)
                count += 1
        print("  {}: {} cases".format(tag, count))

    merged_path = output_dir / "merged_training.jsonl"
    write_jsonl(merged_path, all_rows)

    # Count splits
    train_count = sum(1 for r in all_rows if r["id"].split("_", 1)[1].startswith("train-"))
    val_count = sum(1 for r in all_rows if r["id"].split("_", 1)[1].startswith("val-"))
    test_count = sum(1 for r in all_rows if r["id"].split("_", 1)[1].startswith("test-"))

    total = len(all_rows)
    meta = {
        "description": "Merged S2D + LawBench + USPTO-50k for joint optimization",
        "sources": ["symptom2disease", "lawbench", "uspto50k"],
        "counts": {"train": train_count, "val": val_count, "test": test_count, "total": total},
        "recommended_objective_split": {
            "train": round(train_count / total, 4) if total else 0,
            "val": round(val_count / total, 4) if total else 0,
            "test": round(test_count / total, 4) if total else 0,
        },
    }
    write_metadata(output_dir / "merged_training.meta.json", meta)

    print("  Merged: {} total ({} train / {} val / {} test)".format(
        total, train_count, val_count, test_count
    ))
    print("  Output: {}".format(merged_path))


# ── Main ────────────────────────────────────────────────────────────


def main():
    # type: () -> None
    rng = random.Random(SEED)
    output_dir = DATASETS_DIR / "training"

    prepare_lawbench(output_dir, rng)
    prepare_uspto(output_dir, rng)
    merge_datasets(output_dir)

    print("\nDone! Files in {}".format(output_dir))


if __name__ == "__main__":
    main()
