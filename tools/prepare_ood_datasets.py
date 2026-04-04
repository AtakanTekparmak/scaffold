#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["datasets>=2.14,<3.0"]
# ///
"""Prepare 9 OOD text classification datasets for Meta-Harness comparison.

Downloads datasets from HuggingFace used as OOD evaluation tasks in
Meta-Harness (arxiv 2603.28052) and converts to Scaffold JSONL format.

Each dataset gets 180 cases (50 train / 30 val / 100 test) matching the
Meta-Harness evaluation protocol. Stratified sampling ensures label coverage.

Usage:
    uv run tools/prepare_ood_datasets.py --output-dir examples/datasets/ood
    uv run tools/prepare_ood_datasets.py --dataset scicite --output-dir examples/datasets/ood
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_hf(hf_id, split, config=None):
    # type: (str, str, Optional[str]) -> Any
    from datasets import load_dataset
    kwargs = {"trust_remote_code": True}
    if config:
        return load_dataset(hf_id, config, split=split, **kwargs)
    return load_dataset(hf_id, split=split, **kwargs)


# ── Dataset loaders ─────────────────────────────────────────────────
# Each returns (examples: list of (text, label_str), all_labels: sorted list)


def load_scicite():
    # type: () -> Tuple[List[Tuple[str, str]], List[str]]
    """Citation intent classification (3-way): background / method / result."""
    LABELS = {0: "background", 1: "method", 2: "result"}
    ds = load_hf("allenai/scicite", "test")
    examples = []
    for row in ds:
        text = str(row["string"]).strip()
        if not text:
            continue
        label = LABELS.get(row["label"])
        if label is None:
            continue
        examples.append((text, label))
    return examples, sorted(LABELS.values())


def load_finer139():
    # type: () -> Tuple[List[Tuple[str, str]], List[str]]
    """XBRL financial entity type classification (139-way).

    Converted from NER: each sentence classified by its dominant entity type.
    """
    ds = load_hf("nlpaueb/finer-139", "test")
    label_feature = ds.features["ner_tags"].feature
    label_names = label_feature.names if hasattr(label_feature, "names") else []

    base_types = set()  # type: set
    for name in label_names:
        if name == "O":
            continue
        base = name.split("-", 1)[1] if "-" in name else name
        base_types.add(base)

    examples = []
    for row in ds:
        tokens = row["tokens"]
        tags = row["ner_tags"]
        text = " ".join(str(t) for t in tokens).strip()
        if not text:
            continue
        entity_counts = defaultdict(int)  # type: Dict[str, int]
        for tag_id in tags:
            if tag_id < 0 or tag_id >= len(label_names):
                continue
            tag_name = label_names[tag_id]
            if tag_name == "O":
                continue
            base = tag_name.split("-", 1)[1] if "-" in tag_name else tag_name
            entity_counts[base] += 1
        if not entity_counts:
            continue
        label = max(entity_counts, key=lambda k: entity_counts[k])
        examples.append((text, label))

    return examples, sorted(base_types)


def load_amazon_reviews():
    # type: () -> Tuple[List[Tuple[str, str]], List[str]]
    """Amazon product review star rating (5-way)."""
    # mteb mirror (original amazon_reviews_multi is defunct)
    LABELS = {0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}
    ds = load_hf("mteb/amazon_reviews_multi", "test", config="en")
    examples = []
    for row in ds:
        text = str(row["text"]).strip()
        if not text:
            continue
        label = LABELS.get(row["label"])
        if label is None:
            continue
        examples.append((text, label))
    return examples, sorted(LABELS.values())


def load_financial_phrasebank():
    # type: () -> Tuple[List[Tuple[str, str]], List[str]]
    """Financial sentiment classification (3-way): negative / neutral / positive."""
    LABELS = {0: "negative", 1: "neutral", 2: "positive"}
    # Only has a "train" split
    ds = load_hf("financial_phrasebank", "train", config="sentences_allagree")
    examples = []
    for row in ds:
        text = str(row["sentence"]).strip()
        if not text:
            continue
        label = LABELS.get(row["label"])
        if label is None:
            continue
        examples.append((text, label))
    return examples, sorted(LABELS.values())


def load_go_emotions():
    # type: () -> Tuple[List[Tuple[str, str]], List[str]]
    """Reddit comment emotion classification (28-way).

    Multi-label dataset filtered to single-label examples only.
    """
    ds = load_hf("google-research-datasets/go_emotions", "test")
    label_feature = ds.features["labels"].feature
    label_names = label_feature.names if hasattr(label_feature, "names") else []

    examples = []
    seen_labels = set()  # type: set
    for row in ds:
        labels = row["labels"]
        if len(labels) != 1:
            continue
        text = str(row["text"]).strip()
        if not text:
            continue
        idx = labels[0]
        if idx < 0 or idx >= len(label_names):
            continue
        label = label_names[idx]
        examples.append((text, label))
        seen_labels.add(label)

    return examples, sorted(seen_labels)


def load_banking77():
    # type: () -> Tuple[List[Tuple[str, str]], List[str]]
    """Banking customer intent classification (77-way)."""
    ds = load_hf("PolyAI/banking77", "test")
    label_feature = ds.features["label"]
    label_names = label_feature.names if hasattr(label_feature, "names") else []

    examples = []
    for row in ds:
        text = str(row["text"]).strip()
        if not text:
            continue
        idx = row["label"]
        if idx < 0 or idx >= len(label_names):
            continue
        label = label_names[idx]
        examples.append((text, label))
    return examples, sorted(set(label_names))


def load_ag_news():
    # type: () -> Tuple[List[Tuple[str, str]], List[str]]
    """News topic classification (4-way): World / Sports / Business / Sci-Tech."""
    LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    ds = load_hf("ag_news", "test")
    examples = []
    for row in ds:
        text = str(row["text"]).strip()
        if not text:
            continue
        label = LABELS.get(row["label"])
        if label is None:
            continue
        examples.append((text, label))
    return examples, sorted(LABELS.values())


def load_scitail():
    # type: () -> Tuple[List[Tuple[str, str]], List[str]]
    """Science textual entailment (2-way): entails / neutral."""
    ds = load_hf("allenai/scitail", "test", config="tsv_format")
    examples = []
    seen_labels = set()  # type: set
    for row in ds:
        premise = str(row["premise"]).strip()
        hypothesis = str(row["hypothesis"]).strip()
        text = "Premise: {}\nHypothesis: {}".format(premise, hypothesis)
        label = str(row["label"]).strip().lower()
        examples.append((text, label))
        seen_labels.add(label)
    return examples, sorted(seen_labels)


def load_tweet_eval_hate():
    # type: () -> Tuple[List[Tuple[str, str]], List[str]]
    """Tweet hate speech detection (2-way): not hate / hate."""
    LABELS = {0: "not hate", 1: "hate"}
    ds = load_hf("tweet_eval", "test", config="hate")
    examples = []
    for row in ds:
        text = str(row["text"]).strip()
        if not text:
            continue
        label = LABELS.get(row["label"])
        if label is None:
            continue
        examples.append((text, label))
    return examples, sorted(LABELS.values())


# ── Registry ────────────────────────────────────────────────────────

DATASETS = {
    "scicite": {"loader": load_scicite, "desc": "Citation intent (3-way)"},
    "finer139": {"loader": load_finer139, "desc": "XBRL financial entity types (139-way)"},
    "amazon_reviews": {"loader": load_amazon_reviews, "desc": "Amazon star rating (5-way)"},
    "financial_phrasebank": {"loader": load_financial_phrasebank, "desc": "Financial sentiment (3-way)"},
    "go_emotions": {"loader": load_go_emotions, "desc": "Reddit emotion (28-way)"},
    "banking77": {"loader": load_banking77, "desc": "Banking intent (77-way)"},
    "ag_news": {"loader": load_ag_news, "desc": "News topic (4-way)"},
    "scitail": {"loader": load_scitail, "desc": "Science entailment (2-way)"},
    "tweet_eval_hate": {"loader": load_tweet_eval_hate, "desc": "Tweet hate speech (2-way)"},
}


# ── Sampling & output ──────────────────────────────────────────────


def sample_stratified(examples, total, rng):
    # type: (List[Tuple[str, str]], int, random.Random) -> List[Tuple[str, str]]
    """Sample `total` examples with best-effort stratification by label."""
    by_label = defaultdict(list)  # type: Dict[str, List[str]]
    for text, label in examples:
        by_label[label].append(text)

    labels = sorted(by_label.keys())
    n_labels = len(labels)
    if n_labels == 0:
        return []

    # Shuffle within each label
    for label in labels:
        rng.shuffle(by_label[label])

    # Allocate per label: divide total evenly, distribute remainder
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

    # Fill shortfall from overflow (labels with surplus examples)
    if len(sampled) < total and overflow:
        rng.shuffle(overflow)
        needed = total - len(sampled)
        sampled.extend(overflow[:needed])

    rng.shuffle(sampled)
    return sampled[:total]


def build_label_guide(labels):
    # type: (List[str]) -> str
    return "\n".join("- {}".format(label) for label in labels)


def case_record(split_name, index, text, label, label_guide):
    # type: (str, int, str, str, str) -> Dict[str, Any]
    return {
        "id": "{}-{:03d}-{}".format(split_name, index, label.replace(" ", "_")),
        "input": {
            "task_text": text,
            "label_guide": label_guide,
        },
        "expected": {
            "label": label,
        },
    }


def partition_and_write(name, examples, all_labels, output_dir, total, seed, rng):
    # type: (str, List[Tuple[str, str]], List[str], Path, int, int, random.Random) -> Dict[str, Any]
    """Sample, partition into train/val/test, write JSONL + metadata."""
    sampled = sample_stratified(examples, total, rng)

    # Meta-Harness split: 50 train / 30 val / 100 test
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

    # Write JSONL
    jsonl_path = output_dir / "{}.jsonl".format(name)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Label distribution in sample
    label_dist = defaultdict(int)  # type: Dict[str, int]
    for _, label in sampled:
        label_dist[label] += 1

    metadata = {
        "dataset_name": name,
        "description": DATASETS[name]["desc"],
        "seed": seed,
        "labels": all_labels,
        "label_count": len(all_labels),
        "label_guide": label_guide,
        "counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(rows),
        },
        "recommended_objective_split": {
            "train": round(len(train) / len(rows), 4) if rows else 0.0,
            "val": round(len(val) / len(rows), 4) if rows else 0.0,
            "test": round(len(test) / len(rows), 4) if rows else 0.0,
        },
        "label_distribution_in_sample": dict(sorted(label_dist.items())),
        "source_examples_available": len(examples),
        "output": str(jsonl_path),
    }

    meta_path = output_dir / "{}.meta.json".format(name)
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return metadata


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args():
    # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser(
        description="Prepare OOD datasets for Meta-Harness comparison"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=list(DATASETS.keys()),
        help="Specific dataset to prepare (default: all 9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=180,
        help="Total examples per dataset (default: 180, matching Meta-Harness)",
    )
    return parser.parse_args()


def main():
    # type: () -> None
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)

    names = [args.dataset] if args.dataset else list(DATASETS.keys())

    results = {}  # type: Dict[str, Any]
    for name in names:
        sep = "=" * 60
        print("\n{}".format(sep))
        print("Processing: {} - {}".format(name, DATASETS[name]["desc"]))
        print(sep)

        try:
            loader = DATASETS[name]["loader"]
            examples, all_labels = loader()
            print("  Loaded {} examples with {} unique labels".format(
                len(examples), len(all_labels)
            ))

            metadata = partition_and_write(
                name, examples, all_labels, output_dir, args.total, args.seed, rng
            )
            counts = metadata["counts"]
            results[name] = {
                "status": "ok",
                "counts": counts,
                "labels": len(all_labels),
            }
            print("  Wrote {} cases ({} train / {} val / {} test)".format(
                counts["total"], counts["train"], counts["val"], counts["test"]
            ))

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  ERROR: {}".format(e))
            results[name] = {"status": "error", "error": str(e)}

    # Summary
    print("\n\n" + "=" * 60)
    print("Summary")
    print("-" * 60)
    for name in names:
        result = results.get(name, {"status": "unknown"})
        if result["status"] == "ok":
            print("  {:<25s} {:>3d} labels  {:>3d} cases  OK".format(
                name, result["labels"], result["counts"]["total"]
            ))
        else:
            print("  {:<25s} FAILED: {}".format(name, result.get("error", "?")))
    print()


if __name__ == "__main__":
    main()
