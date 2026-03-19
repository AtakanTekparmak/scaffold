#!/usr/bin/env python3
"""Create a local ARC-AGI-2 subset for Scaffold benchmark runs.

By default this script downloads the official ARC-AGI-2 repository archive from
GitHub, samples a small subset of public training and evaluation tasks, and
writes a Scaffold-friendly JSONL dataset where each case represents one full ARC
task.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


OFFICIAL_ZIP_URL = "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip"


PRESETS = {
    "benchmark": {
        "output": "examples/datasets/arc_agi2_subset.jsonl",
        "metadata": "examples/datasets/arc_agi2_subset.meta.json",
        "train_count": 8,
        "val_count": 4,
        "test_count": 4,
        "max_grid_cells": 225,
        "max_total_pairs": 6,
    },
    "mini": {
        "output": "examples/datasets/arc_agi2_mini.jsonl",
        "metadata": "examples/datasets/arc_agi2_mini.meta.json",
        "train_count": 4,
        "val_count": 2,
        "test_count": 2,
        "max_grid_cells": 144,
        "max_total_pairs": 5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="benchmark",
        help="Sampling preset to use for counts, filters, and default output paths",
    )
    parser.add_argument(
        "--source-dir",
        default="",
        help="Optional local ARC-AGI-2 checkout root. If omitted, the official GitHub archive is downloaded.",
    )
    parser.add_argument(
        "--zip-url",
        default=OFFICIAL_ZIP_URL,
        help=f"Archive URL to download when --source-dir is omitted (default: {OFFICIAL_ZIP_URL})",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL path for the Scaffold dataset snapshot",
    )
    parser.add_argument(
        "--metadata",
        default="",
        help="Output metadata JSON path",
    )
    parser.add_argument("--seed", type=int, default=11, help="Sampling seed")
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Number of public training tasks used for optimization train",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Number of public training tasks reserved for validation",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Number of public evaluation tasks reserved for holdout test",
    )
    parser.add_argument(
        "--max-grid-cells",
        type=int,
        default=None,
        help="Skip tasks containing grids larger than this many cells to keep prompts tractable",
    )
    parser.add_argument(
        "--max-total-pairs",
        type=int,
        default=None,
        help="Skip tasks with more than this many combined train+test pairs",
    )
    return parser.parse_args()


def fetch_official_repo(zip_url: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="arc_agi2_"))
    archive_path = temp_dir / "arc_agi2.zip"
    with urllib.request.urlopen(zip_url) as response, archive_path.open("wb") as out:
        shutil.copyfileobj(response, out)

    extract_dir = temp_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)

    roots = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise SystemExit(
            f"Expected one root directory in extracted archive, found {len(roots)}"
        )
    return roots[0]


def grid_cells(grid: list[list[int]]) -> int:
    return len(grid) * (len(grid[0]) if grid else 0)


def task_is_tractable(task: dict[str, Any], max_grid_cells: int, max_total_pairs: int) -> bool:
    total_pairs = len(task.get("train", [])) + len(task.get("test", []))
    if total_pairs > max_total_pairs:
        return False

    for pair in task.get("train", []) + task.get("test", []):
        for side in ["input", "output"]:
            grid = pair.get(side)
            if grid is None:
                continue
            if grid_cells(grid) > max_grid_cells:
                return False
    return True


def load_task_files(directory: Path, max_grid_cells: int, max_total_pairs: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        if not task_is_tractable(task, max_grid_cells, max_total_pairs):
            continue
        tasks.append(
            {
                "task_id": path.stem,
                "train_examples": task["train"],
                "test_pairs": task["test"],
            }
        )
    return tasks


def sample_tasks(tasks: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count > len(tasks):
        raise SystemExit(f"Requested {count} tasks but only {len(tasks)} are available")
    chosen = list(tasks)
    rng.shuffle(chosen)
    return chosen[:count]


def dataset_case(source_split: str, task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{source_split}-{task['task_id']}",
        "input": {
            "task_id": task["task_id"],
            "train_examples": task["train_examples"],
            "test_inputs": [pair["input"] for pair in task["test_pairs"]],
        },
        "expected": {
            "outputs": [pair["output"] for pair in task["test_pairs"]],
        },
    }


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
    rng = random.Random(args.seed)
    preset = PRESETS[args.preset]

    output_path = Path(args.output or preset["output"])
    metadata_path = Path(args.metadata or preset["metadata"])
    train_count = args.train_count if args.train_count is not None else preset["train_count"]
    val_count = args.val_count if args.val_count is not None else preset["val_count"]
    test_count = args.test_count if args.test_count is not None else preset["test_count"]
    max_grid_cells = (
        args.max_grid_cells if args.max_grid_cells is not None else preset["max_grid_cells"]
    )
    max_total_pairs = (
        args.max_total_pairs if args.max_total_pairs is not None else preset["max_total_pairs"]
    )

    cleanup_root: Path | None = None
    if args.source_dir:
        root = Path(args.source_dir)
    else:
        root = fetch_official_repo(args.zip_url)
        cleanup_root = root.parent.parent

    try:
        training_dir = root / "data" / "training"
        evaluation_dir = root / "data" / "evaluation"
        if not training_dir.is_dir() or not evaluation_dir.is_dir():
            raise SystemExit(
                f"Expected ARC-AGI-2 data directories under {root}, but did not find data/training and data/evaluation"
            )

        training_tasks = load_task_files(
            training_dir, max_grid_cells, max_total_pairs
        )
        evaluation_tasks = load_task_files(
            evaluation_dir, max_grid_cells, max_total_pairs
        )

        sampled_train_pool = sample_tasks(
            training_tasks, train_count + val_count, rng
        )
        sampled_eval_pool = sample_tasks(evaluation_tasks, test_count, rng)
        train_tasks = sampled_train_pool[:train_count]
        val_tasks = sampled_train_pool[train_count:]
        test_tasks = sampled_eval_pool

        rows = [
            *[dataset_case("train", task) for task in train_tasks],
            *[dataset_case("val", task) for task in val_tasks],
            *[dataset_case("test", task) for task in test_tasks],
        ]
        write_jsonl(output_path, rows)

        total = len(rows)
        metadata = {
            "source": "official ARC-AGI-2 public data",
            "source_root": str(root),
            "preset": args.preset,
            "seed": args.seed,
            "filters": {
                "max_grid_cells": max_grid_cells,
                "max_total_pairs": max_total_pairs,
            },
            "counts": {
                "train": len(train_tasks),
                "val": len(val_tasks),
                "test": len(test_tasks),
                "total": total,
            },
            "recommended_objective_split": {
                "train": len(train_tasks) / total if total else 0.0,
                "val": len(val_tasks) / total if total else 0.0,
                "test": len(test_tasks) / total if total else 0.0,
            },
            "train_task_ids": [task["task_id"] for task in train_tasks],
            "val_task_ids": [task["task_id"] for task in val_tasks],
            "test_task_ids": [task["task_id"] for task in test_tasks],
            "output": str(output_path),
        }
        write_json(metadata_path, metadata)

        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "metadata": str(metadata_path),
                    "counts": metadata["counts"],
                    "preset": args.preset,
                },
                indent=2,
            )
        )
    finally:
        if cleanup_root and cleanup_root.exists():
            shutil.rmtree(cleanup_root, ignore_errors=True)


if __name__ == "__main__":
    main()
