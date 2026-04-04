#!/usr/bin/env python3
"""Prepare the Aider polyglot benchmark dataset for scaffold.

Reads the polyglot-benchmark repo and produces a JSONL dataset file
with one entry per exercise.

Usage:
    python3 tools/prepare_polyglot.py \
        --repo polyglot-benchmark \
        --language python \
        --output examples/datasets/aider_polyglot_python.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


def find_exercises(repo_path, language):
    # type: (Path, str) -> List[Path]
    """Find all exercise directories for a given language."""
    exercises_dir = repo_path / language / "exercises" / "practice"
    if not exercises_dir.exists():
        print(f"Error: exercises directory not found: {exercises_dir}", file=sys.stderr)
        sys.exit(1)
    return sorted(d for d in exercises_dir.iterdir() if d.is_dir())


def load_instructions(exercise_dir):
    # type: (Path) -> str
    """Load the exercise instructions from .docs/."""
    docs_dir = exercise_dir / ".docs"
    parts = []

    intro = docs_dir / "introduction.md"
    if intro.exists():
        parts.append(intro.read_text())

    instructions = docs_dir / "instructions.md"
    if instructions.exists():
        parts.append(instructions.read_text())

    append = docs_dir / "instructions.append.md"
    if append.exists():
        parts.append(append.read_text())

    return "\n\n".join(parts).strip()


def load_config(exercise_dir):
    # type: (Path) -> dict
    """Load .meta/config.json."""
    config_path = exercise_dir / ".meta" / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def get_solution_files(config):
    # type: (dict) -> List[str]
    """Extract solution file paths from config."""
    files = config.get("files", {})
    solution = files.get("solution", [])
    return solution


def process_exercise(exercise_dir, language, repo_path):
    # type: (Path, str, Path) -> Optional[dict]
    """Process a single exercise into a dataset entry."""
    name = exercise_dir.name

    instructions = load_instructions(exercise_dir)
    if not instructions:
        print(f"  Skipping {name}: no instructions found", file=sys.stderr)
        return None

    config = load_config(exercise_dir)
    solution_files = get_solution_files(config)

    if not solution_files:
        print(f"  Skipping {name}: no solution files in config", file=sys.stderr)
        return None

    # For now, handle single solution file exercises
    if len(solution_files) > 1:
        print(f"  Warning: {name} has {len(solution_files)} solution files, using first", file=sys.stderr)

    stub_filename = solution_files[0]
    stub_path = exercise_dir / stub_filename
    if not stub_path.exists():
        print(f"  Skipping {name}: stub file not found: {stub_filename}", file=sys.stderr)
        return None

    stub_content = stub_path.read_text()

    # Relative exercise dir from the scaffold project root
    relative_dir = str(exercise_dir.relative_to(repo_path.parent)) if repo_path.is_absolute() else str(exercise_dir)

    return {
        "id": f"{language}/{name}",
        "input": {
            "instructions": instructions,
            "stub_filename": stub_filename,
            "stub_content": stub_content,
            "language": language,
            "exercise_dir": relative_dir,
        },
        "expected": {"passed": True},
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare polyglot benchmark dataset for scaffold")
    parser.add_argument("--repo", required=True, help="Path to cloned polyglot-benchmark repo")
    parser.add_argument("--language", required=True, help="Language to extract (e.g. python, rust, go)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        print(f"Error: repo path not found: {repo_path}", file=sys.stderr)
        sys.exit(1)

    exercises = find_exercises(repo_path, args.language)
    print(f"Found {len(exercises)} exercises for {args.language}", file=sys.stderr)

    entries = []
    for exercise_dir in exercises:
        entry = process_exercise(exercise_dir, args.language, repo_path)
        if entry:
            entries.append(entry)

    # Write JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} exercises to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
