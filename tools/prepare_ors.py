#!/usr/bin/env -S uv run --python 3.11
# /// script
# requires-python = ">=3.11"
# dependencies = ["openreward"]
# ///
"""Prepare an ORS environment dataset for scaffold.

Connects to an ORS environment, fetches tasks and prompts, and writes
a JSONL dataset file with one entry per task.

Usage:
    uv run tools/prepare_ors.py \
        --env owner/env_name \
        --split train \
        --output examples/datasets/ors_env.jsonl \
        [--base-url https://api.openreward.ai]
"""

import argparse
import json
import sys
from typing import Any

from openreward import OpenReward


def fetch_tasks(client, env_name, split):
    # type: (Any, str, str) -> List[Any]
    """Fetch tasks for the given environment and split."""
    env = client.environments.get(name=env_name)
    tasks = env.list_tasks(split=split)
    return list(tasks)


def fetch_prompt_and_tools(client, env_name, task):
    # type: (Any, str, Any) -> Optional[Dict[str, str]]
    """Fetch prompt and tools for a single task via a session."""
    env = client.environments.get(name=env_name)
    try:
        with env.session(task=task) as session:
            prompt = session.get_prompt()
            # Prompt may be a list of blocks or a string
            if isinstance(prompt, list):
                prompt_text = "\n".join(
                    b.text if hasattr(b, "text") else str(b) for b in prompt
                )
            else:
                prompt_text = str(prompt)

        # Fetch tool schemas
        try:
            tools = env.list_tools(format="openai")
            tools_json = json.dumps(tools, ensure_ascii=False)
        except Exception:
            tools_json = "[]"

        return {"task_prompt": prompt_text, "env_tools": tools_json}
    except Exception as e:
        print(f"  Warning: failed to fetch prompt for task {getattr(task, 'id', task)}: {e}", file=sys.stderr)
        return None


def process_task(client, env_name, task):
    # type: (Any, str, Any) -> Optional[Dict]
    """Process a single task into a dataset entry."""
    task_id = getattr(task, "id", None) or str(task)
    result = fetch_prompt_and_tools(client, env_name, task)
    if result is None:
        return None

    return {
        "id": task_id,
        "input": {
            "task_id": task_id,
            "task_prompt": result["task_prompt"],
            "env_tools": result["env_tools"],
            "env_name": env_name,
        },
        "expected": {},
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare ORS environment dataset for scaffold")
    parser.add_argument("--env", required=True, help="ORS environment name (e.g. owner/env_name)")
    parser.add_argument("--split", default="train", help="Task split to fetch (default: train)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--base-url", default=None, help="ORS API base URL (default: SDK default)")
    args = parser.parse_args()

    # Initialize client
    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenReward(**client_kwargs)

    # Fetch tasks
    print(f"Fetching tasks for {args.env} (split={args.split})...", file=sys.stderr)
    tasks = fetch_tasks(client, args.env, args.split)
    print(f"Found {len(tasks)} tasks", file=sys.stderr)

    if not tasks:
        print("Error: no tasks found", file=sys.stderr)
        sys.exit(1)

    # Process each task
    entries = []
    for i, task in enumerate(tasks):
        task_id = getattr(task, "id", None) or str(task)
        print(f"  [{i+1}/{len(tasks)}] Processing task {task_id}...", file=sys.stderr)
        entry = process_task(client, args.env, task)
        if entry:
            entries.append(entry)

    # Write JSONL
    from pathlib import Path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} tasks to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
