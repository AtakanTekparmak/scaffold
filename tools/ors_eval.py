#!/usr/bin/env -S uv run --python 3.11
# /// script
# requires-python = ">=3.11"
# dependencies = ["openreward"]
# ///
"""Evaluate a single ORS task by submitting an agent answer.

Reads the agent's answer from stdin, submits it to the ORS environment
via a session tool call, and outputs a JSON result to stdout.

Always exits 0 — errors are captured as JSON with passed: false.

Usage:
    echo "42" | uv run tools/ors_eval.py \
        --env owner/env_name \
        --task-id abc123 \
        --tool submit \
        [--base-url https://api.openreward.ai] \
        [--answer-key answer]
"""

import argparse
import json
import re
import sys
from typing import Any

from openreward import OpenReward


def strip_markdown_fences(text):
    # type: (str) -> str
    """Remove markdown code fences if present."""
    pattern = r'^```[\w]*\s*\n(.*?)^```\s*$'
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1)
    stripped = text.strip()
    if stripped.startswith('```'):
        lines = stripped.split('\n')
        if lines[-1].strip() == '```':
            return '\n'.join(lines[1:-1])
    return text


def make_tool_args(raw_input, answer_key):
    # type: (str, str) -> Dict[str, Any]
    """Parse stdin into tool call arguments.

    If stdin is valid JSON (dict), use it directly as tool args.
    Otherwise wrap the text as {answer_key: text}.
    """
    text = strip_markdown_fences(raw_input).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return {answer_key: text}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single ORS task")
    parser.add_argument("--env", required=True, help="ORS environment name (e.g. owner/env_name)")
    parser.add_argument("--task-id", required=True, help="Task ID to evaluate")
    parser.add_argument("--tool", required=True, help="Tool name to call (e.g. submit)")
    parser.add_argument("--base-url", default=None, help="ORS API base URL")
    parser.add_argument("--answer-key", default="answer", help="Key name for wrapping text input (default: answer)")
    args = parser.parse_args()

    try:
        # Read answer from stdin
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            result = {"passed": False, "reward": 0.0, "finished": False,
                      "observation": "No input provided (empty stdin)", "total_reward": 0.0}
            print(json.dumps(result))
            sys.exit(0)

        # Initialize client
        client_kwargs = {}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        client = OpenReward(**client_kwargs)

        # Get environment and find the task
        env = client.environments.get(name=args.env)

        # List tasks to find the matching one — try all splits
        target_task = None
        for split in ["train", "validation", "test"]:
            try:
                tasks = env.list_tasks(split=split)
                for t in tasks:
                    tid = getattr(t, "id", None) or str(t)
                    if tid == args.task_id:
                        target_task = t
                        break
            except Exception:
                continue
            if target_task:
                break

        if target_task is None:
            result = {"passed": False, "reward": 0.0, "finished": False,
                      "observation": f"Task {args.task_id} not found in environment {args.env}",
                      "total_reward": 0.0}
            print(json.dumps(result))
            sys.exit(0)

        # Create session and call tool
        tool_args = make_tool_args(raw_input, args.answer_key)

        with env.session(task=target_task) as session:
            tool_result = session.call_tool(args.tool, tool_args)

            # Extract fields from ToolOutput
            reward = getattr(tool_result, "reward", 0.0) or 0.0
            finished = getattr(tool_result, "finished", False) or False

            # Extract observation from blocks
            blocks = getattr(tool_result, "blocks", []) or []
            if blocks:
                observation = "\n".join(
                    b.text if hasattr(b, "text") else str(b) for b in blocks
                )
            else:
                observation = ""

            if not finished:
                print(f"Warning: task not finished after tool call (single-turn bridge)", file=sys.stderr)

            result = {
                "passed": reward > 0,
                "reward": float(reward),
                "finished": finished,
                "observation": observation,
                "total_reward": float(reward),
            }
            print(json.dumps(result))

    except Exception as e:
        result = {"passed": False, "reward": 0.0, "finished": False,
                  "observation": f"Eval error: {e}", "total_reward": 0.0}
        print(json.dumps(result))

    sys.exit(0)


if __name__ == "__main__":
    main()
