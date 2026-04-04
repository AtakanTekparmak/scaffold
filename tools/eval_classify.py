#!/usr/bin/env python3
"""Evaluate a text classification prediction.

Reads the predicted label from stdin, compares to the expected label
passed via --expected. Case-insensitive, whitespace-stripped exact match.

Always exits 0 — mismatches are captured as data, not script errors.

Usage:
    echo "Migraine" | python3 tools/eval_classify.py --expected "Migraine"
"""

import argparse
import json
import re
import sys


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a classification prediction")
    parser.add_argument("--expected", required=True, help="Expected label")
    args = parser.parse_args()

    prediction = sys.stdin.read()
    prediction = strip_markdown_fences(prediction).strip()
    expected = args.expected.strip()

    passed = prediction.lower() == expected.lower()

    result = {
        "passed": passed,
        "predicted": prediction,
        "expected": expected,
    }
    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
