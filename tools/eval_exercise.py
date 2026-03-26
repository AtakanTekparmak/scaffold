#!/usr/bin/env python3
"""Evaluate a single exercise for the Aider polyglot benchmark.

Reads code from stdin, writes it to a temp copy of the exercise directory,
runs tests, and outputs a JSON result to stdout.

Always exits 0 — test failures are captured as data, not script errors.

Usage:
    echo 'def hello(): return "Hello, World!"' | \
        python3 tools/eval_exercise.py \
            --exercise-dir polyglot-benchmark/exercises/python/exercises/practice/hello-world \
            --language python
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

MAX_OUTPUT_LEN = 4000
TEST_TIMEOUT_SECS = 60


def strip_markdown_fences(code):
    # type: (str) -> str
    """Remove markdown code fences if present."""
    # Match ```python ... ``` or ```lang ... ``` or ``` ... ```
    pattern = r'^```[\w]*\s*\n(.*?)^```\s*$'
    match = re.search(pattern, code, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1)
    # Also handle case where the entire string is wrapped (no leading text)
    stripped = code.strip()
    if stripped.startswith('```'):
        lines = stripped.split('\n')
        # Remove first line (```lang) and last line (```)
        if lines[-1].strip() == '```':
            return '\n'.join(lines[1:-1])
    return code


def get_solution_filename(exercise_dir):
    # type: (Path) -> Optional[str]
    """Get the solution filename from .meta/config.json."""
    config_path = exercise_dir / ".meta" / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    files = config.get("files", {})
    solution = files.get("solution", [])
    return solution[0] if solution else None


def get_test_command(language):
    # type: (str) -> List[str]
    """Get the test command for a given language."""
    commands = {
        "python": ["python3", "-m", "pytest", "-x", "--tb=short", "--no-header", "-q"],
        "rust": ["cargo", "test", "--", "--include-ignored"],
        "go": ["go", "test", "./..."],
        "javascript": ["npx", "jest", "--no-coverage"],
        "java": ["./gradlew", "test"],
        "cpp": ["cmake", "--build", "build", "--target", "test"],
    }
    return commands.get(language, ["echo", f"unsupported language: {language}"])


def run_tests(work_dir, language):
    # type: (Path, str) -> Tuple[bool, str, int]
    """Run tests and return (passed, output, exit_code)."""
    cmd = get_test_command(language)
    env = os.environ.copy()

    # Java: remove @Disabled annotations so all tests run
    if language == "java":
        for java_file in work_dir.rglob("*.java"):
            content = java_file.read_text()
            if "@Disabled" in content:
                content = content.replace("@Disabled", "// @Disabled")
                java_file.write_text(content)

    # JavaScript: enable skipped tests
    if language == "javascript":
        for js_file in work_dir.rglob("*.spec.js"):
            content = js_file.read_text()
            if "xtest(" in content or "xdescribe(" in content:
                content = content.replace("xtest(", "test(")
                content = content.replace("xdescribe(", "describe(")
                js_file.write_text(content)

    # C++: enable all tests
    if language == "cpp":
        env["EXERCISM_RUN_ALL_TESTS"] = "1"
        # Build first
        build_dir = work_dir / "build"
        build_dir.mkdir(exist_ok=True)
        subprocess.run(
            ["cmake", "-G", "Unix Makefiles", ".."],
            cwd=build_dir, capture_output=True, timeout=30
        )
        subprocess.run(
            ["make"],
            cwd=build_dir, capture_output=True, timeout=60
        )

    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECS,
            env=env,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output, result.returncode
    except subprocess.TimeoutExpired:
        return False, f"Test timed out after {TEST_TIMEOUT_SECS}s", -1
    except FileNotFoundError as e:
        return False, f"Test command not found: {e}", -1


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single exercise")
    parser.add_argument("--exercise-dir", required=True, help="Path to exercise directory")
    parser.add_argument("--language", required=True, help="Programming language")
    args = parser.parse_args()

    exercise_dir = Path(args.exercise_dir)
    if not exercise_dir.exists():
        result = {"passed": False, "test_output": f"Exercise dir not found: {exercise_dir}", "exit_code": -1}
        print(json.dumps(result))
        sys.exit(0)

    # Read code from stdin
    code = sys.stdin.read()
    if not code.strip():
        result = {"passed": False, "test_output": "No code provided (empty stdin)", "exit_code": -1}
        print(json.dumps(result))
        sys.exit(0)

    # Strip markdown fences
    code = strip_markdown_fences(code)

    # Find solution filename
    solution_filename = get_solution_filename(exercise_dir)
    if not solution_filename:
        result = {"passed": False, "test_output": "Could not determine solution filename from config", "exit_code": -1}
        print(json.dumps(result))
        sys.exit(0)

    # Create temp copy
    work_dir = Path(tempfile.mkdtemp(prefix="scaffold_eval_"))
    try:
        # Copy exercise
        shutil.copytree(exercise_dir, work_dir, dirs_exist_ok=True)

        # Write code to solution file
        solution_path = work_dir / solution_filename
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        solution_path.write_text(code)

        # Run tests
        passed, test_output, exit_code = run_tests(work_dir, args.language)

        # Truncate output
        if len(test_output) > MAX_OUTPUT_LEN:
            test_output = test_output[:MAX_OUTPUT_LEN] + f"\n... (truncated, {len(test_output)} chars total)"

        result = {
            "passed": passed,
            "test_output": test_output,
            "exit_code": exit_code,
        }
        print(json.dumps(result))
    except Exception as e:
        result = {"passed": False, "test_output": f"Eval error: {e}", "exit_code": -1}
        print(json.dumps(result))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
