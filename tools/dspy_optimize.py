#!/usr/bin/env python3
"""Starter DSPy optimization backend for Scaffold.

This worker reads an OptimizationBackendRequest JSON payload from stdin and
prints an OptimizationBackendResponse JSON payload to stdout.

Current scope:
- Uses DSPy to propose candidate harness assignments from expanded tunable domains.
- Uses GEPA when DSPy 3.x is available and the backend can call back into Scaffold.
- Leaves objective evaluation, telemetry, and final selection to Scaffold.

Future scope:
- Promote prompt/system text fields into explicit DSPy modules.
- Let GEPA mutate those text surfaces directly.
- Feed rollout traces back into the proposal prompt.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def load_request() -> dict[str, Any]:
    try:
        return json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        fail(f"failed to parse Scaffold optimization request JSON: {exc}")


def load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def load_local_env() -> None:
    candidates: list[Path] = []
    for root in [
        Path.cwd(),
        Path(os.getenv("SCAFFOLD_WORKSPACE_DIR", "")),
        Path(os.getenv("SCAFFOLD_FILE_DIR", "")),
    ]:
        if not str(root):
            continue
        candidates.append(root / ".env")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        load_dotenv_file(candidate)


def configure_dspy() -> Any:
    try:
        import dspy
    except ImportError as exc:
        fail(
            "DSPy backend requires the 'dspy' package. "
            "Install it in the backend environment and re-run "
            "(for example: uv pip install dspy). "
            f"Original error: {exc}"
        )

    model = os.getenv("SCAFFOLD_DSPY_MODEL", "openai/gpt-4o-mini")
    api_base = os.getenv("SCAFFOLD_DSPY_API_BASE")
    temperature = float(os.getenv("SCAFFOLD_DSPY_TEMPERATURE", "0.0"))
    if "gpt-5" in model.lower():
        # LiteLLM rejects temperature=0 for GPT-5-family models.
        temperature = 1.0

    lm_kwargs: dict[str, Any] = {"temperature": temperature}

    try:
        import litellm

        litellm.drop_params = True
    except Exception:
        pass

    if os.getenv("OPENROUTER_API_KEY"):
        api_base = api_base or "https://openrouter.ai/api/v1"
        if "/" not in model:
            model = f"openai/{model}"
        lm = dspy.LM(
            model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            api_base=api_base,
            **lm_kwargs,
        )
    elif os.getenv("OPENAI_API_KEY"):
        if "/" not in model:
            model = f"openai/{model}"
        lm = dspy.LM(
            model,
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=api_base,
            **lm_kwargs,
        )
    elif os.getenv("ANTHROPIC_API_KEY"):
        if "/" not in model:
            model = f"anthropic/{model}"
        lm = dspy.LM(
            model,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            api_base=api_base,
            **lm_kwargs,
        )
    else:
        lm = dspy.LM(model, api_base=api_base, **lm_kwargs)

    dspy.configure(lm=lm)
    return dspy


def dspy_supports_gepa(dspy: Any) -> bool:
    return hasattr(dspy, "GEPA")


def build_spec(request: dict[str, Any]) -> str:
    dataset_preview = request.get("dataset", [])[: min(8, len(request.get("dataset", [])))]
    return (
        "You are proposing candidate harness assignments for a typed AI runtime.\n"
        "Return a JSON array of candidate assignment objects.\n"
        "Each candidate maps tunable paths to one of the allowed JSON values.\n"
        "Do not invent paths or values. Prefer a diverse but plausible set.\n\n"
        f"Objective: {request['objective_name']}\n"
        f"Task: {request['task']['name']}\n"
        f"Harness: {request['harness']['name']}\n"
        f"Max candidates: {request['max_candidates']}\n\n"
        "Tunable domains:\n"
        f"{json.dumps(request['tunables'], indent=2)}\n\n"
        "Objective definition:\n"
        f"{json.dumps(request['objective'], indent=2)}\n\n"
        "Dataset preview:\n"
        f"{json.dumps(dataset_preview, indent=2)}\n"
    )


def option_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fallback_candidates(request: dict[str, Any]) -> list[dict[str, Any]]:
    tunables = request.get("tunables", [])
    candidates: list[dict[str, Any]] = [{}]

    # Seed with one-path-at-a-time changes so the backend is still useful even
    # when DSPy proposes nothing usable.
    for tunable in tunables:
        path = tunable["path"]
        for option in tunable.get("options", []):
            candidates.append({path: option})
            if len(candidates) >= request["max_candidates"]:
                return candidates[: request["max_candidates"]]

    return candidates[: request["max_candidates"]]


def validate_candidates(
    request: dict[str, Any], proposed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    allowed = {
        tunable["path"]: {option_key(option): option for option in tunable.get("options", [])}
        for tunable in request.get("tunables", [])
    }

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in proposed:
        if not isinstance(candidate, dict):
            continue
        normalized: dict[str, Any] = {}
        for path, value in candidate.items():
            if path not in allowed:
                continue
            key = option_key(value)
            if key not in allowed[path]:
                continue
            normalized[path] = allowed[path][key]

        dedupe_key = option_key(normalized)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        validated.append(normalized)
        if len(validated) >= request["max_candidates"]:
            break

    return validated


def extract_json_payload(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_candidate_payload(raw: str) -> list[dict[str, Any]]:
    payload = extract_json_payload(raw)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict) and "assignments" in parsed:
        parsed = parsed["assignments"]
    if isinstance(parsed, dict):
        return [parsed]
    if not isinstance(parsed, list):
        return []
    return [candidate for candidate in parsed if isinstance(candidate, dict)]


def propose_with_dspy(dspy: Any, request: dict[str, Any]) -> list[dict[str, Any]]:
    class CandidateProposal(dspy.Signature):
        spec = dspy.InputField()
        assignments_json = dspy.OutputField(
            desc=(
                "A JSON array of candidate assignment objects. "
                "Each object maps tunable paths to allowed JSON values. "
                "Return at most the requested number of candidates."
            )
        )

    proposer = dspy.Predict(CandidateProposal)
    result = proposer(spec=build_spec(request))
    raw = getattr(result, "assignments_json", "")
    return parse_candidate_payload(raw)


def build_focus(case: dict[str, Any], index: int) -> str:
    case_id = case.get("id") or f"case_{index + 1}"
    return (
        f"Focus case: {case_id}\n"
        f"Input:\n{json.dumps(case.get('input'), indent=2)}\n"
        f"Expected:\n{json.dumps(case.get('expected'), indent=2)}\n"
    )


def build_gepa_examples(dspy: Any, request: dict[str, Any]) -> list[Any]:
    spec = build_spec(request)
    dataset = request.get("dataset", [])
    if not dataset:
        return [dspy.Example(spec=spec, focus="Optimize overall objective.").with_inputs("spec", "focus")]

    examples = []
    for index, case in enumerate(dataset[: max(1, min(len(dataset), 8))]):
        example = dspy.Example(
            spec=spec,
            focus=build_focus(case, index),
            case_id=case.get("id"),
        ).with_inputs("spec", "focus")
        examples.append(example)
    return examples


def scaffold_cli_command() -> list[str]:
    cli_bin = os.getenv("SCAFFOLD_CLI_BIN")
    source_file = os.getenv("SCAFFOLD_SOURCE_FILE")
    if not cli_bin or not source_file:
        raise RuntimeError(
            "GEPA candidate evaluation requires SCAFFOLD_CLI_BIN and "
            "SCAFFOLD_SOURCE_FILE to be set by Scaffold."
        )
    return [cli_bin, "internal-evaluate-candidate", source_file]


def evaluate_candidate_with_scaffold(
    request: dict[str, Any],
    candidate: dict[str, Any],
    case_id: str | None,
) -> dict[str, Any]:
    command = scaffold_cli_command()
    command.extend(
        [
            "--objective",
            request["objective_name"],
            "--assignments",
            json.dumps(candidate, separators=(",", ":")),
        ]
    )
    if case_id:
        command.extend(["--case-id", case_id])
    if os.getenv("SCAFFOLD_DSPY_EVAL_VERIFY") == "1":
        command.append("--verify")
    if os.getenv("SCAFFOLD_CONFIG_PATH"):
        command.extend(["--config", os.environ["SCAFFOLD_CONFIG_PATH"]])

    completed = subprocess.run(
        command,
        cwd=os.getenv("SCAFFOLD_WORKSPACE_DIR") or None,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Scaffold candidate evaluation failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip() or '<empty>'}"
        )
    return json.loads(completed.stdout)


def feedback_from_report(
    candidate: dict[str, Any], report: dict[str, Any], case_id: str | None
) -> str:
    train = report.get("train", {})
    metrics = train.get("metrics", {})
    metric_parts = ", ".join(
        f"{name}={value:.4f}" for name, value in sorted(metrics.items())
    ) or "no metrics"
    focus = case_id or "all cases"
    return (
        f"Focus: {focus}. "
        f"Primary score={train.get('primary', 0.0):.4f}; "
        f"objective score={train.get('score', 0.0):.4f}. "
        f"Metrics: {metric_parts}. "
        f"Assignments: {json.dumps(candidate, sort_keys=True)}."
    )


def propose_with_gepa(dspy: Any, request: dict[str, Any]) -> list[dict[str, Any]]:
    class CandidateProposal(dspy.Signature):
        spec = dspy.InputField()
        focus = dspy.InputField(
            desc="A dataset case or search angle to optimize candidate assignments for."
        )
        assignments_json = dspy.OutputField(
            desc=(
                "A JSON object mapping tunable paths to allowed JSON values. "
                "Return exactly one candidate assignment object."
            )
        )

    trainset = build_gepa_examples(dspy, request)
    if not trainset:
        return []

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        candidates = validate_candidates(
            request,
            parse_candidate_payload(getattr(pred, "assignments_json", "")),
        )
        if not candidates:
            return dspy.Prediction(
                score=0.0,
                feedback="Returned invalid assignments JSON or unsupported tunable values.",
            )

        candidate = candidates[0]
        case_id = getattr(gold, "case_id", None)
        try:
            report = evaluate_candidate_with_scaffold(request, candidate, case_id)
        except Exception as exc:
            return dspy.Prediction(
                score=0.0,
                feedback=f"Scaffold evaluation failed for this candidate: {exc}",
            )

        score = float(report.get("train", {}).get("primary", 0.0))
        return dspy.Prediction(
            score=score,
            feedback=feedback_from_report(candidate, report, case_id),
        )

    optimizer = dspy.GEPA(
        metric=metric,
        auto=os.getenv("SCAFFOLD_DSPY_GEPA_AUTO", "light"),
        reflection_lm=getattr(getattr(dspy, "settings", None), "lm", None),
    )
    student = dspy.Predict(CandidateProposal)
    with redirect_stdout(StringIO()):
        optimized = optimizer.compile(student, trainset=trainset, valset=trainset)

    candidates: list[dict[str, Any]] = []
    for example in trainset:
        with redirect_stdout(StringIO()):
            prediction = optimized(spec=example.spec, focus=example.focus)
        candidates.extend(
            validate_candidates(
                request,
                parse_candidate_payload(getattr(prediction, "assignments_json", "")),
            )
        )

    with redirect_stdout(StringIO()):
        overall_prediction = optimized(
            spec=build_spec(request),
            focus="Optimize for the overall objective across the full dataset.",
        )
    candidates.extend(
        validate_candidates(
            request,
            parse_candidate_payload(getattr(overall_prediction, "assignments_json", "")),
        )
    )

    return candidates


def main() -> None:
    load_local_env()
    request = load_request()
    dspy = configure_dspy()
    try:
        if dspy_supports_gepa(dspy):
            proposed = propose_with_gepa(dspy, request)
        else:
            proposed = propose_with_dspy(dspy, request)
    except Exception as exc:
        if os.getenv("SCAFFOLD_DSPY_DEBUG") == "1":
            raise
        fail(
            "DSPy backend failed while proposing candidates. "
            "Check provider credentials such as OPENROUTER_API_KEY, OPENAI_API_KEY, "
            "or ANTHROPIC_API_KEY, and optionally override SCAFFOLD_DSPY_MODEL. "
            f"Original error: {exc}"
        )
    validated = validate_candidates(request, proposed)
    if not validated:
        validated = fallback_candidates(request)

    response = {
        "assignments": validated[: request["max_candidates"]],
        "truncated": len(validated) > request["max_candidates"],
    }
    json.dump(response, sys.stdout)


if __name__ == "__main__":
    main()
