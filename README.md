Scaffold Lang
=============

--WIP--

Purpose
-------
- A tools‑first DSL and runtime for LLM‑authored automation that remains verifiable, analyzable, and safe to execute.
- Guarantees that generation mistakes are caught at compile time via parsing, typing and verification — not at runtime.
- Executes verified tasks directly from IR, with harness configuration layered on top of the task graph.

Why not “just Python”
---------------------
- Python accepts any syntactically valid program, but it does not encode scaffold semantics. An LLM can produce Python that runs yet violates the intended pipeline/tool contracts.
- Scaffold encodes those contracts in the language: typed I/O, restricted forms, analyzable control flow, and declared interfaces for tools/prompts/agents/pipelines.
- Errors surface at compile time:
  - Unknown tool/prompt/agent references
  - Type mismatches across steps and fields
  - Invalid field access or binding names
  - Ill‑typed pre/postconditions and agent expressions

Core Concepts
-------------
- Types: Primitive, struct, list, map, option, result, and named aliases. Used everywhere for I/O and validation.
- Artifacts: Typed task-flow slots that make intermediate state explicit and verifiable.
- Tasks: The execution unit. A task wires stages, loops, branches, and emits a typed final output.
- Harnesses: Typed overlays on tasks that bind or tune execution fields like model, temperature, and loop bounds.
- Objectives: Evaluation contracts for future optimization/search over harness space.
- Tools, prompts, and agents: Reusable typed components that tasks invoke as stages.

What Scaffold Guarantee
-----------------
- Parse‑time safety: The grammar only admits the constructs we support. No implicit execution of arbitrary host code.
- Type‑time safety: A checker validates every declaration and expression with a global type environment.
- Verify‑time checks: Optional static analyses (bounds, reachability/deadlock) reject ill‑formed orchestrations.
- Runtime safety: The interpreter continues to validate inputs, outputs, and stage wiring at execution time.

Workflow
--------
- Author: Write a `.scaffold` file using typed components plus first-class tasks and harnesses.
- Check: `scaffold check file.scaffold` — parse + type + verification errors are surfaced with spans.
- Compile: `scaffold compile file.scaffold -o output.json` — lower the verified program to serializable IR.
- Run: `scaffold run file.scaffold --task answer_question --input '{"..."}'` — execute a task directly from IR.
- Harnessed run: `scaffold run file.scaffold --task answer_question --harness baseline --input '{"..."}'` — execute the same task with a typed harness overlay.

Compile‑Time Constraints (Examples)
-----------------------------------
- Undefined references:
  - Agents: tools listed must exist.
  - Pipelines: steps must resolve to a known tool or prompt when called.
- Type mismatches:
  - Pipeline steps must pass arguments that match the callee’s input type.
  - Field access must target declared fields of a struct‑typed value.
  - Pre/postconditions must be boolean; reward/done expressions must type‑check.

Execution Model
---------------
- `scaffold run` is interpreter-first and task-centric.
- The runtime executes verified `task` IR directly instead of generating a temporary crate for inner-loop runs.
- Harnesses are applied as typed runtime overlays on top of a task.
- Objective-driven optimization remains part of the language direction, but the CLI optimizer is under redesign and is not exposed in the current command surface.

CLI Cheatsheet
--------------
- `scaffold check FILE` — parse + type check + verify a scaffold file.
- `scaffold parse FILE` — syntax only (for debugging).
- `scaffold compile FILE [-o ir.json]` — lower to IR JSON.
- `scaffold run FILE --task TASK [--harness H] --input JSON` — execute a task directly from IR.

Repository Layout
-----------------
- `crates/scaffold-syntax` — lexer/parser for the DSL.
- `crates/scaffold-types` — type checker and type environment.
- `crates/scaffold-verify` — static analyses (bounds, reachability, deadlock).
- `crates/scaffold-ir` — serializable IR for types, components, tasks, harnesses, and objectives.
- `crates/scaffold-runtime` — interpreter/runtime utilities (LLM, prompt manager, task execution, error, value, tracing).
- `crates/scaffold-codegen` — retained for future export/deployment work, not the primary execution path.
- `crates/scaffold-cli` — command line interface providing `scaffold`.

For LLMs
--------
- Read the authoring guide: `docs/LLM_GUIDE.md` for exact syntax, patterns, and constraints to generate correct scaffolds.

Design Principles
-----------------
- Constrain the representation so an LLM can reliably produce correct scaffolds and failures are detectable early.
- Keep semantics explicit and analyzable: typed I/O everywhere, named interfaces, finite control constructs.
- Prefer clear, small, composable primitives (tools/prompts) over unconstrained general‑purpose code.
- Make execution semantics interpreter-first so the language meaning is not defined by generated code.

Status
------
- Parser, type checker, verifier, IR, runtime interpreter, and task-centric CLI are integrated.
- Task/harness/objective syntax is present and task execution now runs directly from IR.
- Ongoing: migrate more examples to tasks, expand interpreter coverage for tool-backed stages, and rebuild optimizer/codegen on the new architecture.

Contributing
------------
- Please open issues/PRs with clear problem statements and repros.
- Keep additions aligned with the core goal: verifiable, analyzable scaffolds authored by LLMs.
