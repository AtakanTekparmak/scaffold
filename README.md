Scaffold Lang
=============

--WIP--

Purpose
-------
- A tools‑first DSL and runtime for LLM‑authored automation that remains verifiable, analyzable, and safe to execute.
- Guarantees that generation mistakes are caught at compile time via parsing, typing and verification — not at runtime.
- Enables an interpreted inner loop for rapid iteration, with optional code generation for production binaries.

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
- Tools: Deterministic, typed building blocks. Implementations are restricted expressions/blocks (shell, foreign calls, control flow) with optional spec (pre/post, pure).
- Prompts: Single LLM calls with typed I/O; templates interpolate input values. Output schema is enforced.
- Agents: Multi‑turn LLM controllers with an explicit tool list, system prompt, reward/done/timeout policies.
- Pipelines: Fixed sequences of tool/prompt/evaluable expressions with typed input and output.

What Scaffold Guarantee
-----------------
- Parse‑time safety: The grammar only admits the constructs we support. No implicit execution of arbitrary host code.
- Type‑time safety: A checker validates every declaration and expression with a global type environment.
- Verify‑time checks: Optional static analyses (bounds, reachability/deadlock) reject ill‑formed orchestrations.
- Runtime safety: Interpreter and generated code continue to check pre/postconditions and report structured errors.

Workflow
--------
- Author: Write a `.scaffold` file using types, tools, prompts, agents, and pipelines.
- Check: `scaffold check file.scaffold` — parse + type + verification errors are surfaced with spans.
- Interpret (inner loop): `scaffold run file.scaffold --pipeline my_flow --input '{"..."}'` — no compilation required.
- Codegen (production): `scaffold codegen file.scaffold -o out_dir` — generates a Rust crate with CLI.
- Build (one‑shot binary): `scaffold build file.scaffold -o out_dir [--release]` — codegen + cargo build.

Compile‑Time Constraints (Examples)
-----------------------------------
- Undefined references:
  - Agents: tools listed must exist.
  - Pipelines: steps must resolve to a known tool or prompt when called.
- Type mismatches:
  - Pipeline steps must pass arguments that match the callee’s input type.
  - Field access must target declared fields of a struct‑typed value.
  - Pre/postconditions must be boolean; reward/done expressions must type‑check.

Interpreter vs Codegen
----------------------
- Interpreter (development/hot reload):
  - Executes IR directly with strong runtime checks.
  - Enforces typed shell parsing (int/float/bool/string/bytes) and assembles declared struct outputs from bindings.
  - Supports field access in pipeline assignments (e.g., `let words = result.count`).
- Codegen (production):
  - Emits a Rust crate that uses the same typed interfaces, native JSON schemas for LLM calls, and the same safety rails.
  - Generated crates can be built and distributed without this repository.

CLI Cheatsheet
--------------
- `scaffold check FILE` — parse + type check + verify a scaffold file.
- `scaffold parse FILE` — syntax only (for debugging).
- `scaffold compile FILE [-o ir.json]` — lower to IR JSON.
- `scaffold run FILE [--tool T|--prompt P|--agent A|--pipeline X] --input JSON` — interpret and execute.
- `scaffold codegen FILE -o DIR [--format]` — generate a Rust crate.
- `scaffold build FILE -o DIR [--release] [--runtime-path PATH|SCAFFOLD_RUNTIME_VERSION=x.y]` — generate and compile a binary.

Repository Layout
-----------------
- `crates/scaffold-syntax` — lexer/parser for the DSL.
- `crates/scaffold-types` — type checker and type environment.
- `crates/scaffold-verify` — static analyses (bounds, reachability, deadlock).
- `crates/scaffold-ir` — serializable IR: types, tools, prompts, agents, pipelines.
- `crates/scaffold-interpreter` — interpreter for the IR (Loader/Executor/ForeignRegistry).
- `crates/scaffold-runtime` — runtime utilities (shell, LLM, prompt manager, error, value, tracing).
- `crates/scaffold-codegen` — Rust code generator for tools/prompts/agents/pipelines.
- `crates/scaffold-cli` — command line interface providing `scaffold`.

For LLMs
--------
- Read the authoring guide: `docs/LLM_GUIDE.md` for exact syntax, patterns, and constraints to generate correct scaffolds.

Design Principles
-----------------
- Constrain the representation so an LLM can reliably produce correct scaffolds and failures are detectable early.
- Keep semantics explicit and analyzable: typed I/O everywhere, named interfaces, finite control constructs.
- Prefer clear, small, composable primitives (tools/prompts) over unconstrained general‑purpose code.
- Separate development and production paths: interpreter for the inner loop, codegen for distribution.

Status
------
- Parser, type checker, IR, interpreter, runtime, and CLI are complete and integrated.
- Codegen covers tools/prompts/agents/pipelines; binary builds are supported.
- Ongoing: deeper verification passes, more foreign module patterns, and expanded examples.

Contributing
------------
- Please open issues/PRs with clear problem statements and repros.
- Keep additions aligned with the core goal: verifiable, analyzable scaffolds authored by LLMs.
