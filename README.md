Scaffold
========

A typed DSL and runtime for building, evaluating, and optimizing LLM computation graphs.

Overview
--------
Scaffold provides a language for defining typed computation graphs that wire together LLM prompts, tools, and control flow. Programs are verified at compile time (parsing, type-checking, static analysis) and executed by a built-in interpreter. An evolutionary optimizer with an optional LLM-guided meta-agent searches over graph structure and prompt content to maximize objective scores on datasets.

Core Concepts
-------------
- **Types**: Primitive, struct, list, map, option, and named aliases. Used everywhere for I/O validation.
- **Nodes**: Typed components — `prompt`, `tool`, `agent`, `verify` — with declared input/output types.
- **Graphs**: Computation graphs that wire nodes via steps, loops, conditionals, parallel fan-out, and emit.
- **Objectives**: Evaluation contracts that bind a graph to a dataset with checkers, metrics, and a score expression.
- **Topology**: Structural mutation rules (insert/remove steps, verify wrappers, component swaps) that the optimizer explores.
- **Meta-agent**: An LLM-guided mutation proposer that analyzes the optimization archive and proposes targeted mutations instead of random search.

What Scaffold Guarantees
------------------------
- **Parse-time safety**: The grammar only admits supported constructs.
- **Type-time safety**: A checker validates every declaration and expression.
- **Verify-time checks**: Optional static analyses (bounds, reachability) reject ill-formed orchestrations.
- **Runtime safety**: The interpreter validates inputs, outputs, and step wiring at execution time.

CLI
---
```
scaffold check FILE                     # parse + type check + verify
scaffold compile FILE [-o ir.json]      # lower to IR JSON
scaffold run FILE                       # execute the default graph
scaffold evaluate FILE --objective OBJ  # evaluate on dataset
scaffold optimize FILE --objective OBJ  # evolutionary optimization
  --max-candidates N                    # evolutionary generations
  --meta-model MODEL                    # enable LLM-guided meta-agent
  --meta-full-traces                    # full execution traces for meta-agent
  --concurrency N                       # parallel case evaluation
  --live                                # TUI visualization
  --report-dir DIR                      # persist reports
  --write-best FILE                     # freeze best candidate
```

Optimization
------------
The optimizer runs in three phases:

1. **Seeding** — evaluate the original graph as baseline.
2. **Tunable sweep** — enumerate all declared parameter combinations (skipped when meta-agent is active).
3. **Evolutionary** — mutate parent candidates via topology changes and content rewrites.

**Meta-agent** (`--meta-model`): Instead of random mutations, an LLM analyzes the archive (scores, failures, templates) and proposes targeted mutations. Content rewrites use an instructions-only approach that mechanically preserves data bindings and format blocks from the original template.

**Full execution traces** (`--meta-full-traces`): Sends full step-by-step execution traces for all failed cases to the meta-agent (instead of only the first 5). Also adds changed-case trace analysis showing traces for cases that flipped between generations, giving the meta-agent concrete evidence of what each mutation changed.

**Hierarchical optimization**: Objectives with `sub` blocks optimize sub-graphs first, freeze the best results into the IR, then optimize the parent graph.

Repository Layout
-----------------
- `crates/scaffold-syntax` — lexer/parser for the DSL
- `crates/scaffold-types` — type checker and type environment
- `crates/scaffold-verify` — static analyses (bounds, reachability, deadlock)
- `crates/scaffold-ir` — serializable IR and pretty-printer
- `crates/scaffold-runtime` — interpreter, LLM integration, optimizer, meta-agent
- `crates/scaffold-cli` — CLI binary with TUI visualization

Examples
--------
- `examples/long_memory_oracle_mini.scaffold` — hierarchical memory retrieval benchmark
- `examples/aider_polyglot.scaffold` — Aider polyglot coding benchmark (Python)
- `examples/text_classification.scaffold` — text classification benchmark (Meta-Harness comparison)

For LLMs
--------
Read the authoring guide: `docs/LLM_GUIDE.md` for syntax, patterns, and constraints to generate correct scaffolds.

Contributing
------------
Please open issues/PRs with clear problem statements and repros.
