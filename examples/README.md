# Examples

This directory contains a small, task-first example set for the current Scaffold architecture.

- `tool_task.scaffold`: deterministic tool stages, nested tool composition, and an objective that optimizes over tool variants.
- `component_swap.scaffold`: a deterministic structural-search example where optimization uses `components()` to swap a stage onto a compatible tool and then freezes that choice back into a standalone scaffold.
- `prompt_task.scaffold`: a single prompt stage with a typed harness and a simple objective.
- `revision_loop.scaffold`: a bounded task loop with a pre-check `while:` guard, a carried typed artifact, and an objective over rollout quality.
- `banking77_classification.scaffold`: a real Hugging Face Banking77 slice with prompt variants, system-prompt variants, and a harness/objective setup for comparing baseline vs optimized intent-classification accuracy.
- `symbolic_rule_induction.scaffold`: a small local exact-match benchmark where the model infers a hidden string transformation rule from examples and applies it to a query. It is much cheaper than ARC and avoids public-dataset leakage.
- `symbolic_rule_tiny.scaffold`: the same idea, but only 4 total cases, intended for fast optimization/debugging loops.
- `symbolic_rule_medium.scaffold`: a middle benchmark with 6 compositional rule-induction cases. It is still cheap enough for iteration, but harder than the tiny smoke test and a better target for harness evolution.
- `long_memory_oracle_mini.scaffold`: a small LongMemEval-style benchmark with an explicit `extract -> retrieve -> answer` memory pipeline. It is local and synthetic, but designed to exercise memory artifacts, updates, temporal reasoning, multi-session retrieval, and abstention.
- `longmemeval_oracle.scaffold`: the same typed memory pipeline wired to the official LongMemEval `oracle` release, with both a real-data mini subset for iteration and a larger full objective for evaluation.
- `arc_agi2_benchmark.scaffold`: an ARC-AGI-2 benchmark track with both a cheaper mini slice and a larger benchmark slice, where each dataset row is one full task scored by exact whole-task output matching.
- `arc_agi2_single_model.scaffold`: the same ARC mini track, but with the model pinned to `gpt-5-mini` and a richer `solve -> review -> repair` harness so optimization can improve structure as well as prompts.

Useful commands:

```bash
scaffold check examples/tool_task.scaffold
scaffold check examples/component_swap.scaffold
scaffold check examples/prompt_task.scaffold
scaffold check examples/revision_loop.scaffold
scaffold check examples/banking77_classification.scaffold
scaffold check examples/symbolic_rule_induction.scaffold
scaffold check examples/symbolic_rule_tiny.scaffold
scaffold check examples/symbolic_rule_medium.scaffold
scaffold check examples/long_memory_oracle_mini.scaffold
python3 tools/snapshot_longmemeval_oracle.py
scaffold check examples/longmemeval_oracle.scaffold
scaffold check examples/arc_agi2_benchmark.scaffold
scaffold check examples/arc_agi2_single_model.scaffold
scaffold optimize examples/tool_task.scaffold --objective uppercase_cleaning
scaffold optimize examples/component_swap.scaffold --objective uppercase_cleaning --write-best outputs/component_swap_best.scaffold
```

Objective expressions can read rollout telemetry during optimization, including `rollout.stage_count`, `rollout.tool_call_count`, `rollout.prompt_call_count`, `rollout.agent_turn_count`, `rollout.loop_iteration_count`, and the structured `rollout.trace`.

For a nicer interactive CLI experience, add `--live` to `run`, `evaluate`, or `optimize`. That switches stderr from raw JSONL trace records to human-readable progress logs.

Banking77 experiment workflow:

```bash
# Refresh the local HF snapshot if needed
uv run --python 3.11 --with datasets python tools/snapshot_banking77.py

# Baseline harness defaults
cargo run -q -p scaffold-cli -- evaluate examples/banking77_classification.scaffold --objective banking77_subset_accuracy

# GEPA-backed optimization over the declared harness search space
cargo run -q -p scaffold-cli -- optimize examples/banking77_classification.scaffold --objective banking77_subset_accuracy --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py"

# Archive-backed evolutionary optimization fully inside Scaffold, including lineage
# tracking and narrow optimizer-policy evolution (parent selection / branching)
cargo run -q -p scaffold-cli -- optimize examples/banking77_classification.scaffold --objective banking77_subset_accuracy --backend evolutionary --max-candidates 16

# Freeze the winning harness into a runnable scaffold artifact
cargo run -q -p scaffold-cli -- optimize examples/banking77_classification.scaffold --objective banking77_subset_accuracy --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py" --write-best outputs/banking77_best.scaffold
```

Cheap local symbolic-rule workflow:

```bash
# Baseline on a small exact-match local benchmark with gpt-5-mini fixed
cargo run -q -p scaffold-cli -- evaluate examples/symbolic_rule_induction.scaffold --objective symbolic_rule_accuracy

# Fast archive-backed optimization without ARC-level latency
cargo run -q -p scaffold-cli -- optimize examples/symbolic_rule_induction.scaffold --objective symbolic_rule_accuracy --backend evolutionary --max-candidates 16 --live

# Tiny version for quick iteration
cargo run -q -p scaffold-cli -- evaluate examples/symbolic_rule_tiny.scaffold --objective symbolic_rule_tiny_accuracy --live
cargo run -q -p scaffold-cli -- optimize examples/symbolic_rule_tiny.scaffold --objective symbolic_rule_tiny_accuracy --backend evolutionary --max-candidates 6 --live

# Better middle benchmark for actual harness-improvement experiments
cargo run -q -p scaffold-cli -- evaluate examples/symbolic_rule_medium.scaffold --objective symbolic_rule_medium_accuracy --live
cargo run -q -p scaffold-cli -- optimize examples/symbolic_rule_medium.scaffold --objective symbolic_rule_medium_accuracy --backend evolutionary --max-candidates 6 --report-dir runs/symbolic_rule_medium_accuracy --write-best outputs/symbolic_rule_medium_accuracy_best.scaffold --live

# Stop once the weighted train/val/test primary is "good enough"
cargo run -q -p scaffold-cli -- optimize examples/symbolic_rule_medium.scaffold --objective symbolic_rule_medium_accuracy --backend evolutionary --max-candidates 6 --early-stop-primary-threshold 0.9 --report-dir runs/symbolic_rule_medium_accuracy --write-best outputs/symbolic_rule_medium_accuracy_best.scaffold --live
```

Long-memory workflow:

```bash
# Local LongMemEval-style oracle mini benchmark with explicit memory artifacts
cargo run -q -p scaffold-cli -- evaluate examples/long_memory_oracle_mini.scaffold --objective long_memory_oracle_mini_accuracy --live

# Optimize the memory harness rather than relying on hidden chat history
cargo run -q -p scaffold-cli -- optimize examples/long_memory_oracle_mini.scaffold --objective long_memory_oracle_mini_accuracy --backend evolutionary --max-candidates 8 --report-dir runs/long_memory_oracle_mini_accuracy --write-best outputs/long_memory_oracle_mini_best.scaffold --live

# Build the official LongMemEval oracle snapshots (real-data full + mini)
python3 tools/snapshot_longmemeval_oracle.py

# Evaluate the real-data mini slice first
cargo run -q -p scaffold-cli -- evaluate examples/longmemeval_oracle.scaffold --objective longmemeval_oracle_mini_accuracy --live

# Optimize the typed memory harness on the real-data mini slice
cargo run -q -p scaffold-cli -- optimize examples/longmemeval_oracle.scaffold --objective longmemeval_oracle_mini_accuracy --backend evolutionary --max-candidates 8 --report-dir runs/longmemeval_oracle_mini_accuracy --write-best outputs/longmemeval_oracle_mini_best.scaffold --live

# Then run the larger real-data oracle snapshot for evaluation
cargo run -q -p scaffold-cli -- evaluate examples/longmemeval_oracle.scaffold --objective longmemeval_oracle_accuracy --live
```

ARC-AGI-2 benchmark workflow:

```bash
# Refresh the cheap mini slice first
python3 tools/snapshot_arc_agi2.py --preset mini

# Or refresh the larger benchmark slice
python3 tools/snapshot_arc_agi2.py

# Baseline whole-task exact evaluation on the mini slice
cargo run -q -p scaffold-cli -- evaluate examples/arc_agi2_benchmark.scaffold --objective arc_public_mini

# Baseline whole-task exact evaluation on the larger slice
cargo run -q -p scaffold-cli -- evaluate examples/arc_agi2_benchmark.scaffold --objective arc_public_subset

# GEPA-backed optimization over the declared ARC harness search space on the mini slice
cargo run -q -p scaffold-cli -- optimize examples/arc_agi2_benchmark.scaffold --objective arc_public_mini --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py"

# The same ARC mini benchmark, but with the model fixed to gpt-5-mini so only the harness evolves
cargo run -q -p scaffold-cli -- evaluate examples/arc_agi2_single_model.scaffold --objective arc_public_mini_single_model
cargo run -q -p scaffold-cli -- optimize examples/arc_agi2_single_model.scaffold --objective arc_public_mini_single_model --backend evolutionary --max-candidates 16
cargo run -q -p scaffold-cli -- optimize examples/arc_agi2_single_model.scaffold --objective arc_public_mini_single_model --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py"
cargo run -q -p scaffold-cli -- optimize examples/arc_agi2_single_model.scaffold --objective arc_public_mini_single_model --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py" --live

# The same run, but with candidate summaries, per-rollout JSON artifacts,
# and lineage visualizer files (`lineage.json`, `lineage.mmd`, `lineage.html`) written to disk
cargo run -q -p scaffold-cli -- optimize examples/arc_agi2_benchmark.scaffold --objective arc_public_mini --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py" --report-dir runs/arc_public_mini

# Or freeze the best evolved ARC harness into a standalone scaffold file
cargo run -q -p scaffold-cli -- optimize examples/arc_agi2_benchmark.scaffold --objective arc_public_mini --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py" --write-best outputs/arc_public_mini_best.scaffold

# GEPA-backed optimization over the larger benchmark slice
cargo run -q -p scaffold-cli -- optimize examples/arc_agi2_benchmark.scaffold --objective arc_public_subset --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py"
```
