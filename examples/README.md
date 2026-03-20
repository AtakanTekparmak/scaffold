# Examples

This directory contains a small, task-first example set for the current Scaffold architecture.

- `tool_task.scaffold`: deterministic tool stages, nested tool composition, and an objective that optimizes over tool variants.
- `component_swap.scaffold`: a deterministic structural-search example where optimization uses `components()` to swap a stage onto a compatible tool and then freezes that choice back into a standalone scaffold.
- `prompt_task.scaffold`: a single prompt stage with a typed harness and a simple objective.
- `revision_loop.scaffold`: a bounded task loop with a pre-check `while:` guard, a carried typed artifact, and an objective over rollout quality.
- `banking77_classification.scaffold`: a real Hugging Face Banking77 slice with prompt variants, system-prompt variants, and a harness/objective setup for comparing baseline vs optimized intent-classification accuracy.
- `arc_agi2_benchmark.scaffold`: an ARC-AGI-2 benchmark track with both a cheaper mini slice and a larger benchmark slice, where each dataset row is one full task scored by exact whole-task output matching.

Useful commands:

```bash
scaffold check examples/tool_task.scaffold
scaffold check examples/component_swap.scaffold
scaffold check examples/prompt_task.scaffold
scaffold check examples/revision_loop.scaffold
scaffold check examples/banking77_classification.scaffold
scaffold check examples/arc_agi2_benchmark.scaffold
scaffold optimize examples/tool_task.scaffold --objective uppercase_cleaning
scaffold optimize examples/component_swap.scaffold --objective uppercase_cleaning --write-best outputs/component_swap_best.scaffold
```

Objective expressions can read rollout telemetry during optimization, including `rollout.stage_count`, `rollout.tool_call_count`, `rollout.prompt_call_count`, `rollout.agent_turn_count`, `rollout.loop_iteration_count`, and the structured `rollout.trace`.

Banking77 experiment workflow:

```bash
# Refresh the local HF snapshot if needed
uv run --python 3.11 --with datasets python tools/snapshot_banking77.py

# Baseline harness defaults
cargo run -q -p scaffold-cli -- evaluate examples/banking77_classification.scaffold --objective banking77_subset_accuracy

# GEPA-backed optimization over the declared harness search space
cargo run -q -p scaffold-cli -- optimize examples/banking77_classification.scaffold --objective banking77_subset_accuracy --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py"

# Freeze the winning harness into a runnable scaffold artifact
cargo run -q -p scaffold-cli -- optimize examples/banking77_classification.scaffold --objective banking77_subset_accuracy --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py" --write-best outputs/banking77_best.scaffold
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

# The same run, but with candidate reports written to disk
cargo run -q -p scaffold-cli -- optimize examples/arc_agi2_benchmark.scaffold --objective arc_public_mini --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py" --report-dir runs/arc_public_mini

# Or freeze the best evolved ARC harness into a standalone scaffold file
cargo run -q -p scaffold-cli -- optimize examples/arc_agi2_benchmark.scaffold --objective arc_public_mini --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py" --write-best outputs/arc_public_mini_best.scaffold

# GEPA-backed optimization over the larger benchmark slice
cargo run -q -p scaffold-cli -- optimize examples/arc_agi2_benchmark.scaffold --objective arc_public_subset --backend dspy --backend-command "uv run --python 3.11 --with 'dspy>=3' python tools/dspy_optimize.py"
```
