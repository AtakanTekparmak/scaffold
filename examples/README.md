# Examples

This directory contains a small, task-first example set for the current Scaffold architecture.

- `tool_task.scaffold`: deterministic tool stages, nested tool composition, and an objective that optimizes over tool variants.
- `prompt_task.scaffold`: a single prompt stage with a typed harness and a simple objective.
- `revision_loop.scaffold`: a bounded task loop with a pre-check `while:` guard, a carried typed artifact, and an objective over rollout quality.

Useful commands:

```bash
scaffold check examples/tool_task.scaffold
scaffold check examples/prompt_task.scaffold
scaffold check examples/revision_loop.scaffold
scaffold optimize examples/tool_task.scaffold --objective uppercase_cleaning
```

Objective expressions can read rollout telemetry during optimization, including `rollout.stage_count`, `rollout.tool_call_count`, `rollout.prompt_call_count`, `rollout.agent_turn_count`, `rollout.loop_iteration_count`, and the structured `rollout.trace`.
