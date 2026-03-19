# Examples

This directory contains a small, task-first example set for the current Scaffold architecture.

- `tool_task.scaffold`: deterministic tool stages, nested tool composition, and a harness-selected tool variant.
- `prompt_task.scaffold`: a single prompt stage with a typed harness and a simple objective.
- `revision_loop.scaffold`: a bounded task loop that carries a typed artifact and defines an objective over rollout quality.

Useful commands:

```bash
scaffold check examples/tool_task.scaffold
scaffold check examples/prompt_task.scaffold
scaffold check examples/revision_loop.scaffold
```
