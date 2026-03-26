# Examples

This directory contains example scaffold programs for the v2 architecture.

## Benchmarks

- `long_memory_oracle_mini.scaffold`: A LongMemEval-style benchmark with a hierarchical `extract -> retrieve -> answer` memory pipeline. Exercises memory extraction, temporal retrieval, multi-session reasoning, and abstention. Uses sub-objectives to optimize retrieval independently before optimizing the full answer pipeline.

- `aider_polyglot.scaffold`: An Aider polyglot coding benchmark (Python, 34 Exercism exercises). The graph generates code, runs pytest, and retries on failure. Uses `json_parse()` to inspect structured eval output in checkers and conditionals.

## Quick start

```bash
# Check a scaffold file (parse + type check + verify)
scaffold check examples/long_memory_oracle_mini.scaffold
scaffold check examples/aider_polyglot.scaffold

# Compile to IR JSON
scaffold compile examples/long_memory_oracle_mini.scaffold -o output.json
```

## Long-memory workflow

```bash
# Baseline evaluation
scaffold evaluate examples/long_memory_oracle_mini.scaffold \
  --objective long_memory_mini --live

# Optimize with meta-agent (hierarchical: sub-objectives first, then parent)
scaffold optimize examples/long_memory_oracle_mini.scaffold \
  --objective long_memory_mini \
  --max-candidates 8 \
  --meta-model gpt-4o \
  --report-dir runs/long_memory_mini \
  --write-best outputs/long_memory_mini_best.json \
  --live

# Random mutations (no meta-agent)
scaffold optimize examples/long_memory_oracle_mini.scaffold \
  --objective long_memory_mini \
  --max-candidates 8 \
  --report-dir runs/long_memory_mini \
  --live
```

## Aider polyglot workflow

```bash
# 1. Clone the benchmark repo
git clone https://github.com/Aider-AI/polyglot-benchmark.git

# 2. Prepare the dataset (34 Python exercises)
python3 tools/prepare_polyglot.py \
  --repo polyglot-benchmark \
  --language python \
  --output examples/datasets/aider_polyglot_python.jsonl

# 3. Baseline evaluation
scaffold evaluate examples/aider_polyglot.scaffold \
  --objective aider_polyglot --live

# 4. Optimize with meta-agent
scaffold optimize examples/aider_polyglot.scaffold \
  --objective aider_polyglot \
  --max-candidates 10 \
  --meta-model gpt-4o-mini \
  --live
```

## CLI flags

- `--live`: TUI visualization with real-time chart, log, and lineage tree
- `--meta-model MODEL`: Enable LLM-guided meta-agent mutations (e.g. `gpt-4o`, `gpt-4o-mini`)
- `--max-candidates N`: Number of evolutionary generations
- `--concurrency N`: Parallel case evaluation per candidate
- `--report-dir DIR`: Persist reports, best candidate details, and overrides
- `--write-best FILE`: Freeze best candidate IR to a file
