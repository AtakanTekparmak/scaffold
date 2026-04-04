# Scaffold

A DSL for writing and automatically optimizing LLM pipelines.

Write a typed graph of LLM calls and tools, define an objective with a dataset and checkers, and let an evolutionary optimizer search over prompt rewrites and structural mutations to maximize your score.

```scaffold
node classify: prompt {
    in: { text: string, labels: string }
    out: string
    template: "Classify this text:\n\n{{text}}\n\nLabels: {{labels}}"
    model: "openai/gpt-5.4"
}

graph classify_graph {
    in: { text: string, labels: string }
    out: string
    step result = classify(text: input.text, labels: input.labels)
    emit result
}

objective improve {
    graph: classify_graph
    dataset: file("data/train.jsonl")
    checker correct { output == expected.label }
    metric accuracy { checker: correct }
    score: accuracy
    topology { max_nodes: 8 target_score: 0.9 }
}
```

```bash
scaffold optimize example.scaffold --objective improve
```

## Install

```bash
cargo install --path crates/scaffold-cli
```

Requires `OPENROUTER_API_KEY` in environment.

## CLI

```
scaffold check FILE                        # parse + type check
scaffold run FILE --graph G --input JSON   # execute a graph
scaffold optimize FILE --objective O       # run optimizer
```

## Crates

| Crate | What |
|-------|------|
| `scaffold-syntax` | Lexer, parser |
| `scaffold-types` | Type checker |
| `scaffold-verify` | Static verification |
| `scaffold-ir` | IR types and lowering |
| `scaffold-runtime` | Executor, optimizer, meta-agent, LLM providers |
| `scaffold-cli` | CLI binary |

## Docs

See [`docs/dsl-reference.md`](docs/dsl-reference.md) for full DSL syntax and optimization system reference.
