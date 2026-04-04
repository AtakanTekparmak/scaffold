# Scaffold DSL Reference

## Types

```scaffold
type ClassifyInput = { task_text: string, label_guide: string }
```

Primitives: `bool`, `int`, `float`, `string`, `bytes`, `any`
Containers: `list<T>`, `map<K,V>`, `option<T>`, `{ field: Type, ... }`

## Nodes

A node is a typed computational unit.

```scaffold
node classify: prompt {
    in: { task_text: string, label_guide: string }
    out: string
    template: file("prompts/classify/template")
    system: file("prompts/classify/system")
    model: "openai/gpt-5.4"
    temperature: 0.0
}

node eval_code: tool {
    in: { code: string, dir: string }
    out: string
    shell: "python3 tools/eval.py --dir '{{dir}}'"
    timeout: 120
}
```

**Kinds:** `prompt` (LLM call), `tool` (shell command), `agent` (multi-turn LLM + tools), `verify` (quality gate)

**Config fields:** `template`, `system`, `model`, `temperature`, `max_tokens`, `shell`, `timeout`, `on_error`, `tools`, `max_turns`

Templates use Jinja2: `{{ variable }}`, `{% for %}`, `{% if %}`.

## Graphs

A graph wires nodes into a directed pipeline.

```scaffold
graph classify {
    in: ClassifyInput
    out: string

    step result = classify_text(
        task_text: input.task_text,
        label_guide: input.label_guide
    )
    emit result
}
```

**Statements:**

| Statement | Syntax | Purpose |
|-----------|--------|---------|
| `step` | `step x = node(args)` | Call a node, bind result |
| `emit` | `emit expr` | Return graph output |
| `carry` | `carry x = expr` | Update loop variable |
| `if/else` | `if cond { ... } else { ... }` | Branch |
| `loop` | `loop (max: N, while: cond) { ... }` | Bounded iteration |
| `parallel` | `parallel (x in list, reduce: node) { ... }` | Fan-out |

**Expressions:** `+`, `-`, `*`, `/`, `==`, `!=`, `<`, `>`, `&&`, `||`, field access (`input.field`), builtins (`json_parse`, `len`, `lower`, `trim`, `contains`, `split`, `join`).

## Objectives

Declares how to evaluate and optimize a graph.

```scaffold
objective text_classify {
    graph: classify
    dataset: file("data/train.jsonl")

    checker correct { output == expected.label }
    metric accuracy { checker: correct }
    score: accuracy

    split { train: 0.5, val: 0.2, test: 0.3 }
    select { primary: accuracy }

    tune {
        classify_text.temperature in [0.0, 0.2, 0.5, 0.7]
    }

    topology {
        mutations: [set_config, insert_step, remove_step, replace_component]
        max_nodes: 8
        target_score: 0.9
    }
}
```

- **dataset**: JSONL, each line has `input`, `expected`, optional `id` and `domain` fields
- **checker**: boolean expression per case (`input`, `output`, `expected` in scope)
- **metric**: fraction of cases passing a checker
- **score**: optimization target (metric name or expression)
- **split**: train/val/test fractions
- **tune**: hyperparameter grid search space
- **topology**: structural mutation constraints, max graph size, target score

## Optimization

Three phases:

1. **Seed** — evaluate the original graph to establish a baseline
2. **Tunable sweep** — grid search over `tune` parameters
3. **Evolutionary search** — meta-agent (LLM) proposes typed mutations, runtime validates and evaluates

### Mutations

| Category | Mutations |
|----------|-----------|
| Content | `rewrite_prompt`, `rewrite_system`, `rewrite_shell`, `rewrite_tool_spec`, `attach_example_policy`, `add_local_checker` |
| Structural | `insert_step`, `remove_step`, `replace_component`, `insert_verify`, `wrap_retry`, `fan_out_parallel`, `add_prompt_step`, `set_config` |
| Decomposition | `propose_decomposition` (applies a motif) |

Content mutations are always allowed. Structural mutations are gated by the `topology.mutations` list.

### Motifs

Motifs are reusable graph transformations triggered by failure patterns. When the optimizer detects a failure cluster (e.g., label confusion), it can apply a matching motif to decompose a single step into a multi-step pattern.

| Motif | Pattern | When |
|-------|---------|------|
| `ShortlistSelect` | shortlist → choose → normalize | Label confusion among many classes |
| `VoteCritiqueRepair` | propose → critique → repair (deterministic gate) | High output variance |
| `NormalizeVerify` | raw → normalize | Format/spelling errors |
| `RouterExpert` | route → classify (with domain context) | Clear domain clusters |
| `RetrieveDecide` | retrieve facts → decide | Missing external knowledge |
| `GenerateValidateRefine` | generate → validate (tool) → refine | Programmatic correctness check |

### Meta-Agent Context

The meta-agent receives ~4-8KB of structured context per step:

- Pass/fail matrix and failure clusters
- Step-transition attribution (which step introduced/fixed errors)
- Prior mutation attempts with scores
- Motif suggestions mapped to failure patterns
- Saturated mutation families (blocked from re-proposal)

### Candidate Lifecycle

1. Meta-agent proposes a mutation as JSON
2. Mutation is validated against the type system and topology constraints
3. Applied to graph IR, producing a new candidate
4. Semantic hash computed — duplicates are skipped
5. Candidate evaluated on training cases
6. Results archived; best candidate's graph written as `.scaffold` output

## CLI

```bash
scaffold check FILE                                  # parse + type check + verify
scaffold compile FILE [-o ir.json]                   # lower to IR
scaffold run FILE --graph G --input JSON             # execute a graph
scaffold evaluate FILE --objective O                 # evaluate one candidate
scaffold optimize FILE --objective O                 # full optimization
scaffold optimize FILE --objective O --meta-model M  # with meta-agent
```

## Configuration

Set `OPENROUTER_API_KEY` in environment. Models are specified as `"provider/model"` in node definitions (e.g., `"openai/gpt-5.4"`, `"anthropic/claude-sonnet-4-5-20250929"`).
