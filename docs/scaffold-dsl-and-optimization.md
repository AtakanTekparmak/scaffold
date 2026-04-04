# Scaffold: DSL and Optimization System

## 1. What Scaffold Is

Scaffold is a system for **writing, evaluating, and automatically optimizing LLM-powered computational graphs**. It has three parts:

1. **A domain-specific language (DSL)** for declaring typed nodes (LLM calls, shell tools, agents), wiring them into directed graphs with control flow (if/else, loops, parallel fan-out), and specifying evaluation objectives (dataset, checkers, metrics, score).

2. **A runtime** that executes these graphs: resolves template variables, calls LLM providers, runs shell commands, evaluates checker expressions, and computes scores.

3. **An evolutionary optimizer** that searches over both the prompt content and the graph topology to maximize the evaluation score. An optional **meta-agent** (an LLM itself) can propose mutations instead of random search.

---

## 2. The DSL

A `.scaffold` file contains four kinds of top-level declarations:

### 2.1 Types

```
type ClassifyInput = { task_text: string, label_guide: string }
type Labels = list<string>
```

Primitives: `bool`, `int`, `float`, `string`, `bytes`, `any`
Containers: `list<T>`, `map<K,V>`, `option<T>`, `{ field: Type, ... }`

### 2.2 Nodes

A node is a named computational unit with typed input/output and a kind.

```
node classify_text: prompt {
    in: { task_text: string, label_guide: string }
    out: string
    template: file("prompts/classify/template")  // Jinja2
    system: file("prompts/classify/system")
    model: "openai/gpt-4o"
    temperature: 0.0
}

node eval_code: tool {
    in: { code: string, exercise_dir: string }
    out: string
    shell: "python3 tools/eval.py --dir \"{{exercise_dir}}\" <<'EOF'\n{{code}}\nEOF"
    timeout: 120
}
```

**Node kinds:**
- `prompt` — single LLM call (template → completion)
- `tool` — shell command (deterministic, no hallucination)
- `agent` — multi-turn LLM with tool access
- `verify` — LLM-based quality check (used by `insert_verify` / `wrap_retry` mutations)

**Config fields:** `template`, `system`, `model`, `temperature`, `max_tokens`, `shell`, `timeout`, `on_error: abort | retry(N)`, `tools: [...]`, `max_turns`, `json: { key: expr }`

### 2.3 Graphs

A graph wires nodes into a directed computation with control flow.

```
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

**Statement types:**

| Statement | Syntax | Purpose |
|-----------|--------|---------|
| `step` | `step x = node(args)` | Call a node, bind result to `x` |
| `emit` | `emit expr` | Return value from graph |
| `carry` | `carry x = expr` | Update a loop-local variable |
| `if/else` | `if cond { ... } else { ... }` | Conditional branching |
| `loop` | `loop (max: N, while: cond) { ... }` | Bounded iteration with `carry` for state |
| `parallel` | `parallel (x in list, reduce: node) { ... }` | Fan-out over collection |
| `choose` | `choose [alt1, alt2, alt3]` | Non-deterministic alternative selection |

**Expressions:** standard operators (`+`, `-`, `*`, `/`, `==`, `!=`, `<`, `>`, `&&`, `||`), field access (`input.field`), indexing (`list[0]`), builtins (`json_parse(s)`, `len(x)`, `lower(s)`, `trim(s)`, `contains(h,n)`, `split(s,sep)`, `join(list,sep)`).

**Templates** use Jinja2: `{{ variable }}`, `{% for x in list %}`, `{% if cond %}`, `{% raw %}`.

### 2.4 Objectives

An objective declares how to evaluate and optimize a graph.

```
objective text_classify {
    graph: classify
    dataset: file("datasets/symptom2disease.jsonl")

    checker correct { output == expected.label }
    metric accuracy { checker: correct }
    score: accuracy

    split { train: 0.28, val: 0.17, test: 0.55 }
    select { primary: accuracy }

    tune {
        classify_text.temperature in [0.0, 0.2, 0.5, 0.7, 1.0]
    }

    topology {
        mutations: [set_config, insert_step, remove_step, replace_component]
        max_nodes: 8
        target_score: 0.9
    }
}
```

- **Dataset**: JSONL file where each line has `input`, `expected`, and optional `id` fields.
- **Checker**: boolean expression evaluated per case with access to `input`, `output`, `expected`.
- **Metric**: fraction of cases where a checker passes (`count(passed) / count(total)`).
- **Score**: the optimization objective (can be a metric name or arithmetic expression over metrics).
- **Split**: fraction of dataset allocated to train/val/test. Train is used for per-generation batch evaluation. Val is the rotating evaluation set. Test is held out for final evaluation.
- **Tune**: declares hyperparameter search space (grid swept before evolutionary search).
- **Topology**: constrains structural mutations — which mutation types are allowed, max graph size, which steps must be preserved.

---

## 3. The Intermediate Representation (IR)

The DSL is parsed and lowered to a **spanless, JSON-serializable IR**. Key structs:

```
ScaffoldIR { types, nodes, graphs, objectives }
NodeIR     { name, kind, input: TypeIR, output: TypeIR, config: NodeConfigIR }
GraphIR    { name, input: TypeIR, output: TypeIR, body: Vec<GraphStmtIR> }
StepIR     { name, node, args }
IfIR       { cond, then_body, else_body }
LoopIR     { max, while_cond, body }
```

The IR is the authoritative representation during optimization. Mutations operate on the IR. The optimizer stores candidates as `(GraphIR, overrides)` pairs. At the end, the best candidate's IR is materialized back to `.scaffold` source via `pretty_print()`.

Two key functions bridge DSL source and IR:
- `parse_and_lower(source: &str) -> Result<ScaffoldIR>` — parse `.scaffold` source and lower to IR.
- `pretty_print(ir: &ScaffoldIR) -> String` — render IR back to `.scaffold` source. Also available per-component: `pretty_print_graph()`, `pretty_print_node()`.

---

## 4. The Optimization Loop

### 4.1 Three Phases

**Phase 1: Seed.** Evaluate the original graph on the validation set. This establishes a baseline score.

**Phase 2: Tunable Sweep.** Enumerate all combinations of `tune { ... }` parameter values (e.g., temperature ∈ [0.0, 0.2, 0.5]). Evaluate each. No budget cap — exhaustive grid search. *Skipped when meta-agent is active* (the LLM handles config exploration).

**Phase 3: Evolutionary Search.** Budget-capped mutation loop:

```
while generations < max_generations:
    parent = select_parent(archive)         # sigmoid + novelty weighting
    mutation = propose_mutation(parent)      # meta-agent LLM or random
    new_graph = apply_mutation(parent.graph, mutation)
    if violates_topology(new_graph): continue
    candidate = Candidate { graph: new_graph, overrides: merge(parent.overrides, mutation.overrides) }
    score = evaluate(candidate, val_dataset)
    archive.add(candidate, score)
```

After the budget is exhausted, the best candidate is evaluated on the held-out test set and written as a standalone `.scaffold` file.

### 4.2 Candidate Representation

```rust
struct Candidate {
    id: usize,
    parent_id: Option<usize>,
    graph: GraphIR,                     // The (possibly mutated) graph
    overrides: HashMap<String, Value>,  // Field-level overrides: "node.field" → value
    mutations: Vec<Mutation>,           // Mutations applied from parent
    score: Option<f64>,
    passed_case_ids: Vec<String>,       // Which cases pass (for diff analysis)
    case_results: Vec<CaseResult>,      // Per-case detail (for meta-agent context)
    // ...
}
```

**The override mechanism**: Mutations can either modify the graph IR directly (structural) or store field-level overrides that are applied at execution time. For example, `rewrite_prompt("classify_text", new_template)` stores `"classify_text.template" → "new text"` in the overrides map. When the executor runs `classify_text`, it checks overrides and uses the rewritten template instead of the original.

**Synthetic nodes**: The `add_prompt_step` mutation creates a new node that doesn't exist in the original IR. It stores the node definition as `"_node.review_step" → { kind: "prompt" }` in overrides, and the executor materializes it at runtime.

### 4.3 Parent Selection

Sigmoid-weighted sampling with novelty bonus:

```
midpoint = average(top_3_scores)
for each evaluated candidate c:
    weight = sigmoid(10 * (c.score - midpoint))    # exploit: favor high scores
    weight *= 1 / (1 + c.children_count)           # explore: penalize overused parents
sample from categorical(weights)
```

The sigmoid curve sharply separates above-midpoint from below-midpoint candidates. The novelty term prevents the optimizer from repeatedly mutating the same high-scoring parent (diminishing returns).

*When meta-agent is active*: novelty is disabled because the LLM already reasons about diversity from the archive context.

### 4.4 Evaluation

Each candidate is evaluated by running the graph on every case in the dataset:

1. Build modified IR: replace the objective's graph with the candidate's graph.
2. Create executor with candidate's overrides.
3. Run each case through the graph concurrently (`buffer_unordered(concurrency)`).
4. For each case: evaluate all checkers, record pass/fail, capture step traces and model responses.
5. Compute per-metric scores (fraction of cases passing each checker).
6. Compute final score from the score expression.

**Step tracing**: The executor records `(step_name, truncated_output)` for every step in the graph. This trace is stored in `CaseResult` and shown to the meta-agent so it can diagnose *where* in the pipeline a case fails.

**Early stopping**: Two tiers:
- *Absolute*: if 0 cases pass after the first 10–15, abort (likely a broken template or syntax error).
- *Relative*: if pass rate < 40% of parent's score after 10–15 cases, abort (hopeless mutation — saves eval budget).

### 4.5 Train Batch Sampling

Each generation also evaluates a larger train batch (stratified by domain) to give the meta-agent more failure signal. The train batch is re-sampled each generation.

**Stratified sampling**: Case IDs are prefixed by domain (e.g., `s2d_train-015`, `law_train-042`, `uspto_train-003`). The sampler groups by prefix, allocates at least 1 case per domain, then distributes the remainder proportionally. This prevents random sampling from missing small domains entirely.

---

## 5. The Meta-Agent

When `--meta-model <model>` is passed, the optimizer uses an LLM (the "meta-agent") to propose mutations instead of random search.

### 5.1 Context

The meta-agent receives a structured text context containing:

1. **Objective summary** — checkers, metrics, allowed mutations, tunables.
2. **Parent graph** — pretty-printed `.scaffold` source of the graph being mutated, plus the parent's score.
3. **Lineage** — causal chain from seed to parent, showing each mutation and its score delta.
4. **Archive top-10** — ranked candidates with scores, mutations, and child counts.
5. **Pass/fail matrix** — cases (rows) × candidates (columns), showing P/F. Lets the meta-agent see which cases flip between candidates.
6. **Mutation effects** — grouped by mutation type+target (e.g., `rewrite_prompt(classify_text)`): number of attempts, best score, recent deltas, saturation status.
7. **Failure decomposition** — categorizes failures (output format, runtime errors, algorithm logic, etc.) and counts.
8. **Output pattern analysis** — groups recurring wrong outputs (e.g., "7 cases all predicted 'Protections' when expected Oxidations, Reductions, ..."). Surfaces systematic biases.
9. **Repair effectiveness** — for retry/loop graphs: how often the repair/retry step actually helps (NO-OP, SAME-ERROR, SHIFTED-ERROR tags).
10. **Parent failures** — detailed view of top-5 failed cases with input, expected, actual output, step trace, and model response.
11. **Available nodes** — pretty-printed node definitions (with overridden values if applicable).
12. **DSL syntax reference** — condensed grammar and structural patterns (only when `edit_graph` is available).

### 5.2 Mutation Types

| Mutation | What It Does | Stored As |
|----------|-------------|-----------|
| `rewrite_prompt(node)` | Replace a node's Jinja2 template | Override: `"node.template" → new_text` |
| `rewrite_system(node)` | Replace a node's system prompt | Override: `"node.system" → new_text` |
| `rewrite_shell(node)` | Replace a tool node's shell command | Override: `"node.shell" → new_cmd` |
| `set_config(node, field, value)` | Change a node config value (temperature, model, etc.) | Override: `"node.field" → value` |
| `add_prompt_step(after, name, template)` | Insert a new prompt node as a step | Override: `"_node.name" → {kind}` + field overrides; graph rewired |
| `insert_step(after, node)` | Insert a step calling an existing node | Graph IR modified directly |
| `remove_step(step)` | Remove a step from the graph | Graph IR modified directly |
| `replace_component(step, new_node)` | Swap which node a step calls | Graph IR modified directly |
| `insert_verify(after, verify_node, retries)` | Add a verify gate with retry loop | Graph IR modified directly |
| `wrap_retry(step, verify_node, retries)` | Wrap a step in a verify+retry loop | Graph IR modified directly |
| `edit_graph(graph_source, new_nodes)` | Rewrite the entire graph topology | Graph IR replaced from parsed source |

**Content mutations** (`rewrite_prompt`, `rewrite_system`, `rewrite_shell`, `set_config`) change a single node's behavior without touching the graph structure. They are always allowed regardless of the `topology.mutations` list.

**Structural mutations** (`insert_step`, `remove_step`, `replace_component`, `insert_verify`, `wrap_retry`, `add_prompt_step`) modify the graph IR directly.

**`edit_graph`** is a special "super-mutation" that subsumes all structural mutations. Instead of applying a single atomic structural change, the meta-agent writes the complete modified graph as `.scaffold` DSL source. This is parsed, lowered to IR, and validated against constraints.

### 5.3 The `edit_graph` Mutation in Detail

When the meta-agent proposes `edit_graph`, it provides:
- `graph`: the complete graph body as `.scaffold` source
- `new_nodes` (optional): array of new or modified node definitions as `.scaffold` source
- `description`: a short label for the TUI

**Validation pipeline** (`parse_and_validate_graph_edit`):

1. **Build combined source**: prepend type definitions (so the parser can resolve named types) + new node sources + graph source.
2. **Parse + lower**: call `scaffold_ir::parse_and_lower()` on the combined source. If parsing fails, the proposal is rejected.
3. **Graph name match**: the new graph must have the same name as the parent.
4. **Type match**: input and output types must match the parent (compared via `format_type()` string equality).
5. **Preserved steps**: any steps listed in `topology.preserve` must still exist in the new graph (checked recursively through if/loop/parallel bodies).
6. **Node reference validation**: every `step x = node(...)` must reference a node that either exists in the original IR or was declared in `new_nodes`.
7. **Topology constraints**: `max_nodes`, `max_depth` (if declared).

**How new nodes work in `edit_graph`**: New node definitions provided in `new_nodes` are parsed alongside the graph. Nodes that don't exist in the original IR are extracted and stored as synthetic nodes. At execution time, the executor resolves step references by checking both the original IR's nodes and the synthetic nodes.

### 5.4 Retry Loop

The meta-agent has up to 10 retries per generation. If a proposal fails validation (parse error, type mismatch, unknown node reference, topology violation), the error message is appended to the context and the LLM tries again. Duplicate proposals (same mutation label) are also rejected.

---

## 6. Execution

The `GraphExecutor` walks the graph statement-by-statement:

1. **Step**: Resolve the node (check IR nodes, then synthetic nodes). Collect overrides for this node (`get_node_overrides("node_name")` extracts all `"node_name.*"` keys from the overrides map). Pass to `node_runner::run_node()`.
2. **If/Else**: Evaluate condition, execute the matching branch in a child scope.
3. **Loop**: Evaluate `while` condition each iteration. `carry` statements update the scope for the next iteration.
4. **Parallel**: Iterate over collection, execute body for each item. Optionally reduce results via a node.
5. **Emit**: Evaluate expression, return as graph output.

**Override application at runtime**: When executing a step for node `classify_text`, the executor calls `get_node_overrides("classify_text")` which filters the overrides map for keys like `"classify_text.template"`, `"classify_text.system"`, `"classify_text.temperature"`, etc. The `node_runner` applies these over the node's original config.

**Shell sandboxing**: Shell commands in tool nodes run sandboxed via macOS `sandbox-exec` when the command originates from a meta-agent override (untrusted). User-authored `.scaffold` shell commands run unsandboxed. The sandbox profile blocks network access and restricts file access to the current working directory.

---

## 7. Worked Example: Text Classification

### The Problem

Classify text inputs into categories (medical symptoms → diseases, legal charges, chemical reaction types) using an LLM. The graph is simple: one prompt node, one step, one emit.

### The `.scaffold` File

```
type ClassifyInput = { task_text: string, label_guide: string }

node classify_text: prompt {
    in: { task_text: string, label_guide: string }
    out: string
    template: file("prompts/classify/template")
    system: file("prompts/classify/system")
    model: "openai/gpt-oss-120b"
    temperature: 0.0
}

node retry_classify: prompt {
    in: { task_text: string, label_guide: string, previous_prediction: string }
    out: string
    template: file("prompts/classify_retry/template")
    system: file("prompts/classify/system")
    model: "openai/gpt-oss-120b"
    temperature: 0.0
}

graph classify {
    in: ClassifyInput
    out: string
    step result = classify_text(task_text: input.task_text, label_guide: input.label_guide)
    emit result
}

objective text_classify {
    graph: classify
    dataset: file("datasets/symptom2disease.jsonl")
    checker correct { output == expected.label }
    metric accuracy { checker: correct }
    score: accuracy
    split { train: 0.28, val: 0.17, test: 0.55 }
    tune { classify_text.temperature in [0.0, 0.2, 0.5, 0.7, 1.0] }
    topology { mutations: [set_config, insert_step, remove_step, replace_component] max_nodes: 8 target_score: 0.9 }
}
```

### What the Optimizer Does

1. **Seed**: Run the original graph on the val set. Score: 0.48.
2. **Tunable sweep**: Try temperature 0.0, 0.2, 0.5, 0.7, 1.0. Best stays 0.0.
3. **Evolutionary search** (with meta-agent):
   - Generation 1: meta-agent proposes `edit_graph` — adds domain detection tool + fuzzy matching. Score drops to 0.44.
   - Generation 2: meta-agent proposes `edit_graph` — decomposed pipeline with domain-specific classification. Score drops to 0.36.
   - Generation 3: meta-agent proposes `rewrite_prompt(classify_text)` — better instructions, few-shot examples. Score rises to 0.56.
   - Generation 4: `rewrite_prompt(retry_classify)` — 0.52.
   - Generations 5–8: more `edit_graph` attempts. All score worse than generation 3.
4. **Test eval**: Best candidate (#3) evaluated on 288 test cases. Final score: ~50%.

### Why `edit_graph` Kept Failing

The meta-agent proposed increasingly complex topologies:
- Domain detection via Python tool nodes (shell quoting bugs broke execution)
- Multi-step pipelines (fuzzy match → classify → refine → verify) that added failure modes
- Self-consistency voting via parallel fan-out (each branch could independently fail)

Meanwhile, the simple `rewrite_prompt` — changing the instructions in the classify prompt — improved the score more than any structural change. The core problem was that the LLM model (`gpt-oss-120b`) didn't have enough domain knowledge to distinguish fine-grained chemistry reaction types, and no amount of graph topology can fix a model capability gap.

---

## 8. Current Limitations and Open Problems

### 8.1 `edit_graph` Quality

The `edit_graph` mutation asks the meta-agent to write complete `.scaffold` DSL source. Current problems:

- **Shell quoting in tool nodes**: When the meta-agent creates tool nodes with `shell: "python3 -c '...'"`, it frequently generates broken quoting (nested single quotes, unescaped special characters). The shell command fails at runtime with "unexpected EOF" errors.

- **Complexity bias**: The meta-agent tends to propose maximally complex graphs (multiple new nodes, nested control flow, parallel blocks) when simpler changes would be more effective. There's no mechanism to penalize graph complexity in the score or parent selection.

- **No incremental structural changes**: `edit_graph` is all-or-nothing — the meta-agent must rewrite the entire graph. A small structural change (adding one step) requires regenerating all existing steps correctly. This amplifies error probability.

- **Validation is necessary but not sufficient**: The validation checks types, preserved steps, node references, and topology constraints. But it can't check whether the *semantics* make sense — e.g., whether the arguments passed to a new tool node will actually work at runtime, or whether a loop's while-condition will ever terminate.

### 8.2 Evaluation Efficiency

- **No caching**: If a candidate changes only one node's prompt, all cases are re-evaluated from scratch. Cases that don't touch the changed node produce identical results. Case-level caching based on which nodes changed could save significant eval budget.

- **Limited budget**: Typical runs evaluate 8–15 candidates. With ~50% of structural mutations failing, effective search is only 4–7 productive candidates. Not enough to explore the search space.

### 8.3 Score Signal

- **Exact match only**: The checker `output == expected.label` is binary. A prediction of "Oxidation" when the answer is "Oxidations" scores 0. Fuzzy matching in the checker would give partial credit and smoother gradients for the optimizer.

- **Single score**: All domains (medical, legal, chemistry) contribute equally to one accuracy number. The optimizer can't specialize: improving chemistry at the cost of medical is invisible if the net accuracy stays flat.

### 8.4 Meta-Agent Context

- **No domain decomposition signal**: The meta-agent sees individual failures but not domain-level accuracy. Added "Output Pattern Analysis" helps, but the meta-agent still can't see "s2d=70%, lawbench=55%, uspto=30%".

- **Context size**: The full context (objective + archive + pass/fail matrix + mutation effects + failure detail + node definitions + DSL reference) can reach 30k+ characters. This may hit the meta-agent model's effective attention limits.
