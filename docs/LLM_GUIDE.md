# Scaffold v2 — Language Reference for LLMs

This document is the authoritative reference for generating correct `.scaffold` programs. Follow these rules exactly.

## Program Structure

A scaffold program is a sequence of four declaration kinds, in order:

1. `type` — named type aliases
2. `node` — typed LLM/tool components
3. `graph` — computation workflows composed of nodes
4. `objective` — evaluation and optimization specifications

Comments: `// line comment` and `/* block comment */`.

---

## 1. Types

```
type <Name> = <TypeExpr>
```

**Primitive types:** `bool`, `int`, `float`, `string`, `bytes`, `any`

**Container types:**
- `list<T>` — homogeneous list
- `map<K, V>` — key-value map
- `option<T>` — nullable value

**Struct type (inline):**
```
{ field1: Type1, field2: Type2 }
```

**Named reference:** Any previously declared type name.

**Examples:**
```
type MemoryFact = { source_session_id: string, date: string, fact: string }
type MemoryStore = { facts: list<MemoryFact> }
type ExerciseInput = {
    instructions: string,
    stub_filename: string,
    stub_content: string,
    language: string,
    exercise_dir: string
}
```

---

## 2. Nodes

```
node <name>: <kind> {
    in: <TypeExpr>
    out: <TypeExpr>
    <config fields...>
}
```

### Node Kinds

| Kind | Purpose |
|------|---------|
| `prompt` | LLM call with template |
| `tool` | Shell command execution |
| `agent` | Multi-turn LLM + tools loop |
| `verify` | LLM-based verification |

### Config Fields

| Field | Accepts | Applicable to | Description |
|-------|---------|---------------|-------------|
| `template` | `"text"` or `file("path")` | prompt, agent, verify | Jinja2 template for the prompt |
| `system` | `"text"` or `file("path")` | prompt, agent, verify | System prompt |
| `model` | `"model-id"` | prompt, agent, verify | LLM model (e.g. `"gpt-4o"`) |
| `temperature` | float | prompt, agent, verify | Sampling temperature |
| `max_tokens` | int | prompt, agent, verify | Max output tokens |
| `max_turns` | int | agent | Max conversation turns |
| `tools` | `[name1, name2]` | agent | Available tool nodes |
| `shell` | `"command"` | tool | Shell command with `{{var}}` interpolation |
| `timeout` | int | tool | Timeout in seconds |
| `on_error` | `abort` or `retry(N)` | all | Error strategy |
| `json` | `{ "key": expr }` | prompt, verify | Structured output schema |

### Template Syntax

Templates use Jinja2 syntax. Variables from the node's input type are available:

```
template: "Solve: {{ question }}"
template: file("examples/prompts/solver/template.jinja")
```

For struct inputs like `{ task: Question, memory: Store }`, access fields as `{{ task.question }}`, `{{ memory.facts }}`.

Loops and raw blocks:
```
{% for fact in memory.facts %}
- {{ fact.date }} | {{ fact.fact }}
{% endfor %}

{% raw %}
{"answer": "...", "confidence": 0.9}
{% endraw %}
```

### Tool Shell Commands

Shell templates use `{{var}}` for interpolation. The command runs in bash.

```
node eval_exercise: tool {
    in: { code: string, exercise_dir: string, language: string }
    out: string
    shell: "python3 tools/eval.py --dir \"{{exercise_dir}}\" <<'EOF'\n{{code}}\nEOF"
    timeout: 120
}
```

### Examples

```
node solver: prompt {
    in: { question: string }
    out: { answer: string, confidence: float }
    model: "gpt-4o"
    template: file("prompts/solver/template.jinja")
    system: file("prompts/solver/system.jinja")
    temperature: 0.5
}

node eval_code: tool {
    in: { code: string, test_dir: string }
    out: string
    shell: "python3 tools/eval.py --dir {{test_dir}}"
    timeout: 60
}

node reviewer: verify {
    in: { draft: string, expected: string }
    out: { pass: bool, reason: string }
    model: "gpt-4o-mini"
    template: "Is this correct? Draft: {{ draft }} Expected: {{ expected }}"
}
```

---

## 3. Graphs

```
graph <name> {
    in: <TypeExpr>
    out: <TypeExpr>
    <statements...>
}
```

Graphs wire nodes together via typed steps. They must emit a value matching the `out` type.

### Statements

#### Step — call a node or subgraph

```
step <var> = <node_or_graph>(<args>)
```

Positional argument (passes entire value as input):
```
step memory = extract_memory(input)
```

Named arguments (builds a struct):
```
step retrieved = retrieve_memory(task: input, memory: memory)
```

Field access on prior steps:
```
step result = eval(code: attempt1, dir: input.exercise_dir)
```

#### Emit — return a value from the graph

```
emit <expr>
emit { field1: expr1, field2: expr2 }
```

Every execution path must emit a value matching the graph's `out` type.

#### If — conditional branching

```
if <condition> {
    <statements...>
} else {
    <statements...>
}
```

The `else` block is optional. Steps from the parent scope are visible inside branches. Each branch that needs to return a value must `emit`.

```
if json_parse(result1).passed == false {
    step attempt2 = fix_code(previous_code: attempt1, test_output: json_parse(result1).test_output)
    step result2 = eval_exercise(code: attempt2)
    emit result2
} else {
    emit result1
}
```

#### Loop — bounded iteration

```
loop (max: <expr>, while: <expr>) {
    <statements...>
}
```

Repeats up to `max` times while condition is true. Use `carry` to update state between iterations.

```
loop (max: 3, while: !verified.pass) {
    step draft = solver(input)
    step verified = checker(answer: draft.answer, expected: input.expected)
    carry best = draft
}
```

#### Carry — update a variable in a loop

```
carry <var> = <expr>
```

Only valid inside `loop` bodies. Persists the value for the next iteration.

#### Parallel — fan-out over a collection

```
parallel (<var> in <collection_expr>, reduce: <node>) {
    <statements...>
}
```

Executes body for each element. The optional `reduce` node receives the collected results.

#### Choose — select among alternatives

```
choose [alternative1, alternative2, alternative3]
```

Selects one alternative to execute. Used for structural search during optimization.

### Subgraph Calls

A step can call another graph by name:

```
graph inner {
    in: Question
    out: Answer
    step s = solver(input)
    emit s
}

graph outer {
    in: Question
    out: Answer
    step result = inner(input)     // calls the inner graph
    emit result
}
```

### Expression Syntax

**Literals:** `42`, `3.14`, `"hello"`, `true`, `false`, `null`

**Variables:** `input`, `step_name`, `iteration`

**Field access:** `input.question`, `result.answer`, `output.facts`

**Index access:** `list[0]`, `map["key"]`

**Operators:**
- Arithmetic: `+`, `-`, `*`, `/`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&&`, `||`, `!`

**Built-in functions:**
- `len(x)` — length of string, list, or map
- `contains(haystack, needle)` — string/list containment
- `str(x)`, `int(x)`, `float(x)` — type conversions
- `lower(s)`, `upper(s)`, `trim(s)` — string operations
- `split(s, sep)`, `join(list, sep)` — split/join
- `keys(map)`, `values(map)` — map operations
- `json_parse(s)` — parse JSON string into a structured value

**Examples:**
```
output.answer == expected.answer
len(output.facts) > 0
json_parse(result).passed == true
output.abstain || len(output.evidence_session_ids) > 0
```

---

## 4. Objectives

```
objective <name> {
    graph: <graph_name>
    dataset: <dataset_spec>

    checker <name> { <bool_expr> }
    metric <name> { checker: <checker_name> }
    score: <metric_expr>

    split { train: <f>, val: <f>, test: <f> }
    select { primary: <metric> }
    tune { <tunable_decls> }
    topology { <topology_config> }
    sub <name> { <sub_objective> }
}
```

### Dataset

**File-based (JSONL):**
```
dataset: file("examples/datasets/data.jsonl")
```

Each line must be JSON with `input`, `expected`, and optional `id` fields:
```json
{"id": "case-1", "input": {"question": "..."}, "expected": {"answer": "Paris"}}
```

**Inline cases:**
```
dataset: cases [
    { input: "2+2", expected: "4" },
    { input: { x: 1 }, expected: { y: 2 }, id: "test-1" }
]
```

### Checkers

Boolean expressions evaluated per case. `output` is the graph result, `expected` is from the dataset.

```
checker exact_answer { output.answer == expected.answer }
checker non_empty { len(output.answer) > 0 }
checker tests_pass { json_parse(output).passed == true }
checker evidence_present { output.abstain || len(output.evidence_session_ids) > 0 }
```

### Metrics

A metric computes the fraction of cases where a checker returns true:

```
metric accuracy { checker: exact_answer }
metric pass_rate { checker: tests_pass }
```

### Score

The optimization target. Usually a single metric, but can be an expression:

```
score: accuracy
score: accuracy * 0.7 + coverage * 0.3
```

### Split

Dataset split ratios for train/val/test:

```
split { train: 0.7, val: 0.15, test: 0.15 }
```

### Select

Multi-objective selection:

```
select { primary: accuracy }
select { primary: accuracy, tie_breakers: [coverage, latency] }
```

### Tune

Declares hyperparameter search domains. Each path is `node.field`:

```
tune {
    solver.temperature in [0.0, 0.3, 0.5, 0.7, 1.0]
    solver.model in ["gpt-4o", "gpt-4o-mini"]
    fix_code.temperature in [0.0, 0.2, 0.5]
}
```

### Topology

Configures structural mutation rules for the evolutionary optimizer:

```
topology {
    mutations: [insert_verify, wrap_retry, set_config]
    max_nodes: 8
    target_score: 0.95
    preserve: [critical_step]
}
```

Available mutations:
- `insert_verify` — add a verify gate after a step
- `wrap_retry` — wrap a step in a retry loop with verification
- `insert_step` — add a new step
- `remove_step` — remove a step
- `replace_component` — swap a step's node
- `set_config` — change a tunable parameter (auto-added when `tune` is declared)

Content mutations are always available when the meta-agent is active:
- `rewrite_prompt` — rewrite a prompt node's template instructions
- `rewrite_system` — rewrite a node's system prompt
- `rewrite_shell` — rewrite a tool node's shell command

### Sub-objectives

Hierarchical optimization: optimize inner graphs before the full pipeline.

```
sub retrieval_opt {
    graph: extract_retrieve
    dataset: file("data.jsonl")

    checker has_facts { len(output.facts) > 0 }
    metric coverage { checker: has_facts }
    score: coverage

    tune {
        extract.temperature in [0.0, 0.3]
    }

    topology { mutations: [insert_verify] max_nodes: 5 target_score: 0.95 }
}
```

Sub-objectives have the same structure as the parent objective but **cannot nest further** (one level only). The optimizer runs sub-objectives first (in dependency order), freezes the best results, then optimizes the parent.

---

## Complete Example

```
// Types
type ExerciseInput = {
    instructions: string,
    stub_filename: string,
    stub_content: string,
    language: string,
    exercise_dir: string
}

// Nodes
node solve_code: prompt {
    in: { instructions: string, stub_filename: string, stub_content: string, language: string }
    out: string
    template: file("prompts/solve/template.jinja")
    system: file("prompts/solve/system.jinja")
    model: "gpt-4o"
    temperature: 0.2
}

node fix_code: prompt {
    in: { instructions: string, stub_filename: string, stub_content: string,
          language: string, previous_code: string, test_output: string }
    out: string
    template: file("prompts/fix/template.jinja")
    system: file("prompts/solve/system.jinja")
    model: "gpt-4o"
    temperature: 0.2
}

node eval_exercise: tool {
    in: { code: string, exercise_dir: string, language: string }
    out: string
    shell: "python3 tools/eval.py --dir \"{{exercise_dir}}\" --lang \"{{language}}\" <<'EOF'\n{{code}}\nEOF"
    timeout: 120
}

// Graph — two-attempt solve with retry on failure
graph solve {
    in: ExerciseInput
    out: string

    step attempt1 = solve_code(
        instructions: input.instructions,
        stub_filename: input.stub_filename,
        stub_content: input.stub_content,
        language: input.language
    )
    step result1 = eval_exercise(
        code: attempt1,
        exercise_dir: input.exercise_dir,
        language: input.language
    )

    if json_parse(result1).passed == false {
        step attempt2 = fix_code(
            instructions: input.instructions,
            stub_filename: input.stub_filename,
            stub_content: input.stub_content,
            language: input.language,
            previous_code: attempt1,
            test_output: json_parse(result1).test_output
        )
        step result2 = eval_exercise(
            code: attempt2,
            exercise_dir: input.exercise_dir,
            language: input.language
        )
        emit result2
    } else {
        emit result1
    }
}

// Objective
objective coding_benchmark {
    graph: solve
    dataset: file("datasets/exercises.jsonl")

    checker tests_pass { json_parse(output).passed == true }
    metric pass_rate { checker: tests_pass }
    score: pass_rate

    split { train: 0.7, val: 0.15, test: 0.15 }
    select { primary: pass_rate }

    tune {
        solve_code.temperature in [0.0, 0.2, 0.5, 0.7, 1.0]
        fix_code.temperature in [0.0, 0.2, 0.5]
    }

    topology { mutations: [set_config] max_nodes: 6 target_score: 0.9 }
}
```

---

## CLI Reference

```bash
scaffold check FILE                              # parse + type check + verify
scaffold compile FILE [-o output.json] [--compact] # lower to IR JSON
scaffold run FILE --graph NAME --input JSON       # execute a graph
scaffold evaluate FILE --objective NAME [--live]  # evaluate on dataset
scaffold optimize FILE --objective NAME [--live]  # optimize
    --max-candidates N          # evolutionary generations (default: 20)
    --meta-model MODEL          # LLM-guided meta-agent (e.g. gpt-4o)
    --concurrency N             # parallel case evaluation
    --report-dir DIR            # persist reports and best candidate
    --write-best FILE           # freeze best candidate IR
    --backend grid|evolutionary # optimization strategy
```

## Configuration

API keys via environment variables:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

Or `~/.scaffold/config.toml`:
```toml
default_model = "gpt-4o-mini"

[llm.openai]
api_key = "sk-..."

[llm.anthropic]
api_key = "sk-ant-..."
```

A `.env` file in the working directory is also loaded automatically.
