# Scaffold Task/Harness V1

Status: proposed direction

Companion grammar: [task-harness-v1-grammar.md](/Users/kayaomers/Documents/firstbatch/scaffold/docs/task-harness-v1-grammar.md)

This document defines a stronger V1 for Scaffold built around the user's core objective:

- harnesses should fail to compile when they are structurally miswired
- optimization should search over typed, bounded harness choices
- multi-agent systems should support explicit feedback flow, including sequential loops where error can accumulate

The design in this document intentionally shifts the semantic center of Scaffold away from "tool/prompt/agent/pipeline as the whole product" and toward "task/harness/objective as the product, with tools/prompts/agents as components inside it."

## Why Re-Center The Language

The current implementation already has strong pieces:

- typed declarations
- a parser, type checker, IR, runtime, codegen, and CLI
- runnable tools, prompts, agents, and pipelines
- an `optimize` command that evaluates a fixed program against a dataset

But the current implementation is not yet aligned with the main objective:

- verification is currently a no-op
- `optimize` evaluates a fixed harness but does not search over candidates
- agent reward/done signals are logged but do not drive execution
- harnesses, objectives, and feedback edges are not first-class DSL concepts

If the core goal is the language itself, the strongest path is to make those concepts first-class and let codegen be a later deployment path rather than the semantic foundation.

## Design Principles

1. Immutable task, mutable harness

The task defines the workflow graph, artifact flow, and structural correctness. The harness defines how that graph is instantiated for a run or optimization trial.

2. Typed artifacts, not free-floating text

Feedback between stages must move through named, typed artifacts. Free text is allowed inside those artifacts, but the edge itself must be declared.

3. Optimization only over declared search space

The optimizer may mutate only fields exposed in `harness.tune`. Nothing else is implicitly mutable.

4. Compile-time guarantees are structural

The compiler can prove that dataflow, bindings, loop-carried state, and search spaces are valid. It cannot prove task quality or model intelligence.

5. Execution first, codegen later

V1 should run through an interpreter/executor over typed IR. Codegen is still valuable, but should follow stable semantics rather than define them.

## Core Constructs

V1 introduces eight primary top-level constructs:

- `type`: reusable type definitions
- `artifact`: a typed declaration for payloads that participate in task flow
- `tool`: deterministic executable component
- `prompt`: single-call LLM component
- `agent`: multi-turn LLM component with tool access
- `task`: the typed orchestration graph
- `harness`: the mutable execution policy and search space for a task
- `objective`: the evaluation and optimization definition for a task

`tool`, `prompt`, and `agent` remain important, but they are now inputs to `task`, not the top-level product.

## Proposed V1 Syntax

### Reusable Types And Artifacts

```scaffold
type Source = {
    url: string,
    title: string
}

artifact ResearchNotes = {
    facts: list<string>,
    sources: list<Source>,
    coverage: float
}

artifact DraftAnswer = {
    answer: string,
    confidence: float
}

artifact Critique = {
    score: float,
    issues: list<string>,
    revision_brief: string
}
```

`artifact` is not a second type system. It is a typed declaration with extra semantic meaning for task flow, verification, and objectives.

In V1:

- every `artifact` has a normal schema and participates in the same type system as `type`
- an artifact name may be used anywhere a type name is valid
- the extra meaning is operational: tasks can declare artifact slots of that type, loops can carry those slots, and objectives can inspect them

You can think of `artifact ResearchNotes = { ... }` as "define a named schema and mark it as intended for task flow."

### Components

```scaffold
tool web_search {
    input: { query: string }
    output: ResearchNotes
    impl: foreign.search(query)
}

prompt write_draft {
    input: {
        question: string,
        notes: ResearchNotes
    }
    output: DraftAnswer
    template: file("prompts/write_draft.md")
}

agent critique_draft {
    input: DraftAnswer
    output: Critique
    tools: []
    system: file("prompts/critique.md")
    max_turns: 2
}

agent revise_draft {
    input: {
        draft: DraftAnswer,
        critique: Critique,
        notes: ResearchNotes
    }
    output: DraftAnswer
    tools: [web_search]
    system: file("prompts/revise.md")
    max_turns: 3
}
```

### Tasks

```scaffold
task answer_question {
    input: { question: string }
    output: { answer: string, sources: list<Source> }

    artifacts {
        notes: ResearchNotes
        draft: DraftAnswer
        critique: Critique
    }

    stage gather using tool web_search {
        in: { query: input.question }
        out: notes
    }

    stage write using prompt write_draft {
        in: {
            question: input.question,
            notes: notes
        }
        out: draft
    }

    loop refine {
        max_iters: 2
        carry: [draft, critique]
        until: critique.score >= 0.9 || draft.confidence >= 0.95

        stage review using agent critique_draft {
            in: draft
            out: critique
        }

        stage revise using agent revise_draft {
            in: {
                draft: draft,
                critique: critique,
                notes: notes
            }
            out: draft
        }
    }

    emit {
        answer: draft.answer,
        sources: notes.sources
    }
}
```

### Harnesses

```scaffold
harness answer_default for task answer_question {
    defaults {
        model: "openai/gpt-4.1-mini"
        temperature: 0.2
        timeout_secs: 45
        retries: 1
    }

    bind gather {
        timeout_secs: 10
    }

    bind write {
        model: "openai/gpt-4.1"
        system_prompt: variant("writer", "baseline")
    }

    bind review {
        model: "openai/gpt-4.1-mini"
        temperature: 0.0
        system_prompt: variant("critic", "strict")
    }

    bind revise {
        model: "openai/gpt-4.1"
        tools: [web_search]
    }

    tune {
        write.model in ["openai/gpt-4.1-mini", "openai/gpt-4.1"]
        write.temperature in [0.0, 0.2, 0.5]
        review.system_prompt in variants("critic")
        revise.tools subset_of [web_search]
        refine.max_iters in [1, 2, 3]
    }
}
```

### Objectives

```scaffold
objective answer_quality for task answer_question {
    dataset: file("datasets/answer_quality.jsonl")
    harness: answer_default

    repeats: 3

    constraint non_empty = len(output.answer) > 0
    checker exact = exact(output.answer, expected.answer)
    checker source_recall = set_recall(output.sources, expected.sources)
    judge groundedness = judge_grounded(output.answer, output.sources)
    metric latency = rollout.duration_ms
    metric token_cost = rollout.token_cost

    score =
        1.0 * exact +
        0.25 * source_recall -
        0.1 * groundedness -
        0.0001 * latency -
        0.00001 * token_cost

    split {
        train: 0.7
        val: 0.2
        test: 0.1
    }

    select {
        primary: score
        tie_breakers: [source_recall, -token_cost]
    }
}
```

## Semantic Model

### Task

A task is a typed orchestration graph with:

- declared input and output types
- a set of named artifact slots, each with a declared type
- a sequence or DAG of stages
- optional explicit loops
- a final `emit` mapping from in-scope artifacts to output

Tasks are immutable during optimization.

### Stage

A stage is a task-local node that invokes one component:

- `using tool ...`
- `using prompt ...`
- `using agent ...`

Each stage:

- consumes an input expression
- produces exactly one declared artifact slot whose type matches the stage output
- may be bound by a harness

### Loop

A loop is an explicit cyclic region. Cycles are illegal elsewhere.

A loop must declare:

- `max_iters`
- `carry` artifact slots that may be reassigned across iterations
- at least one termination condition: `while` for pre-checks, `until` for post-checks, or both

This makes optimization and verification tractable.

### Harness

A harness is a typed overlay on a task.

It may define:

- defaults shared across stages
- stage-specific bindings
- prompt variants
- model selection
- tool subsets
- execution policies such as timeout, retries, and loop bounds
- a search space over declared binding paths

The harness may not:

- create new stages
- change artifact types
- bypass undeclared edges
- mutate undeclared fields

### Objective

An objective turns task execution into optimization.

It defines:

- dataset and splits
- evaluation repeats for stochastic runs
- hard `constraint` signals
- executable `checker` signals
- soft `judge` signals
- derived `metric` signals over task output, intermediate traces, and rollout metadata
- a scalar score expression
- selection rules

This split is deliberate:

- `constraint` means hard pass/fail and should act as a gate
- `checker` means executable, replayable scoring logic
- `judge` means subjective or stochastic evaluation
- `metric` means a derived scalar built from the previous lanes plus rollout data

In V1, these declarations are still expressions, but those expressions may call existing `tool`, `prompt`, and `agent` components. That gives the language a practical bridge:

- complex programmatic verification can live in `tool` implementations and be invoked from `checker`
- LLM-backed evaluation can live in dedicated `prompt` or `agent` components and be invoked from `judge`
- the surface stays unified instead of introducing a second evaluator DSL too early

## What The Compiler Should Guarantee

The compiler should reject a program if any of the following are invalid:

- a stage references an unknown tool, prompt, or agent
- a stage input does not match the component input type
- a stage output is bound to an undeclared artifact slot
- a stage output type does not match the declared artifact slot type
- an artifact is used before it is produced
- multiple stages write the same artifact outside a declared loop region
- a loop carries an artifact that is not declared in `carry`
- a loop has no `max_iters` or no termination condition
- a harness binds an unknown task, stage, or field
- a `tune` entry points at a non-mutable field
- a `tune` domain is empty or not finite for V1
- an objective references unknown constraints, checkers, judges, metrics, rollout fields, or task outputs
- an objective targets a harness that is incompatible with the task
- a `constraint` does not evaluate to bool
- a `checker`, `judge`, or `metric` does not evaluate to bool or numeric

These checks are structural. They make "miswired harnesses do not compile" true in a meaningful, enforceable sense.

## What The Compiler Cannot Guarantee

The compiler cannot prove:

- that the chosen models are good
- that prompts are semantically effective
- that textual critique actually improves the task
- that a high score on one dataset generalizes

It also cannot turn a `judge` into a proof. Judge-based signals are useful, but they remain evaluation signals, not compile-time guarantees.

Those are optimization and evaluation questions, not type questions.

## Feedback Flow And Error Accumulation

The language should make feedback channels explicit.

Bad:

- free-text output from one agent shoved into another without a declared artifact
- hidden prompt coupling outside the harness
- implicit state passed through prompts without typed structure

Good:

- critique is a declared artifact
- revise consumes critique explicitly
- objectives can inspect critique score, loop count, and revision lineage
- telemetry can attribute degradation to a specific stage or loop

For V1, all multi-agent feedback should move through typed artifact slots. Text can be one field inside the artifact type, but the edge itself must be explicit and nameable.

## Optimization Model

`scaffold optimize` should become a real search procedure.

Core semantics:

- candidate = task + concrete harness instance
- seed harness = base candidate
- mutation space = only the fields declared under `tune`
- evaluation = repeated rollouts over train split
- selection = reject failed constraints first, then rank by objective score
- validation = periodic holdout check on val split
- final report = best train candidate plus val/test scores, lineage, and traces

V1 search algorithms should start simple:

- random search
- hill climbing
- evolutionary mutation with elitism

Textual optimization methods such as TextGrad or GEPA-style rewrites should be added later and should only operate on declared mutable text surfaces, such as:

- `system_prompt`
- prompt templates
- critique rubric text
- revision instructions

They should not rewrite arbitrary task structure in V1.

## Runtime And Telemetry Requirements

Optimization depends on richer traces than the current repo records.

Each rollout should capture:

- task name, harness name, objective name
- stage start and finish events
- per-agent turn counts
- tool call inputs and outputs
- prompt text hashes or variant ids
- model id, token usage, latency, retry count
- loop iteration counts
- intermediate artifact snapshots or hashes
- final output, score, and error class

This data should support both debugging and credit assignment.

## Proposed IR Shape

The existing IR should grow new top-level structures:

```rust
pub struct TaskIR {
    pub name: String,
    pub input: TypeIR,
    pub output: TypeIR,
    pub artifacts: Vec<ArtifactSlotIR>,
    pub body: Vec<TaskNodeIR>,
    pub emit: Vec<EmitFieldIR>,
}

pub struct HarnessIR {
    pub name: String,
    pub task: String,
    pub defaults: Vec<BindingIR>,
    pub stage_bindings: Vec<StageBindingIR>,
    pub tunables: Vec<TunableIR>,
}

pub struct ObjectiveIR {
    pub name: String,
    pub task: String,
    pub harness: String,
    pub dataset: DatasetSpecIR,
    pub repeats: u32,
    pub constraints: Vec<MetricIR>,
    pub checkers: Vec<MetricIR>,
    pub judges: Vec<MetricIR>,
    pub metrics: Vec<MetricIR>,
    pub score: ExprIR,
    pub split: SplitIR,
}
```

Task bodies should be explicit enough for verification:

```rust
pub enum TaskNodeIR {
    Stage(StageIR),
    Loop(LoopIR),
    Branch(BranchIR),
}
```

The important point is not the exact Rust struct layout. The important point is that task flow, harness overrides, and objective evaluation become part of IR rather than being encoded indirectly through generated Rust.

One additional semantic choice should stay explicit in IR: artifact declarations and type declarations share the same underlying type system. Artifact-ness is a role/tag, not a separate schema language.

## Recommended Execution Strategy

V1 should run through an interpreter over IR.

Why:

- optimization requires many candidate evaluations
- interpreter execution avoids full regenerate-and-rebuild cycles for every mutation
- structural semantics become easier to stabilize before codegen
- telemetry collection is more straightforward

Codegen should remain as a later path:

- freeze a task
- freeze a concrete harness instance
- optionally emit a deployable binary

## Phased Implementation Plan

### Phase 0: lock the semantics

Deliverable:

- this design doc
- agreement on syntax and guarantees

### Phase 1: syntax, AST, types, and IR

Crates:

- `crates/scaffold-syntax`
- `crates/scaffold-types`
- `crates/scaffold-ir`

Work:

- add tokens and parser support for `artifact`, `task`, `stage`, `loop`, `harness`, `objective`, `metric`, `emit`, `bind`, and `tune`
- extend AST with task/harness/objective nodes
- add type environment support for task-local artifact scopes, with artifacts sharing the same underlying type system as `type`
- add harness binding-path validation in the type layer
- lower to new IR structures

Success bar:

- `scaffold check` can parse and type-check the new constructs even before execution exists

### Phase 2: real verification

Crate:

- `crates/scaffold-verify`

Work:

- add definite assignment checks for artifacts
- enforce explicit loops for cycles
- validate loop-carried artifacts
- validate harness binding paths and tunable domains
- validate objective references

Success bar:

- intentionally miswired task/harness files fail at compile time

### Phase 3: interpreter execution

Primary crate:

- `crates/scaffold-runtime`

Possible clean split:

- add a new `crates/scaffold-engine` if execution logic becomes too large for `scaffold-runtime`

Work:

- execute `TaskIR` directly
- materialize a concrete harness instance for a run
- run tools/prompts/agents with stage-level bindings
- implement explicit loop execution and artifact store
- emit structured trace events

Success bar:

- `scaffold run file.scaffold --task X --harness Y --input ...` works without code generation

### Phase 4: optimizer

Primary surface:

- `crates/scaffold-cli`
- runtime or engine support for evaluation workers

Work:

- convert `optimize` from evaluator to search loop
- add candidate mutation over `tune` declarations
- add dataset splits, repeats, lineage, caching, and early stopping
- support train/val/test reporting

Success bar:

- `scaffold optimize` can produce an improved concrete harness instance

### Phase 5: codegen for frozen tasks

Crate:

- `crates/scaffold-codegen`

Work:

- generate deployable Rust from `TaskIR` plus a concrete harness instance
- keep codegen as an export path, not the execution authority

Success bar:

- optimized harnesses can be packaged for production without changing semantics

## Decisions To Lock Early

These decisions reduce churn later:

1. Harnesses mutate only declared fields.
2. Artifact flow is explicit and typed.
3. Artifacts are part of the same type system as `type`; they are not a parallel schema mechanism.
4. Loops must be explicit and bounded.
5. Objectives score runs, not source text.
6. Textual optimization may mutate only declared prompt surfaces.
7. Codegen follows interpreter semantics, not the other way around.

## Suggested Immediate Next Moves

If this direction is accepted, the next concrete work items should be:

1. add `docs/task-harness-v1.md`
2. define the exact grammar for `task`, `harness`, and `objective`
3. update AST and IR sketches before touching runtime execution
4. choose whether interpreter execution lives in `scaffold-runtime` or a new execution crate
5. implement verification before implementing evolutionary search

That order keeps the project aligned with the objective: a language that makes harness optimization possible because the structure is explicit, typed, and verifiable.
