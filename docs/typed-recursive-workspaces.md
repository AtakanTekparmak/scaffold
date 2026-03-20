# Scaffold Typed Recursive Workspaces

Status: proposed direction

Companion docs:

- [task-harness-v1.md](/Users/kayaomers/Documents/firstbatch/scaffold/docs/task-harness-v1.md)
- [task-harness-v1-grammar.md](/Users/kayaomers/Documents/firstbatch/scaffold/docs/task-harness-v1-grammar.md)

This document proposes how Scaffold should absorb the strongest ideas from recursive REPL-based agents without giving up its core promise of typed, verifiable orchestration.

The immediate motivation is clear:

- plain chat-history agents suffer from context rot
- recursive subcalls are often the right answer
- persistent working memory is useful
- but arbitrary Python REPL execution should not become the semantic core of Scaffold

The goal is to get the benefits of recursive language models while keeping Scaffold grounded in:

- typed state
- explicit dataflow
- bounded execution
- verifiable wiring
- optimization over declared policy surfaces

This document assumes the stronger Scaffold goal:

- let an LLM synthesize a harness specialized for the task and the model
- allow prompts, tools, memory policy, and orchestration structure to evolve
- but require every evolved harness to remain inside a typed, verifiable search space

## Thesis

Scaffold should support recursive agentic work over persistent state, but that state should be modeled as typed workspaces rather than untyped REPL locals.

In other words:

- do not make Python the language of truth
- do make recursive workspaces a first-class runtime concept

More concretely:

- the objective and invariants stay fixed
- the harness may evolve aggressively
- recursive workspaces should be part of that searchable harness surface

## What We Want From RLM Systems

The useful ideas from recursive REPL agents are:

1. Persistent working state

The model should not need to repack the entire world into every prompt. It should be able to read and update a durable stateful workspace.

2. Recursive subtasking

The model should be able to hand off a bounded subproblem to another model call or another task.

3. Depth-aware routing

Root reasoning and deeper recursive calls often want different models, temperatures, and budgets.

4. Versioned context and history

State snapshots and previous reasoning traces should be inspectable and reusable.

5. Runtime traces for debugging and optimization

If the system makes recursive calls, we need to see that structure in telemetry and score it in objectives.

## What We Do Not Want

We should not import the following as Scaffold semantics:

1. Arbitrary host-language execution as the default agent primitive

That breaks the compile-time story and makes behavior hard to verify.

2. Prompt text as the only interface contract

Tools, subtasks, and state should be declared in the language, not merely described in prose.

3. Free-floating local variables as hidden state

The optimizer and verifier need named, typed state surfaces.

4. Unbounded recursion or hidden side effects

Recursive execution needs depth, cost, and mutation boundaries.

## Core Proposal

Add a first-class `workspace` concept to Scaffold.

A workspace is a typed, versioned state container that tasks and subtasks can read and update through explicit operations.

The workspace becomes the structured alternative to a REPL namespace.

## New Concepts

### `workspace`

A workspace declaration defines the durable state shape for recursive work.

```scaffold
workspace ResearchSession {
    slots {
        corpus: list<string>
        chunk_summaries: list<string>
        findings: list<string>
        open_questions: list<string>
        draft_answer: string
    }
}
```

Properties:

- every slot is typed
- every mutation is explicit
- snapshots are versioned
- telemetry can attribute reads and writes

### `subtask`

A subtask is a typed recursive call to another task or agentic unit.

```scaffold
subtask summarize_chunk {
    input: {
        query: string,
        chunk: string
    }
    output: {
        summary: string,
        evidence: list<string>
    }
}
```

In practice, this may lower to an internal task call, a prompt call, or a runtime recursive execution node, but at the language level it stays typed.

### `memory_policy`

The harness should control how workspace state is managed as context pressure grows.

```scaffold
memory_policy default_memory {
    snapshot_each_turn: true
    summarize_when {
        slot: chunk_summaries
        over_items: 50
        using: prompt summarize_summaries
        into: chunk_summaries
    }
    retain_last {
        slot: open_questions
        items: 20
    }
}
```

This is the typed, optimizable replacement for ad hoc context compression inside prompt logic.

## Searchable Structure, Not Just Knobs

Recursive workspaces only matter for Scaffold if they can participate in harness synthesis.

That means the optimizer should be able to search over choices like:

- whether to recurse at all
- how to decompose into subtasks
- which model to use at each depth
- when to batch subcalls
- when to summarize workspace state
- when to snapshot or discard old state
- whether a critique/revision branch exists
- which tools are available to recursive stages

These are not just scalar hyperparameters. They are structured harness decisions.

So the right mental model is:

- task = fixed contract
- harness = legal strategy program
- workspace/memory policy = part of that strategy program

## Structural Mutation Surfaces

Scaffold should allow structure to evolve only through declared surfaces.

Examples:

```scaffold
tune {
    planner.enabled in [true, false]
    dispatch.strategy in ["single", "batched", "adaptive"]
    runtime.depth_policy in ["flat", "recursive", "adaptive"]
    session.memory_policy in ["none", "rolling_summary", "hierarchical"]
    revise.tools subset_of [web_search, calculator, retrieve_notes]
}
```

This is the crucial difference from unconstrained host-language evolution:

- the system may discover structure
- but only among legal, typed alternatives

## Task Extensions

Tasks should be able to declare and operate over workspaces.

```scaffold
task answer_with_workspace {
    input: {
        question: string,
        docs: list<string>
    }
    output: {
        answer: string
    }

    workspace session: ResearchSession

    init session {
        corpus: input.docs
        chunk_summaries: []
        findings: []
        open_questions: [input.question]
        draft_answer: ""
    }

    loop investigate {
        max_iters: 8
        while: len(session.open_questions) > 0

        stage pick_question using tool pop_question {
            in: {
                questions: session.open_questions
            }
            out: next_question
        }

        stage dispatch using subtask summarize_chunk {
            foreach: session.corpus
            in: {
                query: next_question.question,
                chunk: item
            }
            collect: chunk_results
        }

        update session {
            chunk_summaries += chunk_results[*].summary
            findings += flatten(chunk_results[*].evidence)
            draft_answer = synthesize(session.findings, input.question)
        }
    }

    emit {
        answer: session.draft_answer
    }
}
```

The syntax above is illustrative rather than final, but the semantics matter:

- workspaces are task-owned state
- state transitions are explicit
- subtasks are typed
- recursive work is visible to telemetry and optimization

## Harness Extensions

Recursive workspaces need harness-level policy controls.

```scaffold
harness research_search for task answer_with_workspace {
    defaults {
        model: "gpt-4o-mini"
        query_model: "gpt-4o-mini"
        recursion_model: "gpt-4o-mini"
        max_depth: 2
        max_turns: 8
    }

    bind dispatch {
        model: "gpt-4o-mini"
        temperature: 0.0
    }

    bind investigate {
        max_iters: 8
    }

    bind session {
        memory_policy: "default_memory"
    }

    tune {
        dispatch.model in ["gpt-4o-mini", "gpt-5-mini"]
        dispatch.temperature in [0.0, 0.2]
        dispatch.strategy in ["single", "batched", "adaptive"]
        investigate.max_iters in [4, 8, 12]
        session.memory_policy in ["default_memory", "aggressive_summary"]
        runtime.max_depth in [1, 2, 3]
    }
}
```

The important shift is that recursion, summarization, batching, and compression become typed harness policy surfaces.

## Mapping From REPL Concepts To Scaffold Concepts

| REPL/RLM idea | Scaffold equivalent |
|---|---|
| `context` | workspace slot or task input |
| `context_0`, `context_1` | workspace snapshots / versioned state |
| `history_0`, `history_1` | transcript artifacts |
| `llm_query()` | subtask or prompt stage |
| `spawn_<name>()` | typed subtask invocation |
| depth-based model routing | harness runtime policy |
| REPL locals | explicit workspace slots |
| final-answer marker | typed task `emit` |

This mapping preserves the useful behavior while removing the hidden semantics.

## Tool Evolution

Tools should be evolvable, but only through declared surfaces.

Allowed evolution surfaces:

- tool selection and subsets
- routing order
- retry policy
- timeout policy
- tool instruction text
- tool implementation variant selection
- tool-enabled vs tool-disabled structural branches

High-risk surfaces that should be separate modes:

- arbitrary host-language code rewriting
- undeclared side effects
- mutation of external services without versioning

If Scaffold supports evolving tools, it should do so through:

- versioned tool variants
- typed inputs and outputs
- explicit capability declarations
- replayable traces

That keeps the optimizer honest.

## Verification Implications

Typed recursive workspaces are only worth adding if they preserve structural guarantees.

The verifier should prove:

- every workspace slot has a declared type
- updates assign values compatible with that type
- recursive calls are bounded by depth and/or budget
- subtasks consume and produce declared schemas
- snapshot and memory-policy references resolve
- no task reads a workspace slot before initialization

The verifier cannot prove reasoning quality, but it can prove that the recursive machinery is structurally sound.

## Runtime Implications

The runtime will need:

1. Workspace state manager

- typed slots
- versioned snapshots
- read/write audit trail

2. Recursive execution engine

- subtask invocation
- depth accounting
- budget accounting
- model routing by depth/policy

3. Memory-policy executor

- summarization hooks
- retention rules
- snapshot rules

4. Recursive telemetry

- task tree
- per-depth model usage
- workspace mutations
- snapshot lineage

## Objective Implications

Recursive workspaces create new meaningful objective signals:

- workspace growth rate
- summary compression ratio
- recursive call count
- depth usage
- answer quality under bounded memory
- stability across repeated runs

This is useful because context-rot mitigation should be evaluated, not merely assumed.

## Suggested Implementation Order

1. Add materialization for optimized harnesses

- `optimize --write-best`
- `optimize --write-prompts`

2. Add a minimal `workspace` IR and runtime model

- typed slots
- initialization
- explicit updates
- telemetry

3. Add typed `subtask` execution

- bounded recursive calls
- per-depth routing

4. Add `memory_policy`

- summarization
- snapshotting
- retention

5. Add optimizer support over recursive-workspace policies

- model-by-depth
- summary policies
- recursion depth
- batching choices

This order keeps the current task/harness architecture intact while growing toward the RLM-inspired direction.

## Recommendation

Scaffold should adopt recursive workspaces as a first-class concept.

It should not adopt arbitrary Python REPL semantics as the core language.

The strongest synthesis is:

- typed task graph
- typed persistent workspace
- typed recursive subtasks
- harness-controlled memory policy
- optimization over those policies and structural alternatives

That gives Scaffold a principled answer to context rot while staying faithful to its compile-time-verifiable objective.
