# Topology Synthesis V2

Scaffold needs a reset in emphasis.

The current system is good at:
- typed artifacts
- deterministic evaluation
- harness overlays
- prompt/tool/component tuning
- lineage and report infrastructure

It is not yet good at:
- synthesizing genuinely new computation graphs
- evolving architecture instead of local fields
- allocating work across multiple specialized agents/stages
- inserting verifier gates and alternate control flow as first-class structure

This document proposes a V2 direction where topology itself becomes a first-class typed search object.

## Diagnosis

Today, the task graph is mostly fixed and the harness mutates fields on named targets.

That means search mostly happens over:
- prompt text
- component choice
- booleans like `enabled`
- scalar settings

Even with recent additions like `strategy`, the search space is still a predeclared choice among a few blocks. That is useful, but it is not enough for real harness synthesis.

The core issue is architectural:
- the executable graph is the program
- but the mutable object is mostly an overlay on the program

For real topology synthesis, the mutable object must be the graph itself, subject to type and safety constraints.

## V2 Goal

Scaffold should optimize a typed hierarchical graph, not just a fixed task plus field overrides.

A candidate should be:
- a typed computation graph
- built from legal blocks
- connected through typed ports/artifacts
- statically checked before execution
- serializable and lineage-tracked

The optimizer should mutate:
- node type
- subgraph shape
- routing structure
- loops and verifier gates
- prompt text and node-local configs

without leaving the legal typed search space.

## What To Keep

Keep these parts:
- type system and artifact typing
- objective/evaluator model
- runtime traces and rollout artifacts
- candidate archive and lineage
- stage-local diagnostics
- model/tool/prompt execution primitives

These are still good foundations.

## What To Demote

These should stop being the center of the architecture:
- fixed task graph plus mutable harness overlay
- stage-local field tuning as the main search mechanism
- prompt optimization as the main story
- treating topology changes as an afterthought

Harness overlays can remain as a convenience layer, but they should not define the limits of search.

## New Core Model

V2 should separate four things clearly:

1. Interface
- task input/output types
- artifact schemas
- safety and budget constraints
- evaluator contract

2. Library
- legal node kinds
- legal subgraph combinators
- legal rewrites
- type/effect rules

3. Candidate
- the current typed graph
- node configs and prompt text
- memory policies
- verifier policies

4. Optimizer
- mutation policy
- parent selection
- branch budgeting
- archive descriptors

The interface is fixed.
The library bounds the search space.
The candidate evolves.
The optimizer decides how evolution happens.

## Candidate Graph

A V2 candidate is a hierarchical typed graph.

Graph elements:
- nodes
- ports
- edges
- subgraphs
- guards
- loop boundaries
- verification gates

Node kinds should include at least:
- `prompt`
- `tool`
- `agent`
- `router`
- `judge`
- `verify`
- `reduce`
- `memory_write`
- `memory_retrieve`
- `subtask`

Each node has:
- input port schema
- output port schema
- local config surface
- optional prompt/system text

Each edge carries a typed artifact or port value.

## Structural Search

Topology search should happen through legal graph rewrites, not arbitrary code mutation.

Useful rewrite operators:
- add node
- remove node
- replace node kind
- replace component
- insert verifier after node
- split one node into two nodes
- wrap node in retry/review loop
- branch on router output
- fan out to parallel candidates then reduce
- collapse subgraph into a reusable subtask
- bypass a subgraph with a shortcut path

Each rewrite must preserve:
- type compatibility
- declared artifact ownership rules
- evaluator contract
- safety/resource limits

That is the key difference from unconstrained code evolution.

## Hierarchy

Topology synthesis should be hierarchical by default.

Instead of one flat graph, allow:
- subgraphs
- subtasks
- reusable strategy fragments
- nested workspaces/memory scopes

This matters because the most valuable topology changes are often:
- planner -> worker -> verifier
- retrieve -> answer -> verify
- direct path vs decomposed path
- parallel candidates -> judge

These are naturally hierarchical patterns.

## Verifier Gates

Verifier gates should be first-class nodes, not an afterthought.

Examples:
- output schema verifier
- abstention verifier
- evidence sufficiency verifier
- consistency verifier
- answer-vs-context verifier

These gates can:
- accept
- reject
- abstain
- trigger repair
- route to fallback subgraph

This is one of the biggest missing pieces today.

## Memory

Memory should be explicit and typed.

Not:
- hidden chat history

Instead:
- typed memory artifacts
- memory write nodes
- memory retrieve nodes
- memory consolidation nodes
- scoped memory stores

This lets topology synthesis discover things like:
- direct-read strategy
- compressed fact memory
- hybrid memory plus direct evidence
- update-aware retrieval plus verifier

## DSL Direction

The DSL should probably move toward graph fragments and search templates.

Conceptually:

```scaffold
task answer_memory_question {
    input: LongMemEvalInput
    output: LongMemEvalAnswer

    graph {
        source input

        strategy direct_read {
            node direct_answer: prompt direct_answer_longmemeval_memory
            connect input -> direct_answer
            emit direct_answer.output
        }

        strategy compressed_memory {
            node extract: prompt extract_longmemeval_memory
            node retrieve: prompt retrieve_longmemeval_memory
            node answer: prompt answer_longmemeval_memory
            node verify: prompt verify_longmemeval_answer

            connect input -> extract
            connect input, extract.output -> retrieve
            connect input, retrieve.output -> answer
            connect input, retrieve.output, answer.output -> verify
            emit verify.output
        }
    }
}
```

Then the optimizer can mutate:
- which strategy exists
- what nodes are in each strategy
- how they connect
- where verifier gates sit

## Search Surfaces

Search should happen on three layers:

1. Topology
- node/subgraph structure
- routing
- loops
- verifier placement

2. Components
- prompt/tool/agent choice
- model routing
- memory policy

3. Text
- prompt edits
- system prompt edits
- stage-local repair text

The optimizer should not start from text.
It should start from topology and components, then use text mutation inside promising subgraphs.

## Optimizer Design

The current archive/lineage work is still useful, but the unit of optimization changes.

The archive should store:
- candidate graph
- lineage
- graph descriptors
- split metrics
- stage/subgraph diagnostics
- resource use

Selection should be:
- graph-aware
- lineage-aware
- validation-aware

Good descriptor axes:
- topology family
- verifier usage
- memory usage
- cost/latency bucket
- abstention behavior
- primary score bucket

MAP-Elites still fits well here.

## Migration

Do not keep extending the current surface indefinitely.

Suggested path:

1. Retire the current surface to legacy status.
- keep it runnable only as a temporary reference
- stop using it as the long-term design target

2. Add a parallel V2 IR.
- graph-first
- hierarchical
- typed ports and rewrites

3. Implement a minimal V2 runtime.
- enough for prompt/tool/agent/router/verify nodes

4. Port one benchmark fully.
- LongMemEval is a strong candidate
- especially direct-read vs memory-pipeline vs verified-memory

5. Only then expand optimizer sophistication.

This avoids burying the reset inside compatibility hacks.

## Recommendation

Yes, a ground-up rethink is justified.

Not because the current work is wasted, but because the current architecture still assumes:
- fixed graph
- mutable overlay

and real topology synthesis needs:
- mutable typed graph
- fixed interface and constraints

That is the architectural pivot.
