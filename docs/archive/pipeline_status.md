# Scaffold DSL - Pipeline Status Map

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SCAFFOLD PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

  .scaffold file
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   LEXER     │────▶│   PARSER    │────▶│    TYPE     │────▶│  VERIFIER   │
│             │     │             │     │   CHECKER   │     │             │
│ [DONE ✓]    │     │ [DONE ✓]    │     │ [DONE ✓]    │     │ [PARTIAL]   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
       ┌───────────────────────────────────────────────────────────┘
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  IR GEN     │────▶│  OPTIMIZER  │────▶│  CODEGEN    │────▶│  RUNTIME    │
│             │     │             │     │             │     │             │
│ [DONE ✓]    │     │ [MISSING]   │     │ [MISSING]   │     │ [MISSING]   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
   .json IR ───────────────────────────────────────────────▶ (alternative)
                                                            Direct IR
                                                            Interpreter
```

---

## Stage-by-Stage Breakdown

### 1. LEXER (`scaffold-syntax/lexer.rs`) ✅ COMPLETE

**What it does:**
- Tokenizes `.scaffold` source files
- Handles keywords, operators, literals, identifiers
- Skips whitespace and comments

**What we have:**
```
✓ All keywords (task, type, grounding, subgoal, etc.)
✓ All operators (==, !=, ->, &&, ||, etc.)
✓ Literals (int, float, string, bool, null)
✓ Duration literals (5s, 10m, 1h)
✓ Size literals (256mb, 4gb)
✓ Line/block comments
```

**What's missing:**
```
○ Better error recovery on invalid tokens
○ Unicode identifier support
```

---

### 2. PARSER (`scaffold-syntax/parser.rs`) ✅ COMPLETE

**What it does:**
- Recursive descent parser → AST
- Handles full grammar from design doc

**What we have:**
```
✓ Type declarations (primitives, structs, list, map, option)
✓ Task declarations (input, output, decompose, subgoal, verify, on_fail)
✓ Grounding declarations (tool, llm, sandbox)
✓ Expressions (binary ops, field access, function calls)
✓ Subgoal items (pre, post, done, options, reward, timeout)
✓ Verification statements (reachable, no_deadlock, bounded, grounded, terminates)
✓ Failure strategies (retry, rollback, abort, replan)
```

**What's missing:**
```
○ Error recovery (currently fails on first error)
○ Better span tracking for nested expressions
○ Tool declarations (v2 feature)
○ Import statements (v2 feature)
```

---

### 3. TYPE CHECKER (`scaffold-types/checker.rs`) ✅ COMPLETE (Basic)

**What it does:**
- Validates type consistency across the scaffold
- Resolves named types
- Checks expression types

**What we have:**
```
✓ Type definition collection
✓ Named type resolution
✓ Struct field access checking
✓ Binary operator type checking
✓ Pre/post/done must be bool
✓ Reward must be numeric
✓ Subgoal completeness (must have done + options)
✓ Grounding action collection
```

**What's missing:**
```
○ Input/output type flow analysis
  - Does subgoal chain produce the output type?

○ Pre/post condition chaining verification
  - Does subgoal[i].post imply subgoal[i+1].pre?

○ Full type inference for complex expressions
  - Currently many things default to 'any'

○ Generic type support
  - list<T> where T is inferred

○ Tool signature checking (v2)
  - Verify tool(args) matches tool declaration
```

---

### 4. VERIFIER (`scaffold-verify/`) ⚠️ PARTIAL

**What it does:**
- Static analysis for safety properties
- Graph-based deadlock detection
- Bounds analysis

**What we have:**
```
✓ Deadlock detection (cycle finding via petgraph)
✓ Basic bounds tracking (from bounded() calls)
✓ Grounding completeness check
✓ Termination condition presence check
✓ Decomposition graph construction
```

**What's missing:**
```
○ Reachability analysis (STUBBED)
  - Currently just collects variables
  - Need: symbolic execution or abstract interpretation
  - Need: SMT solver integration (z3) for pre/post reasoning

○ Actual bounds verification (STUBBED)
  - Currently just records declared bounds
  - Need: loop analysis to PROVE bounds hold

○ Termination proofs (STUBBED)
  - Currently checks syntax only
  - Need: ranking functions, well-founded orderings

○ State space analysis
  - What states are reachable?
  - Is goal state always reachable?

○ Grounding coverage
  - Are all code paths grounded?
  - Any orphan actions?
```

---

### 5. IR GENERATION (`scaffold-ir/`) ✅ COMPLETE

**What it does:**
- Lowers AST to serializable IR
- JSON output for storage/transmission

**What we have:**
```
✓ Full IR structure definitions
✓ AST → IR lowering
✓ JSON serialization/deserialization
✓ Source span preservation
✓ Verification results in IR
```

**What's missing:**
```
○ Binary serialization (msgpack/protobuf)
○ IR optimization passes
○ IR validation (post-lowering checks)
```

---

### 6. OPTIMIZER ❌ MISSING

**What it would do:**
- Optimize IR before codegen
- Simplify expressions
- Inline trivial subgoals
- Dead code elimination

**Needed:**
```
○ Constant folding
○ Dead subgoal elimination
○ Decomposition graph simplification
○ Common subexpression elimination in conditions
○ Timeout normalization
```

---

### 7. CODE GENERATOR ❌ MISSING

**What it would do:**
- IR → executable code in target language

**Needed for each target:**

```
┌─────────────────────────────────────────────────────────┐
│                    CODE GENERATION                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  IR ──┬──▶ Python                                       │
│       │    ├─ Type definitions (dataclasses/TypedDict) │
│       │    ├─ Task class with subgoal methods          │
│       │    ├─ Grounding wrappers                       │
│       │    └─ Runtime integration hooks                │
│       │                                                 │
│       ├──▶ Rust                                        │
│       │    ├─ Type definitions (structs)               │
│       │    ├─ Task trait implementations               │
│       │    ├─ Async subgoal execution                  │
│       │    └─ Tool trait bounds                        │
│       │                                                 │
│       ├──▶ TypeScript                                  │
│       │    ├─ Type definitions (interfaces)            │
│       │    ├─ Task class                               │
│       │    └─ Promise-based execution                  │
│       │                                                 │
│       └──▶ WASM (future)                               │
│            └─ Portable execution                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 8. RUNTIME ❌ MISSING

**What it would do:**
- Execute compiled scaffolds
- Manage subgoal state machine
- Interface with tools/LLMs
- Handle failures and replanning

**Components needed:**

```
┌─────────────────────────────────────────────────────────┐
│                       RUNTIME                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Executor   │  │   State     │  │   Tool      │    │
│  │             │  │   Manager   │  │   Registry  │    │
│  │ - Run task  │  │ - Track     │  │ - Register  │    │
│  │ - Schedule  │  │   subgoal   │  │   tools     │    │
│  │   subgoals  │  │   progress  │  │ - Invoke    │    │
│  │ - Handle    │  │ - Snapshot  │  │ - Sandbox   │    │
│  │   failures  │  │ - Rollback  │  │   isolation │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                │                │            │
│         └────────────────┼────────────────┘            │
│                          ▼                             │
│                 ┌─────────────┐                        │
│                 │   LLM       │                        │
│                 │   Interface │                        │
│                 │ - Prompting │                        │
│                 │ - Parsing   │                        │
│                 │ - Retries   │                        │
│                 └─────────────┘                        │
│                          │                             │
│                          ▼                             │
│                 ┌─────────────┐                        │
│                 │  Trace      │                        │
│                 │  Logger     │                        │
│                 │ - Events    │                        │
│                 │ - Metrics   │                        │
│                 │ - Debug     │                        │
│                 └─────────────┘                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Feature Matrix

| Feature                          | Status      | Crate              |
|----------------------------------|-------------|-------------------|
| **Syntax**                       |             |                   |
| Lexer                            | ✅ Done     | scaffold-syntax   |
| Parser                           | ✅ Done     | scaffold-syntax   |
| AST definitions                  | ✅ Done     | scaffold-syntax   |
| Error reporting (ariadne)        | ✅ Done     | scaffold-cli      |
| **Type System**                  |             |                   |
| Primitive types                  | ✅ Done     | scaffold-types    |
| Compound types (list, map, etc)  | ✅ Done     | scaffold-types    |
| Struct types                     | ✅ Done     | scaffold-types    |
| Type resolution                  | ✅ Done     | scaffold-types    |
| Expression type checking         | ✅ Done     | scaffold-types    |
| Type inference                   | ⚠️ Partial  | scaffold-types    |
| Generic types                    | ❌ Missing  | -                 |
| **Verification**                 |             |                   |
| Deadlock detection               | ✅ Done     | scaffold-verify   |
| Grounding check                  | ✅ Done     | scaffold-verify   |
| Bounds tracking                  | ⚠️ Partial  | scaffold-verify   |
| Reachability analysis            | ❌ Stub     | scaffold-verify   |
| Termination proofs               | ❌ Stub     | scaffold-verify   |
| SMT-based verification           | ❌ Missing  | -                 |
| **IR**                           |             |                   |
| IR structure                     | ✅ Done     | scaffold-ir       |
| JSON serialization               | ✅ Done     | scaffold-ir       |
| Binary serialization             | ❌ Missing  | -                 |
| **CLI**                          |             |                   |
| `parse` command                  | ✅ Done     | scaffold-cli      |
| `check` command                  | ✅ Done     | scaffold-cli      |
| `compile` command                | ✅ Done     | scaffold-cli      |
| `run` command                    | ❌ Missing  | -                 |
| LSP server                       | ❌ Missing  | -                 |
| **Codegen**                      |             |                   |
| Python backend                   | ❌ Missing  | -                 |
| Rust backend                     | ❌ Missing  | -                 |
| TypeScript backend               | ❌ Missing  | -                 |
| **Runtime**                      |             |                   |
| Task executor                    | ❌ Missing  | -                 |
| State management                 | ❌ Missing  | -                 |
| Tool registry                    | ❌ Missing  | -                 |
| LLM integration                  | ❌ Missing  | -                 |
| Sandbox execution                | ❌ Missing  | -                 |
| Execution tracing                | ❌ Missing  | -                 |
| **v2 Features**                  |             |                   |
| Tool declarations                | ❌ Missing  | -                 |
| Import statements                | ❌ Missing  | -                 |
| Scaffold fading                  | ❌ Missing  | -                 |

---

## Recommended Implementation Order

### Phase 1: Complete Verification (Current Gap)
```
1. Add z3 dependency
2. Implement symbolic pre/post condition analysis
3. Implement actual reachability checking
4. Add verification result details to IR
```

### Phase 2: Python Codegen (Fastest Path to Execution)
```
1. Define Python runtime interface
2. Implement IR → Python codegen
3. Generate dataclasses for types
4. Generate task executor skeleton
```

### Phase 3: Python Runtime
```
1. Implement task executor loop
2. Implement state manager
3. Implement tool registry + invocation
4. Add basic LLM integration (anthropic SDK)
5. Implement failure handling (retry, rollback, replan)
```

### Phase 4: Tool System (v2)
```
1. Add tool declarations to grammar
2. Update parser
3. Add tool signature type checking
4. Update codegen to emit typed tool calls
```

### Phase 5: Polish
```
1. LSP server for IDE support
2. Better error messages
3. Documentation generator
4. Test framework for scaffolds
```

---

## Current File Map

```
scaffold-lang/
├── Cargo.toml                     # Workspace
├── docs/
│   ├── pipeline_status.md         # This file
│   └── tool_grammar_v2.md         # v2 tool design
├── examples/
│   ├── navigate.scaffold          # Navigation example
│   └── exploit_finder.scaffold    # Security research example
└── crates/
    ├── scaffold-syntax/           # ✅ COMPLETE
    │   └── src/
    │       ├── ast.rs             # AST nodes
    │       ├── lexer.rs           # Tokenizer
    │       ├── parser.rs          # Recursive descent
    │       └── lib.rs
    ├── scaffold-types/            # ✅ COMPLETE (basic)
    │   └── src/
    │       ├── types.rs           # Type definitions
    │       ├── checker.rs         # Type checking
    │       └── lib.rs
    ├── scaffold-verify/           # ⚠️ PARTIAL
    │   └── src/
    │       ├── reachability.rs    # Stub
    │       ├── deadlock.rs        # Working
    │       ├── bounds.rs          # Partial
    │       └── lib.rs
    ├── scaffold-ir/               # ✅ COMPLETE
    │   └── src/
    │       ├── ir.rs              # IR structures
    │       ├── serialize.rs       # JSON + lowering
    │       └── lib.rs
    ├── scaffold-cli/              # ✅ COMPLETE
    │   └── src/
    │       └── main.rs            # CLI commands
    │
    │   # MISSING CRATES:
    ├── scaffold-codegen/          # ❌ TODO
    ├── scaffold-runtime/          # ❌ TODO
    └── scaffold-lsp/              # ❌ TODO (nice to have)
```
