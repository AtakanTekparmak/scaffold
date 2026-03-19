# Scaffold Task/Harness V1 Grammar

Status: proposed surface grammar

This document is the parser-facing companion to [task-harness-v1.md](/Users/kayaomers/Documents/firstbatch/scaffold/docs/task-harness-v1.md).

Its job is narrower than the main design doc:

- lock the exact V1 surface syntax
- keep the grammar parser-friendly
- make clear which rules are syntactic and which are semantic checks

## Goals

The grammar is designed to support the stronger direction for Scaffold:

- tasks are first-class orchestration units
- harnesses are typed overlays on tasks
- objectives define evaluation and optimization
- artifact flow is explicit and typed
- optimization mutates only declared tunable fields

## Non-Goals

This grammar does not try to solve every future language feature.

In particular, V1 intentionally does not add:

- arbitrary graph rewrites from harnesses
- free-form mutation of undeclared code
- implicit cycles
- multiple unrelated ways to encode the same orchestration shape

## Parsing Conventions

- Whitespace and newlines are insignificant except inside string literals.
- Line comments and block comments follow the current language rules.
- Trailing commas are allowed in list literals and record literals.
- `type` and `artifact` share the same underlying `type_expr` grammar.
- `artifact` is a typed declaration with extra semantic meaning, not a second schema language.

## Top-Level Grammar

```ebnf
program
    = { declaration } ;

declaration
    = type_decl
    | artifact_decl
    | extern_crate_decl
    | foreign_decl
    | tool_decl
    | prompt_decl
    | agent_decl
    | task_decl
    | harness_decl
    | objective_decl ;

type_decl
    = "type" IDENT "=" type_expr ;

artifact_decl
    = "artifact" IDENT "=" type_expr ;
```

`extern_crate_decl`, `foreign_decl`, `tool_decl`, `prompt_decl`, and `agent_decl` keep their existing syntax for V1 unless explicitly revised later.

## Types

V1 keeps the current type system and extends its use across tasks and artifacts.

```ebnf
type_expr
    = primitive_type
    | named_type
    | struct_type
    | list_type
    | map_type
    | option_type
    | result_type ;

primitive_type
    = "bool"
    | "int"
    | "float"
    | "string"
    | "bytes"
    | "any" ;

named_type
    = IDENT ;

struct_type
    = "{" [ type_field { "," type_field } [ "," ] ] "}" ;

type_field
    = IDENT ":" type_expr ;

list_type
    = "list" "<" type_expr ">" ;

map_type
    = "map" "<" type_expr "," type_expr ">" ;

option_type
    = "option" "<" type_expr ">"
    | type_expr "?" ;

result_type
    = "result" "<" type_expr "," type_expr ">" ;
```

## Tasks

Tasks are the semantic center of V1.

```ebnf
task_decl
    = "task" IDENT "{"
        task_input_decl
        task_output_decl
        [ task_artifacts_decl ]
        task_node { task_node }
        task_emit_decl
      "}" ;

task_input_decl
    = "input" ":" type_expr ;

task_output_decl
    = "output" ":" type_expr ;

task_artifacts_decl
    = "artifacts" "{"
        artifact_slot_decl { artifact_slot_decl }
      "}" ;

artifact_slot_decl
    = IDENT ":" type_expr ;

task_node
    = stage_decl
    | loop_decl
    | branch_decl ;
```

### Stages

```ebnf
stage_decl
    = "stage" IDENT "using" stage_kind IDENT "{"
        stage_in_decl
        stage_out_decl
        [ stage_when_decl ]
      "}" ;

stage_kind
    = "tool"
    | "prompt"
    | "agent" ;

stage_in_decl
    = "in" ":" expr ;

stage_out_decl
    = "out" ":" IDENT ;

stage_when_decl
    = "when" ":" expr ;
```

`stage_when_decl` is optional. It exists so V1 can express simple conditional activation without requiring the harness to mutate graph structure.

### Loops

```ebnf
loop_decl
    = "loop" IDENT "{"
        loop_max_iters_decl
        loop_carry_decl
        [ loop_while_decl ]
        [ loop_until_decl ]
        task_node { task_node }
      "}" ;

loop_max_iters_decl
    = "max_iters" ":" expr ;

loop_carry_decl
    = "carry" ":" "[" [ IDENT { "," IDENT } [ "," ] ] "]" ;

loop_while_decl
    = "while" ":" expr ;

loop_until_decl
    = "until" ":" expr ;
```

V1 requires explicit loops. Cycles are illegal elsewhere.

### Branches

```ebnf
branch_decl
    = "if" expr "{"
        task_node { task_node }
      "}"
      [ "else" "{"
          task_node { task_node }
        "}" ] ;
```

V1 uses structured branches rather than arbitrary graph syntax. Semantic checks determine whether all branches produce the artifacts needed later.

### Emit

```ebnf
task_emit_decl
    = "emit" "{"
        emit_field { emit_field }
      "}" ;

emit_field
    = IDENT ":" expr ;
```

## Harnesses

Harnesses are typed overlays on tasks.

```ebnf
harness_decl
    = "harness" IDENT "for" "task" IDENT "{"
        [ harness_defaults_decl ]
        { harness_bind_decl }
        [ harness_tune_decl ]
      "}" ;

harness_defaults_decl
    = "defaults" "{"
        binding_stmt { binding_stmt }
      "}" ;

harness_bind_decl
    = "bind" IDENT "{"
        binding_stmt { binding_stmt }
      "}" ;

binding_stmt
    = binding_key ":" expr ;

binding_key
    = IDENT { "." IDENT } ;

harness_tune_decl
    = "tune" "{"
        tune_stmt { tune_stmt }
      "}" ;

tune_stmt
    = binding_path tune_operator finite_domain ;

binding_path
    = IDENT { "." IDENT } ;

tune_operator
    = "in"
    | "subset_of" ;

finite_domain
    = list_literal
    | variants_domain ;

variants_domain
    = "variants" "(" STRING ")" ;
```

Examples of legal binding paths:

- `write.model`
- `write.temperature`
- `review.system_prompt`
- `refine.max_iters`

The grammar allows any path shape; semantic validation restricts which paths are mutable and what value domains are valid.

## Objectives

Objectives define evaluation and optimization.

```ebnf
objective_decl
    = "objective" IDENT "for" "task" IDENT "{"
        objective_dataset_decl
        objective_harness_decl
        [ objective_repeats_decl ]
        objective_eval_decl { objective_eval_decl }
        objective_score_decl
        [ objective_split_decl ]
        [ objective_select_decl ]
      "}" ;

objective_dataset_decl
    = "dataset" ":" dataset_spec ;

dataset_spec
    = file_call
    | inline_dataset ;

file_call
    = "file" "(" STRING ")" ;

inline_dataset
    = "[" [ inline_case { "," inline_case } [ "," ] ] "]" ;

inline_case
    = "{"
        "input" ":" expr
        [ "," "expected" ":" expr ]
        [ "," "id" ":" STRING ]
      "}" ;

objective_harness_decl
    = "harness" ":" IDENT ;

objective_repeats_decl
    = "repeats" ":" INT ;

objective_eval_decl
    = objective_constraint_decl
    | objective_checker_decl
    | objective_judge_decl
    | objective_metric_decl ;

objective_constraint_decl
    = "constraint" IDENT "=" expr ;

objective_checker_decl
    = "checker" IDENT "=" expr ;

objective_judge_decl
    = "judge" IDENT "=" expr ;

objective_metric_decl
    = "metric" IDENT "=" expr ;

The `expr` grammar remains shared. This means a `constraint`, `checker`, `judge`, or `metric` may call existing named components where supported by semantics:

- tool calls for executable checkers
- prompt calls for lightweight model judges
- agent calls for richer multi-turn judges

objective_score_decl
    = "score" "=" expr ;

objective_split_decl
    = "split" "{"
        "train" ":" number_lit
        "val" ":" number_lit
        "test" ":" number_lit
      "}" ;

objective_select_decl
    = "select" "{"
        "primary" ":" expr
        [ "tie_breakers" ":" "[" [ expr { "," expr } [ "," ] ] "]" ]
      "}" ;
```

## Expression Grammar Additions Required By V1

The current language surface is not yet enough for task/harness/objective syntax. V1 requires general-purpose list and record literals in expressions.

```ebnf
expr
    = logical_or_expr ;

logical_or_expr
    = logical_and_expr { "||" logical_and_expr } ;

logical_and_expr
    = equality_expr { "&&" equality_expr } ;

equality_expr
    = relational_expr { ("==" | "!=") relational_expr } ;

relational_expr
    = additive_expr { ("<" | ">" | "<=" | ">=") additive_expr } ;

additive_expr
    = multiplicative_expr { ("+" | "-") multiplicative_expr } ;

multiplicative_expr
    = postfix_expr { ("*" | "/") postfix_expr } ;

postfix_expr
    = primary_expr { postfix_suffix } ;

postfix_suffix
    = "." IDENT
    | "(" [ expr { "," expr } [ "," ] ] ")" ;

primary_expr
    = literal
    | IDENT
    | list_literal
    | record_literal
    | "(" expr ")" ;

list_literal
    = "[" [ expr { "," expr } [ "," ] ] "]" ;

record_literal
    = "{"
        [ record_field { "," record_field } [ "," ] ]
      "}" ;

record_field
    = IDENT ":" expr ;

literal
    = INT
    | FLOAT
    | STRING
    | "true"
    | "false"
    | "null" ;

number_lit
    = INT
    | FLOAT ;
```

### Why These Additions Matter

V1 depends on these forms:

- stage inputs use record literals
- loop `carry` uses list literals
- `tune` domains use list literals
- inline datasets use record literals
- `emit` mappings and objective expressions need normal field access and composition

Without these expression forms, the task-oriented syntax becomes awkward or impossible.

## Reserved Keywords To Add

The following structural keywords should be reserved for V1 grammar support:

- `artifact`
- `artifacts`
- `stage`
- `using`
- `when`
- `emit`
- `harness`
- `defaults`
- `bind`
- `tune`
- `objective`
- `dataset`
- `constraint`
- `checker`
- `judge`
- `metric`
- `score`
- `split`
- `select`
- `repeats`
- `train`
- `val`
- `test`
- `primary`
- `tie_breakers`
- `carry`
- `until`
- `subset_of`

Existing keywords such as `task`, `tool`, `prompt`, `agent`, `if`, `else`, `loop`, `for`, `in`, and `variants` remain in use.

## Semantic Checks Beyond Grammar

These rules are not purely syntactic, but V1 relies on them:

### Task checks

- exactly one `input`, one `output`, and one `emit` per task
- each artifact slot name is unique within the task
- each `stage out` references a declared artifact slot
- each stage output type matches the referenced artifact slot type
- each artifact is definitely assigned before use
- loops may only reassign slots named in `carry`
- all artifacts referenced in `emit` must be definitely assigned along every path

### Harness checks

- the harness target task must exist
- each `bind` target must name an existing stage or loop in that task
- each binding path must refer to a mutable harness-controlled field
- each `tune` domain must be finite in V1
- `subset_of` is only valid for list-like configurable fields

### Objective checks

- the objective target task must exist
- the objective harness must target the same task
- at least one objective evaluation declaration must be present
- `constraint` declarations must evaluate to bool
- `checker`, `judge`, and `metric` declarations must evaluate to bool or numeric
- `score` may reference only declared constraints, checkers, judges, metrics, plus approved namespaces such as `output`, `expected`, and `rollout`
- split weights must be valid and sum to 1.0

## Deliberate Constraints In V1

Some constraints are intentionally strict because they support verification and optimization:

- task declarations use explicit artifact slots instead of implicit variables
- cycles must be written as `loop`
- harnesses may configure stages but may not add or delete them
- objectives score executions, not source text
- search domains must be finite

These constraints are not accidental; they are what make "miswired harnesses do not compile" and "optimization searches declared surfaces" achievable.

## Immediate Parser Impact

This grammar implies the following syntax-layer work:

- add new keywords to the lexer
- add AST nodes for artifact declarations, task bodies, stage nodes, loop nodes, harnesses, and objectives
- extend expression parsing to support list literals and record literals
- keep existing tool/prompt/agent parsing intact where possible

That should be the next implementation step after agreeing on this grammar.
