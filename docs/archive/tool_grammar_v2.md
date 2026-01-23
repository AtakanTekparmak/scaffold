# Tool Declarations - Grammar Extension (v2)

## New Grammar Rules

```ebnf
(* Extended declarations *)
declaration    = task_decl | type_decl | grounding_decl | tool_decl | import_decl ;

(* Tool declaration - defines a callable interface *)
tool_decl      = "tool" IDENT "(" [ param_list ] ")" [ "->" type_expr ] tool_body ;
param_list     = param { "," param } ;
param          = IDENT ":" type_expr ;
tool_body      = ";" (* abstract - implemented externally *)
               | "{" tool_impl "}" ;

(* Tool implementation - target-specific code *)
tool_impl      = { target_impl } ;
target_impl    = IDENT ":" ( STRING | block_string ) ;  (* target: code *)
block_string   = "```" { any } "```" ;  (* multiline code block *)

(* Import external tool libraries *)
import_decl    = "import" STRING [ "as" IDENT ] ;

(* Updated grounding - now references tools by name with args *)
ground_target  = tool_call | llm_call | sandbox_call ;
tool_call      = IDENT "(" [ arg_list ] ")" ;  (* tool_name(args) *)
arg_list       = expr { "," expr } ;
llm_call       = "llm" "(" STRING [ "," llm_opts ] ")" ;
sandbox_call   = "sandbox" "(" IDENT "," tool_call ")" ;  (* sandbox(env, tool(args)) *)

(* LLM options *)
llm_opts       = "{" { llm_opt } "}" ;
llm_opt        = "model" ":" STRING
               | "temperature" ":" FLOAT
               | "schema" ":" type_expr ;  (* structured output *)
```

## Example: Exploit Finder with Typed Tools

```scaffold
// Import a tool library (provides tool signatures)
import "security_tools.scaffold" as sec

// Declare tool signatures (abstract - runtime provides impl)
tool checksec(binary: string) -> {
    nx: bool,
    pie: bool,
    canary: bool,
    relro: string
};

tool move(env: Environment, dx: int, dy: int) -> MoveResult;

// Declare with inline implementations per target
tool read_memory(addr: int, size: int) -> list<int> {
    python: ```
        import ctypes
        return list(ctypes.string_at(addr, size))
    ```
    rust: ```
        unsafe { std::slice::from_raw_parts(addr as *const u8, size).to_vec() }
    ```
}

// Declare tool with validation
tool execute_payload(payload: string) -> ExecutionResult {
    pre: payload.length < 4096  // compile-time check on args
    post: result.exit_code != null

    python: "subprocess.run(payload, shell=True, capture_output=True)"
}

// Types for tool results
type MoveResult = { success: bool, new_position: Position }
type ExecutionResult = {
    exit_code: int,
    stdout: string,
    stderr: string,
    timeout: bool
}

type Environment = {
    pid: int,
    base_addr: int,
    writable_regions: list<MemoryRegion>
}

type MemoryRegion = { start: int, end: int, perms: string }

// Grounding now uses typed tool calls
grounding navigation_env {
    // Direct tool invocation - type checked!
    action move_north: move(env, 0, 1)
    action move_south: move(env, 0, -1)

    // Tool in sandbox with resource limits
    action analyze: sandbox(isolated, checksec(target.binary_path)) {
        timeout: 30s
        memory: 256mb
        network: false
    }

    // LLM with structured output matching a type
    action suggest_bypass: llm("Suggest ASLR bypass for {target.arch}") {
        model: "claude-3"
        temperature: 0.2
        schema: { technique: string, steps: list<string>, confidence: float }
    }

    // Chained tools
    action leak_and_compute: chain(
        read_memory(got_addr, 8),
        compute_base(result)
    )
}

task exploit {
    input: { target: Target }
    output: ExecutionResult

    subgoal recon {
        pre: target.binary_path != null
        // Now type-checked: checksec returns the right type
        done: protections.nx != null && protections.pie != null
        options: [analyze]  // references grounding action
        timeout: 30s
    }

    // ...
}
```

## Type Checking Additions

### Tool Signature Verification
```
1. When grounding references tool(args):
   - Verify tool is declared
   - Verify arg count matches param count
   - Verify each arg type matches param type

2. When subgoal uses action in options:
   - Verify action is grounded
   - Propagate tool return type to subgoal context
```

### New Type Rules
```
tool_call(T, args) : ReturnType(T)
  where args : ParamTypes(T)

sandbox(env, tool_call) : ReturnType(tool_call)
  with timeout semantics (may return timeout error)

llm(prompt, {schema: T}) : T
  structured output guaranteed to match schema

chain(t1, t2, ..., tn) : ReturnType(tn)
  where each ti can reference 'result' from t(i-1)
```

## IR Changes

```rust
// Tool declaration in IR
struct ToolDeclIR {
    name: String,
    params: Vec<ParamIR>,
    return_type: Option<TypeIR>,
    implementations: HashMap<String, String>,  // target -> code
    preconditions: Vec<ExprIR>,
    postconditions: Vec<ExprIR>,
}

struct ParamIR {
    name: String,
    param_type: TypeIR,
}

// Updated grounding target
enum GroundTargetIR {
    ToolCall {
        tool: String,           // tool name
        args: Vec<ExprIR>,      // typed arguments
    },
    LlmCall {
        prompt: String,
        model: Option<String>,
        temperature: Option<f64>,
        output_schema: Option<TypeIR>,
    },
    Sandbox {
        environment: String,
        inner: Box<GroundTargetIR>,
        limits: SandboxLimits,
    },
    Chain {
        steps: Vec<GroundTargetIR>,
    },
}
```

## Codegen Output (Python example)

```python
# Generated from tool declarations
from typing import TypedDict, List
from scaffold_runtime import Tool, Sandbox, LLM

class MoveResult(TypedDict):
    success: bool
    new_position: Position

class checksec(Tool):
    def __call__(self, binary: str) -> ChecksecResult:
        # Implementation injected at runtime or from tool_impl
        ...

class move(Tool):
    def __call__(self, env: Environment, dx: int, dy: int) -> MoveResult:
        ...

# Generated grounding
class NavigationEnvGrounding:
    def __init__(self, runtime):
        self.env = runtime.get_env()
        self.move = runtime.get_tool("move")
        self.checksec = runtime.get_tool("checksec")

    def move_north(self) -> MoveResult:
        return self.move(self.env, 0, 1)

    def analyze(self) -> ChecksecResult:
        with Sandbox(timeout=30, memory="256mb", network=False):
            return self.checksec(self.target.binary_path)

    def suggest_bypass(self) -> BypassSuggestion:
        return LLM.call(
            prompt=f"Suggest ASLR bypass for {self.target.arch}",
            model="claude-3",
            temperature=0.2,
            schema=BypassSuggestion,
        )
```

## Migration Path

1. **v1 (current)**: `tool("string")` - untyped, string passed to runtime
2. **v1.5**: Allow both old syntax and new `tool_name(args)` syntax
3. **v2**: Deprecate string-based tools, require declarations

Old code:
```scaffold
action move_north: tool("env.move(0, 1)")
```

New code:
```scaffold
tool move(env: Env, dx: int, dy: int) -> MoveResult;
action move_north: move(env, 0, 1)
```