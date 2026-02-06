# Scaffold DSL Reference

Scaffold is a domain-specific language for defining LLM-based agents and workflows with support for RL optimization.

## Core Concepts

- **Tools**: Functions with implementations (shell commands, foreign calls)
- **Prompts**: LLM calls with structured input/output
- **Agents**: LLM + tools loop with reward/done signals
- **Pipelines**: Sequential composition of tools and prompts

## Syntax Reference

### Types

```scaffold
type Position = { x: int, y: int }
type Result = { success: bool, message: string }
```

**Primitive types:** `bool`, `int`, `float`, `string`, `bytes`, `any`
**Container types:** `list<T>`, `map<K, V>`, `option<T>`, `result<T, E>`

### Tools

```scaffold
tool word_counter {
    input: { text: string }
    output: { count: int }
    impl: shell("echo '{text}' | wc -w")
}

// Build structured output without shell JSON
tool word_counter_struct {
    input: { text: string }
    output: { count: int }
    impl: json { count: 42 }
    // `map { ... }` is a synonym for `json { ... }`
}

// With pre/post conditions
tool safe_divide {
    input: { a: int, b: int }
    output: int
    spec {
        pre: b != 0
        post: output * b == a
    }
    impl: a / b
}
```

### Prompts

```scaffold
prompt summarize {
    input: { text: string }
    output: { summary: string }
    template: "Summarize this text:\n\n{text}"
}

// With system prompt from file
prompt analyze {
    input: { data: string }
    output: { analysis: string }
    system: file("prompts/analyzer.md")
    template: "Analyze: {data}"
}
```

### Agents

Agents combine an LLM with tools in a loop.

```scaffold
agent researcher {
    input: { question: string }
    output: { answer: string, sources: list<string> }

    tools: [web_search, read_file]

    system: "You are a research assistant..."

    max_turns: 10

    // RL optimization signals
    reward: length(answer) > 50
    done: sources.length > 0 && length(answer) > 100

    // Error handling (optional)
    on_error: retry(3)    // or: abort (default)
    timeout: 60           // seconds
}
```

**Error strategies:**
- `abort` - Fail immediately on error (default)
- `retry(N)` - Retry up to N times before failing

### Pipelines

Pipelines chain tools and prompts sequentially.

```scaffold
pipeline analyze_and_summarize {
    input: { text: string }
    output: { result: string }

    steps {
        let word_count = word_counter(text)
        let summary = summarize(text)
    }

    // RL optimization signal
    reward: word_count.count > 0
}
```

Pipelines can also call agents and use control flow:

```scaffold
pipeline research_flow {
    input: { question: string }
    output: { answer: string }

    steps {
        let research = researcher(question)

        if length(research.answer) > 0 {
            let final = writer(research)
        } else {
            let final = fallback_writer(question)
        }

        parallel {
            { let draft = writer(research) }
            { let critique = critic(research) }
        }
    }
}
```

Notes:
- `if`/`match` branches execute their own step blocks.
- `parallel` executes each branch block concurrently, returns no value, and does not export branch-local bindings.

### Foreign Functions (FFI)

```scaffold
extern crate regex = "1.10"

foreign rust str_utils {
    fn length(text: string) -> int
    fn to_upper(text: string) -> string
}
```

## CLI Usage

```bash
# Parse and validate
scaffold parse example.scaffold

# Compile to IR (JSON)
scaffold compile example.scaffold -o output.json

# Run a tool
scaffold run example.scaffold --tool my_tool --input '{"text": "hello"}'

# Run an agent
scaffold run example.scaffold --agent my_agent --input '{"question": "..."}'

# Run a pipeline
scaffold run example.scaffold --pipeline my_pipeline --input '{"data": "..."}'

# Generate Rust code
scaffold codegen example.scaffold -o generated/
```

## Configuration

Create `~/.scaffold/config.toml`:

```toml
default_model = "gpt-4o-mini"

[llm.openai]
api_key = "sk-..."

[llm.anthropic]
api_key = "sk-ant-..."
```

Or use environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
