# Scaffold Installation & Usage

## Installation

Install the Scaffold CLI globally using Cargo:

```bash
# From the repository root
cargo install --path crates/scaffold-cli

# Or install all workspace binaries
cargo install --path .
```

This installs the `scaffold` binary to `~/.cargo/bin/`.

## Quick Start

### 1. Check a scaffold file (parse + type check + verify)

```bash
scaffold check examples/simple_tool.scaffold
scaffold check examples/comprehensive.scaffold -v  # verbose
```

### 2. Run directly with the interpreter

```bash
# Run a specific tool
scaffold run examples/simple_tool.scaffold --tool count_words -i '{"text": "hello world"}'

# Run a specific pipeline
scaffold run examples/text_analysis.scaffold --pipeline analyze_text -i '{"text": "sample text"}'

# Run an agent
scaffold run examples/agent_orchestration.scaffold --agent quick_researcher -i '{"question": "What is Rust?"}'

# Run a prompt
scaffold run examples/comprehensive.scaffold --prompt summarize -i '{"text": "Long text here...", "max_words": 50}'

# Input from file
scaffold run myfile.scaffold --tool my_tool -i @input.json
```

### 3. Compile to IR (JSON)

```bash
scaffold compile examples/simple_tool.scaffold -o output.json
scaffold compile examples/simple_tool.scaffold --compact  # minified JSON
```

### 4. Generate Rust code

```bash
scaffold codegen examples/simple_tool.scaffold -o ./generated
```

This creates a Rust crate in `./generated/` with:
- `Cargo.toml`
- `src/lib.rs` (types, tools, prompts, agents, pipelines)
- `src/main.rs` (CLI entrypoint)

### 5. Build a native binary

```bash
scaffold build examples/simple_tool.scaffold -o ./output
scaffold build examples/simple_tool.scaffold --release  # optimized build
```

This generates code and runs `cargo build`, outputting the binary path.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SCAFFOLD_RUNTIME_PATH` | Path to scaffold-runtime crate (auto-detected) |
| `OPENAI_API_KEY` | Required for prompts/agents (LLM calls) |

## Command Reference

| Command | Description |
|---------|-------------|
| `scaffold check <file>` | Parse, type check, and verify |
| `scaffold compile <file>` | Compile to IR JSON |
| `scaffold codegen <file> -o <dir>` | Generate Rust code |
| `scaffold build <file>` | Generate + compile to binary |
| `scaffold run <file>` | Execute via interpreter |
| `scaffold parse <file>` | Parse only (debug) |

## Run Options

```
scaffold run <file> [OPTIONS]

Options:
  --tool <NAME>       Run a tool
  --prompt <NAME>     Run a prompt
  --agent <NAME>      Run an agent
  --pipeline <NAME>   Run a pipeline
  -i, --input <JSON>  Input JSON (or @filename)
  -v, --verbose       Show verbose output
  --verify            Enable verification checks
```
