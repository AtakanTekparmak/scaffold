//! Generated CLI entrypoint
//!
//! Run with: cargo run -- --help

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "multi_agent_flow")]
#[command(about = "Generated scaffold CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a pipeline
    Pipeline {
        /// Name of the pipeline to run
        name: String,
        /// JSON input
        #[arg(short, long)]
        input: String,
    },
    /// Run a tool
    Tool {
        /// Name of the tool to run
        name: String,
        /// JSON input
        #[arg(short, long)]
        input: String,
    },
    /// Run an agent
    Agent {
        /// Name of the agent to run
        name: String,
        /// JSON input
        #[arg(short, long)]
        input: String,
    },
    /// Run a prompt
    Prompt {
        /// Name of the prompt to run
        name: String,
        /// JSON input
        #[arg(short, long)]
        input: String,
    },
    /// List available components
    List,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pipeline { name, input: input_json } => {
            match name.as_str() {
            "multi_agent_flow" => {
                let input: multi_agent_flow_lib::pipelines::multi_agent_flow::Input = serde_json::from_str(&input_json)?;
                let result = multi_agent_flow_lib::pipelines::multi_agent_flow::run(input).await?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
                _ => eprintln!("Unknown pipeline: {}. Use 'list' to see available pipelines.", name),
            }
        }
        Commands::Tool { name, input: input_json } => {
            match name.as_str() {
            "word_counter" => {
                let input: multi_agent_flow_lib::tools::word_counter::Input = serde_json::from_str(&input_json)?;
                let result = multi_agent_flow_lib::tools::word_counter::run(input).await?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
                _ => eprintln!("Unknown tool: {}. Use 'list' to see available tools.", name),
            }
        }
        Commands::Agent { name, input: input_json } => {
            match name.as_str() {
            "researcher" => {
                let input: multi_agent_flow_lib::agents::researcher::Input = serde_json::from_str(&input_json)?;
                let result = multi_agent_flow_lib::agents::researcher::run(input).await?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            "writer" => {
                let input: multi_agent_flow_lib::agents::writer::Input = serde_json::from_str(&input_json)?;
                let result = multi_agent_flow_lib::agents::writer::run(input).await?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
                _ => eprintln!("Unknown agent: {}. Use 'list' to see available agents.", name),
            }
        }
        Commands::Prompt { name, input: input_json } => {
            match name.as_str() {
            "summarize" => {
                let input: multi_agent_flow_lib::prompts::summarize::Input = serde_json::from_str(&input_json)?;
                let result = multi_agent_flow_lib::prompts::summarize::run(input).await?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
                _ => eprintln!("Unknown prompt: {}. Use 'list' to see available prompts.", name),
            }
        }
        Commands::List => {
            println!("Available components:\n");
            println!("Pipelines:");
            println!("  - multi_agent_flow");
            println!("\nTools:");
            println!("  - word_counter");
            println!("\nAgents:");
            println!("  - researcher
  - writer");
            println!("\nPrompts:");
            println!("  - summarize");
        }
    }

    Ok(())
}
