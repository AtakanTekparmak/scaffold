//! Agent code generation
//!
//! Generates Rust implementations for scaffold agents (multi-turn LLM with tools).
//! Agents use rig's agent builder with tool registration for native tool calling.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use scaffold_ir::{AgentIR, StringOrFileIR};

use crate::types::gen_type;
use crate::util::{to_pascal_case, to_snake_case};

/// Generate the agents mod.rs
pub fn gen_agents_mod(agent_names: &[&str]) -> TokenStream {
    let mods: Vec<_> = agent_names
        .iter()
        .map(|name| {
            let mod_name = format_ident!("{}", to_snake_case(name));
            quote! {
                pub mod #mod_name;
            }
        })
        .collect();

    let uses: Vec<_> = agent_names
        .iter()
        .map(|name| {
            let mod_name = format_ident!("{}", to_snake_case(name));
            let struct_name = format_ident!("{}Agent", to_pascal_case(name));
            quote! {
                pub use #mod_name::#struct_name;
            }
        })
        .collect();

    quote! {
        //! Agent implementations

        #(#mods)*

        #(#uses)*
    }
}

/// Generate an agent module
pub fn gen_agent_module(agent: &AgentIR) -> TokenStream {
    let agent_name = &agent.name;
    let struct_name = format_ident!("{}Agent", to_pascal_case(agent_name));
    let input_type = gen_type(&agent.input);
    let output_type = gen_type(&agent.output);

    // Generate system prompt
    let system_str = gen_string_or_file(&agent.system);

    // Generate tool list
    let tool_names: Vec<_> = agent.tools.iter().collect();

    // Generate model (optional, falls back to config default)
    let model_str = agent.model.as_deref();

    // Generate max_turns
    let max_turns = agent.max_turns.unwrap_or(10);

    // Generate tool registration calls for rig agent builder
    let tool_registrations: Vec<_> = agent
        .tools
        .iter()
        .map(|t| {
            let tool_struct = format_ident!("{}Tool", to_pascal_case(t));
            quote! {
                .tool(crate::tools::#tool_struct::new())
            }
        })
        .collect();

    let doc = format!("Agent: {}", agent_name);

    quote! {
        #![doc = #doc]

        use crate::types::*;
        use scaffold_runtime::prelude::*;
        use scaffold_runtime::rig::agent::AgentBuilder;
        use scaffold_runtime::rig::completion::Prompt;
        use scaffold_runtime::rig::client::CompletionClient;
        use schemars::JsonSchema;

        /// Input type for this agent
        pub type Input = #input_type;

        /// Output type for this agent
        pub type Output = #output_type;

        /// Agent implementation struct
        #[derive(Clone, Debug)]
        pub struct #struct_name {
            system_prompt: String,
            model: Option<String>,
            max_turns: u64,
        }

        impl #struct_name {
            /// Create a new agent instance
            pub fn new() -> Self {
                Self {
                    system_prompt: #system_str.to_string(),
                    model: #model_str.map(|s| s.to_string()),
                    max_turns: #max_turns,
                }
            }

            /// Get the model to use (agent-specific or config default)
            pub fn model(&self) -> &str {
                self.model.as_deref().unwrap_or(&scaffold_runtime::config().default_model)
            }

            /// Get the agent name
            pub fn name(&self) -> &'static str {
                #agent_name
            }

            /// Get the system prompt
            pub fn system_prompt(&self) -> &str {
                &self.system_prompt
            }

            /// Get available tool names
            pub fn available_tools(&self) -> &'static [&'static str] {
                &[#(#tool_names),*]
            }

            /// Get the JSON schema for the expected output (derived from schemars)
            pub fn output_schema() -> serde_json::Value {
                let schema = schemars::schema_for!(Output);
                serde_json::to_value(schema).unwrap_or_default()
            }

            /// Execute the agent using rig's agent builder with native tool calling
            ///
            /// This creates a rig agent with registered tools and executes the
            /// agentic loop. The LLM will automatically call tools as needed and
            /// return a final structured response.
            pub async fn execute<C: CompletionClient + Send + Sync>(
                &self,
                client: &C,
                model: &str,
            ) -> impl FnOnce(Input) -> std::pin::Pin<Box<dyn std::future::Future<Output = scaffold_runtime::Result<Output>> + Send + 'static>>
            where
                C: Clone + 'static,
            {
                let system_prompt = self.system_prompt.clone();
                let max_turns = self.max_turns;
                let client = client.clone();
                let model = model.to_string();

                move |input: Input| {
                    Box::pin(async move {
                        // Serialize input to include in the prompt
                        let input_json = serde_json::to_string_pretty(&input)
                            .map_err(|e| scaffold_runtime::Error::SerializationError(e.to_string()))?;

                        // Build output schema instructions
                        let output_schema = Self::output_schema();
                        let schema_str = serde_json::to_string_pretty(&output_schema)
                            .unwrap_or_default();

                        // Build the full system prompt with output format instructions
                        let full_preamble = format!(
                            "{}\n\nWhen you have completed the task, respond with ONLY valid JSON matching this schema (no markdown, no explanation):\n{}",
                            system_prompt,
                            schema_str
                        );

                        // Build rig agent with tools
                        let completion_model = client.completion_model(&model);
                        let rig_agent = AgentBuilder::new(completion_model)
                            .preamble(&full_preamble)
                            #(#tool_registrations)*
                            .max_tokens(4096)
                            .build();

                        // Execute the agent with the input
                        let user_message = format!("Input: {}", input_json);
                        let response = rig_agent.prompt(&user_message)
                            .await
                            .map_err(|e| scaffold_runtime::Error::Runtime(format!("Agent execution failed: {}", e)))?;

                        // Parse the response as our output type
                        let output: Output = serde_json::from_str(&response)
                            .map_err(|e| scaffold_runtime::Error::ParseError(
                                format!("Failed to parse agent output: {}. Response was: {}", e, response)
                            ))?;

                        Ok(output)
                    })
                }
            }

            /// Execute the agent directly with input (convenience method)
            pub async fn run_with<C: CompletionClient + Clone + Send + Sync + 'static>(
                &self,
                client: &C,
                model: &str,
                input: Input,
            ) -> scaffold_runtime::Result<Output> {
                // Serialize input to include in the prompt
                let input_json = serde_json::to_string_pretty(&input)
                    .map_err(|e| scaffold_runtime::Error::SerializationError(e.to_string()))?;

                // Build output schema instructions
                let output_schema = Self::output_schema();
                let schema_str = serde_json::to_string_pretty(&output_schema)
                    .unwrap_or_default();

                // Build the full system prompt with output format instructions
                let full_preamble = format!(
                    "{}\n\nWhen you have completed the task, respond with ONLY valid JSON matching this schema (no markdown, no explanation):\n{}",
                    self.system_prompt,
                    schema_str
                );

                // Build rig agent with tools
                let completion_model = client.completion_model(model);
                let rig_agent = AgentBuilder::new(completion_model)
                    .preamble(&full_preamble)
                    #(#tool_registrations)*
                    .max_tokens(4096)
                    .build();

                // Execute the agent with the input
                let user_message = format!("Input: {}", input_json);
                let response = rig_agent.prompt(&user_message)
                    .await
                    .map_err(|e| scaffold_runtime::Error::Runtime(format!("Agent execution failed: {}", e)))?;

                // Parse the response as our output type
                let output: Output = serde_json::from_str(&response)
                    .map_err(|e| scaffold_runtime::Error::ParseError(
                        format!("Failed to parse agent output: {}. Response was: {}", e, response)
                    ))?;

                Ok(output)
            }
        }

        impl Default for #struct_name {
            fn default() -> Self {
                Self::new()
            }
        }

        /// Run the agent with the given input using environment-configured client
        pub async fn run(input: Input) -> scaffold_runtime::Result<Output> {
            use scaffold_runtime::rig::providers::openai;
            use scaffold_runtime::rig::client::ProviderClient;

            let agent = #struct_name::new();
            let client = openai::Client::from_env();
            let model = agent.model();
            agent.run_with(&client, model, input).await
        }
    }
}

/// Generate code for StringOrFileIR
fn gen_string_or_file(s: &StringOrFileIR) -> TokenStream {
    match s {
        StringOrFileIR::Literal { value } => {
            quote! { #value }
        }
        StringOrFileIR::File { path } => {
            // Include file at compile time
            quote! { include_str!(#path) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_ir::{ErrorStrategyIR, TypeIR};
    use std::collections::HashMap;

    #[test]
    fn test_gen_agents_mod() {
        let names = vec!["research_agent", "coding_agent"];
        let code = gen_agents_mod(&names).to_string();
        assert!(code.contains("research_agent"));
        assert!(code.contains("coding_agent"));
        assert!(code.contains("ResearchAgentAgent"));
        assert!(code.contains("CodingAgentAgent"));
    }

    #[test]
    fn test_gen_simple_agent() {
        let agent = AgentIR {
            name: "researcher".to_string(),
            input: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("query".to_string(), TypeIR::String);
                    f
                },
            },
            output: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("answer".to_string(), TypeIR::String);
                    f.insert(
                        "sources".to_string(),
                        TypeIR::List {
                            element: Box::new(TypeIR::String),
                        },
                    );
                    f
                },
            },
            tools: vec!["web_search".to_string(), "read_file".to_string()],
            system: StringOrFileIR::Literal {
                value: "You are a research assistant.".to_string(),
            },
            model: Some("gpt-4o".to_string()),
            max_turns: Some(5),
            reward: None,
            done: None,
            on_error: ErrorStrategyIR::default(),
            timeout: None,
        };

        let code = gen_agent_module(&agent).to_string();
        assert!(code.contains("ResearcherAgent"));
        assert!(code.contains("run_with"));
        assert!(code.contains("AgentBuilder"));
        assert!(code.contains("WebSearchTool"));
        assert!(code.contains("ReadFileTool"));
    }
}
