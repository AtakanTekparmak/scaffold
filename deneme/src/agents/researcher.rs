#![doc = "Agent: researcher"]
use crate::types::*;
use scaffold_runtime::prelude::*;
use scaffold_runtime::rig::agent::AgentBuilder;
use scaffold_runtime::rig::client::CompletionClient;
use scaffold_runtime::rig::completion::Prompt;
use schemars::JsonSchema;
#[doc = r" Input type for this agent"]
pub type Input = ();
#[doc = r" Output type for this agent"]
pub type Output = ();
#[doc = r" Agent implementation struct"]
#[derive(Clone, Debug)]
pub struct ResearcherAgent {
    system_prompt: String,
    model: Option<String>,
    max_turns: u64,
}
impl ResearcherAgent {
    #[doc = r" Create a new agent instance"]
    pub fn new() -> Self {
        Self {
            system_prompt: "You are a research agent. Answer concisely.".to_string(),
            model: "openai/gpt-5.2".map(|s| s.to_string()),
            max_turns: 3u64,
        }
    }
    #[doc = r" Get the model to use (agent-specific or config default)"]
    pub fn model(&self) -> &str {
        self.model
            .as_deref()
            .unwrap_or(&scaffold_runtime::config().default_model)
    }
    #[doc = r" Get the agent name"]
    pub fn name(&self) -> &'static str {
        "researcher"
    }
    #[doc = r" Get the system prompt"]
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }
    #[doc = r" Get available tool names"]
    pub fn available_tools(&self) -> &'static [&'static str] {
        &["word_counter"]
    }
    #[doc = r" Get the JSON schema for the expected output (derived from schemars)"]
    pub fn output_schema() -> serde_json::Value {
        let schema = schemars::schema_for!(Output);
        serde_json::to_value(schema).unwrap_or_default()
    }
    #[doc = r" Execute the agent using rig's agent builder with native tool calling"]
    #[doc = r""]
    #[doc = r" This creates a rig agent with registered tools and executes the"]
    #[doc = r" agentic loop. The LLM will automatically call tools as needed and"]
    #[doc = r" return a final structured response."]
    pub async fn execute<C: CompletionClient + Send + Sync>(
        &self,
        client: &C,
        model: &str,
    ) -> impl FnOnce(
        Input,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = scaffold_runtime::Result<Output>> + Send + 'static>,
    >
    where
        C: Clone + 'static,
    {
        let system_prompt = self.system_prompt.clone();
        let max_turns = self.max_turns;
        let client = client.clone();
        let model = model.to_string();
        move |input: Input| {
            Box::pin(async move {
                let input_json = serde_json::to_string_pretty(&input)
                    .map_err(|e| scaffold_runtime::Error::SerializationError(e.to_string()))?;
                let output_schema = Self::output_schema();
                let schema_str = serde_json::to_string_pretty(&output_schema).unwrap_or_default();
                let full_preamble = format ! ("{}\n\nWhen you have completed the task, respond with ONLY valid JSON matching this schema (no markdown, no explanation):\n{}" , system_prompt , schema_str) ;
                let completion_model = client.completion_model(&model);
                let rig_agent = AgentBuilder::new(completion_model)
                    .preamble(&full_preamble)
                    .tool(crate::tools::WordCounterTool::new())
                    .max_tokens(4096)
                    .build();
                let user_message = format!("Input: {}", input_json);
                let response = rig_agent.prompt(&user_message).await.map_err(|e| {
                    scaffold_runtime::Error::Runtime(format!("Agent execution failed: {}", e))
                })?;
                let output: Output = serde_json::from_str(&response).map_err(|e| {
                    scaffold_runtime::Error::ParseError(format!(
                        "Failed to parse agent output: {}. Response was: {}",
                        e, response
                    ))
                })?;
                Ok(output)
            })
        }
    }
    #[doc = r" Execute the agent directly with input (convenience method)"]
    pub async fn run_with<C: CompletionClient + Clone + Send + Sync + 'static>(
        &self,
        client: &C,
        model: &str,
        input: Input,
    ) -> scaffold_runtime::Result<Output> {
        let input_json = serde_json::to_string_pretty(&input)
            .map_err(|e| scaffold_runtime::Error::SerializationError(e.to_string()))?;
        let output_schema = Self::output_schema();
        let schema_str = serde_json::to_string_pretty(&output_schema).unwrap_or_default();
        let full_preamble = format ! ("{}\n\nWhen you have completed the task, respond with ONLY valid JSON matching this schema (no markdown, no explanation):\n{}" , self . system_prompt , schema_str) ;
        let completion_model = client.completion_model(model);
        let rig_agent = AgentBuilder::new(completion_model)
            .preamble(&full_preamble)
            .tool(crate::tools::WordCounterTool::new())
            .max_tokens(4096)
            .build();
        let user_message = format!("Input: {}", input_json);
        let response = rig_agent.prompt(&user_message).await.map_err(|e| {
            scaffold_runtime::Error::Runtime(format!("Agent execution failed: {}", e))
        })?;
        let output: Output = serde_json::from_str(&response).map_err(|e| {
            scaffold_runtime::Error::ParseError(format!(
                "Failed to parse agent output: {}. Response was: {}",
                e, response
            ))
        })?;
        Ok(output)
    }
}
impl Default for ResearcherAgent {
    fn default() -> Self {
        Self::new()
    }
}
#[doc = r" Run the agent with the given input using environment-configured client"]
pub async fn run(input: Input) -> scaffold_runtime::Result<Output> {
    use scaffold_runtime::rig::client::ProviderClient;
    use scaffold_runtime::rig::providers::openai;
    let agent = ResearcherAgent::new();
    let client = openai::Client::from_env();
    let model = agent.model();
    agent.run_with(&client, model, input).await
}
