#![doc = "Prompt: summarize"]
use crate::types::*;
use scaffold_runtime::prelude::*;
use scaffold_runtime::rig::client::CompletionClient;
use scaffold_runtime::rig::extractor::Extractor;
use schemars::JsonSchema;
#[derive(
    Clone,
    Debug,
    Default,
    PartialEq,
    serde :: Serialize,
    serde :: Deserialize,
    schemars :: JsonSchema,
)]
pub struct Input {
    pub text: String,
}
#[derive(
    Clone,
    Debug,
    Default,
    PartialEq,
    serde :: Serialize,
    serde :: Deserialize,
    schemars :: JsonSchema,
)]
pub struct Output {
    pub summary: String,
}
#[doc = r" Prompt implementation struct"]
#[derive(Clone, Debug)]
pub struct SummarizePrompt {
    template: String,
}
impl SummarizePrompt {
    #[doc = r" Create a new prompt instance"]
    pub fn new() -> Self {
        Self {
            template: "Summarize briefly: {text}".to_string(),
        }
    }
    #[doc = r" Get the prompt name"]
    pub fn name(&self) -> &'static str {
        "summarize"
    }
    #[doc = r" Get the template string"]
    pub fn template(&self) -> &str {
        &self.template
    }
    #[doc = r" Get the JSON schema for the expected output (derived from schemars)"]
    pub fn output_schema() -> serde_json::Value {
        let schema = schemars::schema_for!(Output);
        serde_json::to_value(schema).unwrap_or_default()
    }
    #[doc = r" Execute the prompt using rig extractor for native structured output"]
    #[doc = r""]
    #[doc = r" The extractor uses the model's native JSON mode to ensure the output"]
    #[doc = r" conforms to the Output type's schema."]
    pub async fn execute<C: CompletionClient>(
        &self,
        client: &C,
        model: &str,
        input: Input,
    ) -> scaffold_runtime::Result<Output> {
        let rendered = self.render_template(&input)?;
        let extractor = client.extractor::<Output>(model).build();
        let output = extractor
            .extract(&rendered)
            .await
            .map_err(|e| scaffold_runtime::Error::Runtime(format!("Extraction failed: {}", e)))?;
        Ok(output)
    }
    #[doc = r" Render the template with the given input"]
    fn render_template(&self, input: &Input) -> scaffold_runtime::Result<String> {
        let input_json = serde_json::to_value(input)
            .map_err(|e| scaffold_runtime::Error::SerializationError(e.to_string()))?;
        let mut result = self.template.clone();
        if let serde_json::Value::Object(map) = input_json {
            for (key, value) in map {
                let placeholder = format!("{{{}}}", key);
                let replacement = match value {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                };
                result = result.replace(&placeholder, &replacement);
            }
        }
        Ok(result)
    }
}
impl Default for SummarizePrompt {
    fn default() -> Self {
        Self::new()
    }
}
#[doc = r" Run the prompt with the given input using environment-configured OpenAI client"]
pub async fn run(input: Input) -> scaffold_runtime::Result<Output> {
    use scaffold_runtime::rig::client::ProviderClient;
    use scaffold_runtime::rig::providers::openai;
    let prompt = SummarizePrompt::new();
    let client = openai::Client::from_env();
    let model = &scaffold_runtime::config().default_model;
    prompt.execute(&client, model, input).await
}
