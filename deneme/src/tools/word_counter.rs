#![doc = "Tool: word_counter"]
use crate::types::*;
use scaffold_runtime::prelude::*;
use scaffold_runtime::rig::completion::request::ToolDefinition;
use scaffold_runtime::rig::tool::Tool;
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
    pub count: i64,
}
#[doc = r" Tool implementation struct"]
#[derive(Clone, Debug, Default)]
pub struct WordCounterTool {}
impl WordCounterTool {
    #[doc = r" Create a new tool instance"]
    pub fn new() -> Self {
        Self::default()
    }
    #[doc = r" Get the tool name"]
    pub fn name(&self) -> &'static str {
        "word_counter"
    }
    #[doc = r" Get the JSON schema for the input type"]
    pub fn input_schema() -> serde_json::Value {
        {
            let schema = schemars::schema_for!(());
            serde_json::to_value(schema).unwrap_or_default()
        }
    }
    #[doc = r" Execute the tool with the given input (synchronous)"]
    pub fn execute(&self, input: Input) -> scaffold_runtime::Result<Output> {
        let result = (|| -> scaffold_runtime::Result<Output> {
            Ok({
                let __out =
                    scaffold_runtime::shell::execute(&format!("echo '{}' | wc -w", input.text))?;
                serde_json::from_str::<Output>(__out.trim())
                    .map_err(|e| scaffold_runtime::Error::ParseError(e.to_string()))?
            })
        })();
        let _ = &result;
        result
    }
}
impl Tool for WordCounterTool {
    const NAME: &'static str = "word_counter";
    type Args = Input;
    type Output = Output;
    type Error = scaffold_runtime::ToolError;
    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "word_counter".to_string(),
            description: "Execute the word_counter tool".to_string(),
            parameters: Self::input_schema(),
        }
    }
    async fn call(&self, args: Self::Args) -> std::result::Result<Self::Output, Self::Error> {
        self.execute(args)
            .map_err(|e| scaffold_runtime::ToolError::from(e))
    }
}
#[doc = r" Run the tool with the given input (async wrapper for CLI compatibility)"]
pub async fn run(input: Input) -> scaffold_runtime::Result<Output> {
    let tool = WordCounterTool::new();
    tool.execute(input)
}
