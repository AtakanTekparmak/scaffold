//! Prompt code generation
//!
//! Generates Rust implementations for scaffold prompts (single LLM calls).
//! Prompts use rig extractors for native structured output with type enforcement.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use scaffold_ir::{PromptIR, StringOrFileIR, TypeIR};

use crate::types::gen_type;
use crate::util::{to_ident, to_pascal_case, to_snake_case};

/// Generate the prompts mod.rs
pub fn gen_prompts_mod(prompt_names: &[&str]) -> TokenStream {
    let mods: Vec<_> = prompt_names
        .iter()
        .map(|name| {
            let mod_name = format_ident!("{}", to_snake_case(name));
            quote! {
                pub mod #mod_name;
            }
        })
        .collect();

    let uses: Vec<_> = prompt_names
        .iter()
        .map(|name| {
            let mod_name = format_ident!("{}", to_snake_case(name));
            let struct_name = format_ident!("{}Prompt", to_pascal_case(name));
            quote! {
                pub use #mod_name::#struct_name;
            }
        })
        .collect();

    quote! {
        //! Prompt implementations

        #(#mods)*

        #(#uses)*
    }
}

/// Generate input/output type for prompts, handling inline structs
fn gen_io_type(ty: &TypeIR, type_name: &str) -> (TokenStream, TokenStream) {
    match ty {
        TypeIR::Struct { fields } if !fields.is_empty() => {
            let struct_ident = format_ident!("{}", type_name);
            let field_defs: Vec<_> = fields
                .iter()
                .map(|(name, field_ty)| {
                    let field_name = format_ident!("{}", to_ident(name));
                    let field_type = gen_type(field_ty);
                    quote! { pub #field_name: #field_type }
                })
                .collect();

            let struct_def = quote! {
                #[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
                pub struct #struct_ident {
                    #(#field_defs),*
                }
            };

            (struct_def, quote! { #struct_ident })
        }
        TypeIR::Struct { fields } if fields.is_empty() => (quote! {}, quote! { () }),
        other => {
            let ty = gen_type(other);
            (quote! {}, ty)
        }
    }
}

/// Generate a prompt module
pub fn gen_prompt_module(prompt: &PromptIR) -> TokenStream {
    let prompt_name = &prompt.name;
    let struct_name = format_ident!("{}Prompt", to_pascal_case(prompt_name));

    // Generate input/output types, creating inline structs if needed
    let (input_struct_def, input_type) = gen_io_type(&prompt.input, "Input");
    let (output_struct_def, output_type) = gen_io_type(&prompt.output, "Output");

    // Generate template string
    let template_str = gen_string_or_file(&prompt.template);

    // Generate system prompt if present
    let system_field = if prompt.system.is_some() {
        quote! {
            system: Option<String>,
        }
    } else {
        quote! {}
    };

    let system_init = match &prompt.system {
        Some(s) => {
            let system_str = gen_string_or_file(s);
            quote! { system: Some(#system_str.to_string()), }
        }
        None => quote! {},
    };

    let doc = format!("Prompt: {}", prompt_name);

    // Generate the extractor builder code with or without preamble
    let extractor_build = if prompt.system.is_some() {
        quote! {
            client.extractor::<Output>(model)
                .preamble(self.system.as_ref().unwrap())
                .build()
        }
    } else {
        quote! {
            client.extractor::<Output>(model).build()
        }
    };

    // Determine if we need type aliases (when struct def is empty, the type is already defined elsewhere)
    let input_needs_alias = input_struct_def.is_empty();
    let output_needs_alias = output_struct_def.is_empty();

    let input_type_decl = if input_needs_alias {
        quote! {
            /// Input type for this prompt
            pub type Input = #input_type;
        }
    } else {
        // Struct is defined inline, use it directly
        quote! {}
    };

    let output_type_decl = if output_needs_alias {
        quote! {
            /// Output type for this prompt
            pub type Output = #output_type;
        }
    } else {
        // Struct is defined inline, use it directly
        quote! {}
    };

    quote! {
        #![doc = #doc]

        use crate::types::*;
        use scaffold_runtime::prelude::*;
        use scaffold_runtime::rig::client::CompletionClient;
        use scaffold_runtime::rig::extractor::Extractor;
        use schemars::JsonSchema;

        #input_struct_def

        #output_struct_def

        #input_type_decl

        #output_type_decl

        /// Prompt implementation struct
        #[derive(Clone, Debug)]
        pub struct #struct_name {
            template: String,
            #system_field
        }

        impl #struct_name {
            /// Create a new prompt instance
            pub fn new() -> Self {
                Self {
                    template: #template_str.to_string(),
                    #system_init
                }
            }

            /// Get the prompt name
            pub fn name(&self) -> &'static str {
                #prompt_name
            }

            /// Get the template string
            pub fn template(&self) -> &str {
                &self.template
            }

            /// Get the JSON schema for the expected output (derived from schemars)
            pub fn output_schema() -> serde_json::Value {
                let schema = schemars::schema_for!(Output);
                serde_json::to_value(schema).unwrap_or_default()
            }

            /// Execute the prompt using rig extractor for native structured output
            ///
            /// The extractor uses the model's native JSON mode to ensure the output
            /// conforms to the Output type's schema.
            pub async fn execute<C: CompletionClient>(
                &self,
                client: &C,
                model: &str,
                input: Input,
            ) -> scaffold_runtime::Result<Output> {
                // Interpolate template with input
                let rendered = self.render_template(&input)?;

                // Build extractor with the output type's schema
                let extractor = #extractor_build;

                // Extract structured output using native API enforcement
                let output = extractor.extract(&rendered)
                    .await
                    .map_err(|e| scaffold_runtime::Error::Runtime(format!("Extraction failed: {}", e)))?;

                Ok(output)
            }

            /// Render the template with the given input
            fn render_template(&self, input: &Input) -> scaffold_runtime::Result<String> {
                let input_json = serde_json::to_value(input)
                    .map_err(|e| scaffold_runtime::Error::SerializationError(e.to_string()))?;

                let mut result = self.template.clone();

                // Simple template interpolation: replace {field} with input.field
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

        impl Default for #struct_name {
            fn default() -> Self {
                Self::new()
            }
        }

        /// Run the prompt with the given input using environment-configured OpenAI client
        pub async fn run(input: Input) -> scaffold_runtime::Result<Output> {
            use scaffold_runtime::rig::providers::openai;
            use scaffold_runtime::rig::client::ProviderClient;

            let prompt = #struct_name::new();
            let client = openai::Client::from_env();
            let model = &scaffold_runtime::config().default_model;
            prompt.execute(&client, model, input).await
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
    use scaffold_ir::TypeIR;
    use std::collections::HashMap;

    #[test]
    fn test_gen_prompts_mod() {
        let names = vec!["classify_text", "summarize"];
        let code = gen_prompts_mod(&names).to_string();
        assert!(code.contains("classify_text"));
        assert!(code.contains("summarize"));
        assert!(code.contains("ClassifyTextPrompt"));
        assert!(code.contains("SummarizePrompt"));
    }

    #[test]
    fn test_gen_simple_prompt() {
        let prompt = PromptIR {
            name: "classify".to_string(),
            input: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("text".to_string(), TypeIR::String);
                    f
                },
            },
            output: TypeIR::Struct {
                fields: {
                    let mut f = HashMap::new();
                    f.insert("category".to_string(), TypeIR::String);
                    f.insert("confidence".to_string(), TypeIR::Float);
                    f
                },
            },
            template: StringOrFileIR::Literal {
                value: "Classify the following text: {text}".to_string(),
            },
            system: None,
        };

        let code = gen_prompt_module(&prompt).to_string();
        assert!(code.contains("ClassifyPrompt"));
        assert!(code.contains("execute"));
        assert!(code.contains("output_schema"));
    }
}
