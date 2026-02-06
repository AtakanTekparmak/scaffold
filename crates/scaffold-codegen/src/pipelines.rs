//! Pipeline code generation
//!
//! Generates Rust implementations for scaffold pipelines (fixed sequences of prompts/tools).

use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};
use scaffold_ir::{
    AgentIR, PipelineCallIR, PipelineIR, PipelineStepIR, PromptIR, ToolExprIR, ToolIR, TypeDefIR,
};

use crate::types::gen_type;
use crate::util::{to_ident, to_pascal_case, to_snake_case};
use scaffold_ir::TypeIR;

/// Generate the pipelines mod.rs
pub fn gen_pipelines_mod(pipeline_names: &[&str]) -> TokenStream {
    let mods: Vec<_> = pipeline_names
        .iter()
        .map(|name| {
            let mod_name = format_ident!("{}", to_snake_case(name));
            quote! {
                pub mod #mod_name;
            }
        })
        .collect();

    let uses: Vec<_> = pipeline_names
        .iter()
        .map(|name| {
            let mod_name = format_ident!("{}", to_snake_case(name));
            let struct_name = format_ident!("{}Pipeline", to_pascal_case(name));
            quote! {
                pub use #mod_name::#struct_name;
            }
        })
        .collect();

    quote! {
        //! Pipeline implementations

        #(#mods)*

        #(#uses)*
    }
}

/// Generate input/output type for pipelines
fn gen_io_type(ty: &TypeIR, type_name: &str) -> (TokenStream, TokenStream, bool) {
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
                #[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
                pub struct #struct_ident {
                    #(#field_defs),*
                }
            };

            (struct_def, quote! { #struct_ident }, false)
        }
        TypeIR::Struct { fields } if fields.is_empty() => (quote! {}, quote! { () }, true),
        other => {
            let ty = gen_type(other);
            (quote! {}, ty, true)
        }
    }
}

/// Generate a pipeline module
pub fn gen_pipeline_module(
    pipeline: &PipelineIR,
    tools_index: &std::collections::HashMap<String, ToolIR>,
    prompts_index: &std::collections::HashMap<String, PromptIR>,
    agents_index: &std::collections::HashMap<String, AgentIR>,
    types_index: &std::collections::HashMap<String, TypeDefIR>,
) -> TokenStream {
    let pipeline_name = &pipeline.name;
    let struct_name = format_ident!("{}Pipeline", to_pascal_case(pipeline_name));

    // Generate input/output struct definitions
    let (input_struct_def, input_type, needs_input_alias) = gen_io_type(&pipeline.input, "Input");
    let (output_struct_def, output_type, needs_output_alias) =
        gen_io_type(&pipeline.output, "Output");

    // Generate type aliases only when needed
    let input_alias = if needs_input_alias {
        quote! { pub type Input = #input_type; }
    } else {
        quote! {}
    };
    let output_alias = if needs_output_alias {
        quote! { pub type Output = #output_type; }
    } else {
        quote! {}
    };

    // Extract input field names for distinguishing from local variables
    let input_fields: std::collections::HashSet<String> = match &pipeline.input {
        TypeIR::Struct { fields } => fields.keys().cloned().collect(),
        _ => std::collections::HashSet::new(),
    };

    // Generate step code
    let step_code = gen_pipeline_steps(
        &pipeline.steps,
        &input_fields,
        &pipeline.output,
        tools_index,
        prompts_index,
        agents_index,
        types_index,
    );

    let doc = format!("Pipeline: {}", pipeline_name);

    quote! {
        #![doc = #doc]

        use crate::types::*;
        use scaffold_runtime::prelude::*;

        #input_struct_def

        #output_struct_def

        #input_alias

        #output_alias

        /// Pipeline implementation struct
        #[derive(Clone, Debug, Default)]
        pub struct #struct_name {
            // Pipeline can hold configuration if needed
        }

        impl #struct_name {
            /// Create a new pipeline instance
            pub fn new() -> Self {
                Self::default()
            }

            /// Get the pipeline name
            pub fn name(&self) -> &'static str {
                #pipeline_name
            }

            /// Execute the pipeline with the given input
            pub async fn execute(&self, input: Input) -> scaffold_runtime::Result<Output> {
                use scaffold_runtime::rig::providers::openai;
                use scaffold_runtime::rig::client::ProviderClient;
                let model_id = scaffold_runtime::config().default_model.clone();
                let (provider, model_name) = scaffold_runtime::config::parse_model_id(&model_id);
                let openrouter_key = std::env::var("OPENROUTER_API_KEY")
                    .ok()
                    .or_else(|| scaffold_runtime::config().get_api_key("openrouter").map(|s| s.to_string()));
                let use_openrouter = openrouter_key.is_some() || provider == "openrouter";
                if use_openrouter {
                    if let Some(key) = openrouter_key.as_deref() {
                        std::env::set_var("OPENAI_API_KEY", key);
                    }
                    if let Some(base_url) = scaffold_runtime::config().get_base_url("openrouter") {
                        std::env::set_var("OPENAI_BASE_URL", base_url);
                    } else {
                        std::env::set_var("OPENAI_BASE_URL", "https://openrouter.ai/api/v1");
                    }
                }
                let client = openai::Client::from_env();
                let model = if use_openrouter && provider != "openrouter" {
                    model_id.as_str()
                } else {
                    model_name
                };
                #step_code
            }
        }

        /// Run the pipeline with the given input
        pub async fn run(input: Input) -> scaffold_runtime::Result<Output> {
            let pipeline = #struct_name::new();
            pipeline.execute(input).await
        }
    }
}

/// Resolve a type, looking up Named types in the types index
fn resolve_type<'a>(
    ty: &'a TypeIR,
    types_index: &'a std::collections::HashMap<String, TypeDefIR>,
) -> &'a TypeIR {
    match ty {
        TypeIR::Named { name } => {
            if let Some(type_def) = types_index.get(name) {
                &type_def.definition
            } else {
                ty
            }
        }
        _ => ty,
    }
}

/// Generate code for pipeline steps
fn gen_pipeline_steps(
    steps: &[PipelineStepIR],
    input_fields: &std::collections::HashSet<String>,
    output_type: &TypeIR,
    tools_index: &std::collections::HashMap<String, ToolIR>,
    prompts_index: &std::collections::HashMap<String, PromptIR>,
    agents_index: &std::collections::HashMap<String, AgentIR>,
    types_index: &std::collections::HashMap<String, TypeDefIR>,
) -> TokenStream {
    let empty_locals: std::collections::HashSet<String> = std::collections::HashSet::new();
    let (step_code, last_expr, local_vars) = gen_pipeline_steps_with_locals(
        steps,
        input_fields,
        tools_index,
        prompts_index,
        agents_index,
        &empty_locals,
    );

    // The last step's result should be returned
    if steps.is_empty() {
        quote! { Ok(Default::default()) }
    } else {
        // Resolve the output type (in case it's a Named reference)
        let resolved_output = resolve_type(output_type, types_index);

        // If pipeline output is a struct with fields, try to construct it from bindings
        // Only use fields that have matching bindings
        if let TypeIR::Struct { fields } = resolved_output {
            let matching_fields: Vec<_> =
                fields.keys().filter(|k| local_vars.contains(*k)).collect();

            if matching_fields.len() == fields.len() {
                // All output fields have matching bindings
                let field_inits: Vec<_> = fields
                    .keys()
                    .map(|k| {
                        let ident = format_ident!("{}", to_ident(k));
                        quote! { #ident: #ident }
                    })
                    .collect();
                return quote! {
                    #step_code
                    let __output = Output { #(#field_inits),* };
                    Ok(__output)
                };
            }
        }
        // Fallback: return last step result
        let fallback = last_expr.unwrap_or_else(|| quote! { Default::default() });
        quote! { #step_code Ok(#fallback) }
    }
}

fn gen_pipeline_steps_with_locals(
    steps: &[PipelineStepIR],
    input_fields: &std::collections::HashSet<String>,
    tools_index: &std::collections::HashMap<String, ToolIR>,
    prompts_index: &std::collections::HashMap<String, PromptIR>,
    agents_index: &std::collections::HashMap<String, AgentIR>,
    initial_locals: &std::collections::HashSet<String>,
) -> (
    TokenStream,
    Option<TokenStream>,
    std::collections::HashSet<String>,
) {
    // Track local variables from step bindings
    let mut local_vars: std::collections::HashSet<String> = initial_locals.clone();
    let mut step_codes: Vec<TokenStream> = Vec::new();
    let mut last_expr: Option<TokenStream> = None;

    for (idx, step) in steps.iter().enumerate() {
        let call_code = gen_pipeline_call(
            &step.call,
            input_fields,
            &local_vars,
            tools_index,
            prompts_index,
            agents_index,
        );

        let (binding_ident, binding_name) = match &step.binding {
            Some(name) => (format_ident!("{}", to_ident(name)), name.clone()),
            None => (format_ident!("__step_{}", idx), format!("__step_{}", idx)),
        };

        step_codes.push(quote! {
            let #binding_ident = #call_code;
        });

        local_vars.insert(binding_name);
        last_expr = Some(quote! { #binding_ident });
    }

    (quote! { #(#step_codes)* }, last_expr, local_vars)
}

/// Generate code for a pipeline call
fn gen_pipeline_call(
    call: &PipelineCallIR,
    input_fields: &std::collections::HashSet<String>,
    local_vars: &std::collections::HashSet<String>,
    tools_index: &std::collections::HashMap<String, ToolIR>,
    prompts_index: &std::collections::HashMap<String, PromptIR>,
    agents_index: &std::collections::HashMap<String, AgentIR>,
) -> TokenStream {
    match call {
        PipelineCallIR::Prompt { name, args } => {
            let prompt_mod = format_ident!("{}", to_snake_case(name));
            let prompt_struct = format_ident!("{}Prompt", to_pascal_case(name));
            let arg_codes: Vec<_> = args
                .iter()
                .map(|a| gen_tool_expr(a, input_fields, local_vars))
                .collect();

            // Look up the prompt to determine its input type
            if let Some(prompt_ir) = prompts_index.get(name) {
                match &prompt_ir.input {
                    TypeIR::Struct { fields } if !fields.is_empty() => {
                        // Construct the prompt's Input struct
                        let field_inits: Vec<_> = fields
                            .keys()
                            .enumerate()
                            .map(|(i, k)| {
                                let fname = format_ident!("{}", to_ident(k));
                                let val = arg_codes
                                    .get(i)
                                    .cloned()
                                    .unwrap_or(quote! { Default::default() });
                                quote! { #fname: #val }
                            })
                            .collect();
                        quote! {
                            {
                                let __prompt_input = crate::prompts::#prompt_mod::Input { #(#field_inits),* };
                                crate::prompts::#prompt_struct::new().execute(&client, model, __prompt_input).await?
                            }
                        }
                    }
                    _ => {
                        // Simple input types
                        if args.is_empty() {
                            quote! { crate::prompts::#prompt_struct::new().execute(&client, model, ()).await? }
                        } else if args.len() == 1 {
                            let arg = &arg_codes[0];
                            quote! { crate::prompts::#prompt_struct::new().execute(&client, model, #arg).await? }
                        } else {
                            quote! { crate::prompts::#prompt_struct::new().execute(&client, model, (#(#arg_codes),*)).await? }
                        }
                    }
                }
            } else {
                // Prompt not in index, fall back to direct passing
                if args.is_empty() {
                    quote! { crate::prompts::#prompt_struct::new().execute(&client, model, ()).await? }
                } else if args.len() == 1 {
                    let arg = &arg_codes[0];
                    quote! { crate::prompts::#prompt_struct::new().execute(&client, model, #arg).await? }
                } else {
                    quote! { crate::prompts::#prompt_struct::new().execute(&client, model, (#(#arg_codes),*)).await? }
                }
            }
        }
        PipelineCallIR::Tool { name, args } => {
            let tool_struct = format_ident!("{}Tool", to_pascal_case(name));
            let tool_mod = format_ident!("{}", to_snake_case(name));
            let arg_codes: Vec<_> = args
                .iter()
                .map(|a| gen_tool_expr(a, input_fields, local_vars))
                .collect();

            // Look up the tool to determine its input type
            if let Some(tool_ir) = tools_index.get(name) {
                match &tool_ir.input {
                    TypeIR::Struct { fields } if !fields.is_empty() => {
                        // Construct the tool's Input struct
                        let field_inits: Vec<_> = fields
                            .keys()
                            .enumerate()
                            .map(|(i, k)| {
                                let fname = format_ident!("{}", to_ident(k));
                                let val = arg_codes
                                    .get(i)
                                    .cloned()
                                    .unwrap_or(quote! { Default::default() });
                                quote! { #fname: #val }
                            })
                            .collect();
                        quote! {
                            {
                                let __tool_input = crate::tools::#tool_mod::Input { #(#field_inits),* };
                                crate::tools::#tool_struct::new().execute(__tool_input)?
                            }
                        }
                    }
                    TypeIR::Struct { fields } if fields.is_empty() => {
                        quote! {
                            {
                                let __tool_input = crate::tools::#tool_mod::Input {};
                                crate::tools::#tool_struct::new().execute(__tool_input)?
                            }
                        }
                    }
                    _ => {
                        // Simple input types
                        if args.is_empty() {
                            quote! { crate::tools::#tool_struct::new().execute(())? }
                        } else if args.len() == 1 {
                            let arg = &arg_codes[0];
                            quote! { crate::tools::#tool_struct::new().execute(#arg)? }
                        } else {
                            quote! { crate::tools::#tool_struct::new().execute((#(#arg_codes),*))? }
                        }
                    }
                }
            } else {
                // Tool not in index, fall back to direct passing
                if args.is_empty() {
                    quote! { crate::tools::#tool_struct::new().execute(())? }
                } else if args.len() == 1 {
                    let arg = &arg_codes[0];
                    quote! { crate::tools::#tool_struct::new().execute(#arg)? }
                } else {
                    quote! { crate::tools::#tool_struct::new().execute((#(#arg_codes),*))? }
                }
            }
        }
        PipelineCallIR::Agent { name, args } => {
            let agent_mod = format_ident!("{}", to_snake_case(name));
            let agent_struct = format_ident!("{}Agent", to_pascal_case(name));
            let arg_codes: Vec<_> = args
                .iter()
                .map(|a| gen_tool_expr(a, input_fields, local_vars))
                .collect();

            if let Some(agent_ir) = agents_index.get(name) {
                match &agent_ir.input {
                    TypeIR::Struct { fields } if !fields.is_empty() => {
                        let field_inits: Vec<_> = fields
                            .keys()
                            .enumerate()
                            .map(|(i, k)| {
                                let fname = format_ident!("{}", to_ident(k));
                                let val = arg_codes
                                    .get(i)
                                    .cloned()
                                    .unwrap_or(quote! { Default::default() });
                                quote! { #fname: #val }
                            })
                            .collect();
                        quote! {
                            {
                                let __agent_input = crate::agents::#agent_mod::Input { #(#field_inits),* };
                                crate::agents::#agent_struct::new().run_with(&client, model, __agent_input).await?
                            }
                        }
                    }
                    _ => {
                        if args.is_empty() {
                            quote! { crate::agents::#agent_struct::new().run_with(&client, model, ()).await? }
                        } else if args.len() == 1 {
                            let arg = &arg_codes[0];
                            quote! { crate::agents::#agent_struct::new().run_with(&client, model, #arg).await? }
                        } else {
                            quote! { crate::agents::#agent_struct::new().run_with(&client, model, (#(#arg_codes),*)).await? }
                        }
                    }
                }
            } else if args.is_empty() {
                quote! { crate::agents::#agent_struct::new().run_with(&client, model, ()).await? }
            } else if args.len() == 1 {
                let arg = &arg_codes[0];
                quote! { crate::agents::#agent_struct::new().run_with(&client, model, #arg).await? }
            } else {
                quote! { crate::agents::#agent_struct::new().run_with(&client, model, (#(#arg_codes),*)).await? }
            }
        }
        PipelineCallIR::Expr { expr } => {
            // Generate code directly from the expression
            let code = gen_tool_expr(expr, input_fields, local_vars);
            quote! { #code }
        }
        PipelineCallIR::If {
            condition,
            then_steps,
            else_steps,
        } => {
            let cond_code = gen_pipeline_expr(condition, input_fields, local_vars);
            let (then_code, then_last, _) = gen_pipeline_steps_with_locals(
                then_steps,
                input_fields,
                tools_index,
                prompts_index,
                agents_index,
                local_vars,
            );
            let (else_code, else_last, _) = gen_pipeline_steps_with_locals(
                else_steps,
                input_fields,
                tools_index,
                prompts_index,
                agents_index,
                local_vars,
            );
            let then_tail = then_last.unwrap_or_else(|| quote! { Default::default() });
            let else_tail = else_last.unwrap_or_else(|| quote! { Default::default() });
            quote! {
                {
                    if #cond_code {
                        #then_code
                        #then_tail
                    } else {
                        #else_code
                        #else_tail
                    }
                }
            }
        }
        PipelineCallIR::Match { scrutinee, arms } => {
            let scrut_code = gen_pipeline_expr(scrutinee, input_fields, local_vars);
            if arms.is_empty() {
                return quote! { Default::default() };
            }
            let mut clauses: Vec<TokenStream> = Vec::new();
            for (idx, arm) in arms.iter().enumerate() {
                let pat = gen_pipeline_expr(&arm.pattern, input_fields, local_vars);
                let (arm_code, arm_last, _) = gen_pipeline_steps_with_locals(
                    &arm.steps,
                    input_fields,
                    tools_index,
                    prompts_index,
                    agents_index,
                    local_vars,
                );
                let arm_tail = arm_last.unwrap_or_else(|| quote! { Default::default() });
                let clause = if idx == 0 {
                    quote! {
                        if #scrut_code == #pat {
                            #arm_code
                            #arm_tail
                        }
                    }
                } else {
                    quote! {
                        else if #scrut_code == #pat {
                            #arm_code
                            #arm_tail
                        }
                    }
                };
                clauses.push(clause);
            }
            quote! {
                {
                    #(#clauses)*
                    else {
                        Default::default()
                    }
                }
            }
        }
        PipelineCallIR::Parallel { branches } => {
            if branches.is_empty() {
                return quote! { () };
            }
            let mut branch_blocks: Vec<TokenStream> = Vec::new();
            let mut branch_idents: Vec<proc_macro2::Ident> = Vec::new();

            for (idx, branch) in branches.iter().enumerate() {
                let (branch_code, _branch_last, _) = gen_pipeline_steps_with_locals(
                    branch,
                    input_fields,
                    tools_index,
                    prompts_index,
                    agents_index,
                    local_vars,
                );
                let ident = format_ident!("__branch_{}", idx);
                branch_idents.push(ident);
                branch_blocks.push(quote! {
                    async {
                        #branch_code
                        Ok::<(), scaffold_runtime::Error>(())
                    }
                });
            }

            quote! {
                {
                    let (#(#branch_idents),*) = tokio::join!(#(#branch_blocks),*);
                    #( #branch_idents?; )*
                    ()
                }
            }
        }
    }
}

/// Generate a tool expression for pipeline steps
/// Input field names are prefixed with "input.", local variables are used directly
fn gen_tool_expr(
    expr: &ToolExprIR,
    input_fields: &std::collections::HashSet<String>,
    local_vars: &std::collections::HashSet<String>,
) -> TokenStream {
    match expr {
        ToolExprIR::Ident { name } => {
            let ident = format_ident!("{}", to_ident(name));
            if local_vars.contains(name) {
                // Local variable from previous step
                quote! { #ident.clone() }
            } else if input_fields.contains(name) {
                // Input field
                quote! { input.#ident.clone() }
            } else {
                // Unknown - assume it's a local variable (might be defined later)
                quote! { #ident.clone() }
            }
        }
        ToolExprIR::FieldAccess { base, field } => {
            let base_code = gen_tool_expr(base, input_fields, local_vars);
            let field_ident = format_ident!("{}", to_ident(field));
            quote! { #base_code.#field_ident.clone() }
        }
        ToolExprIR::MapLiteral { entries } => {
            let entry_tokens: Vec<_> = entries
                .iter()
                .map(|entry| {
                    let key_lit = Literal::string(&entry.key);
                    let val_code = gen_tool_expr(&entry.value, input_fields, local_vars);
                    quote! { #key_lit: #val_code }
                })
                .collect();
            quote! { serde_json::json!({ #(#entry_tokens),* }) }
        }
        ToolExprIR::Literal { value } => gen_literal(value),
        ToolExprIR::Expr { expr } => gen_pipeline_expr(expr, input_fields, local_vars),
        _ => {
            // For complex expressions, fall back to a placeholder
            quote! { Default::default() }
        }
    }
}

/// Generate expression for pipeline context (input fields are prefixed)
fn gen_pipeline_expr(
    expr: &scaffold_ir::ExprIR,
    input_fields: &std::collections::HashSet<String>,
    local_vars: &std::collections::HashSet<String>,
) -> TokenStream {
    match expr {
        scaffold_ir::ExprIR::Ident { name } => {
            let ident = format_ident!("{}", to_ident(name));
            if local_vars.contains(name) {
                quote! { #ident.clone() }
            } else if input_fields.contains(name) {
                quote! { input.#ident.clone() }
            } else {
                quote! { #ident.clone() }
            }
        }
        scaffold_ir::ExprIR::FieldAccess { base, field } => {
            let base_code = gen_pipeline_expr(base, input_fields, local_vars);
            let field_ident = format_ident!("{}", to_ident(field));
            quote! { #base_code.#field_ident }
        }
        scaffold_ir::ExprIR::Literal { value } => gen_literal(value),
        scaffold_ir::ExprIR::Binary { left, op, right } => {
            let left_code = gen_pipeline_expr(left, input_fields, local_vars);
            let right_code = gen_pipeline_expr(right, input_fields, local_vars);
            match op.as_str() {
                "==" => quote! { (#left_code == #right_code) },
                "!=" => quote! { (#left_code != #right_code) },
                "<" => quote! { (#left_code < #right_code) },
                "<=" => quote! { (#left_code <= #right_code) },
                ">" => quote! { (#left_code > #right_code) },
                ">=" => quote! { (#left_code >= #right_code) },
                "&&" => quote! { (#left_code && #right_code) },
                "||" => quote! { (#left_code || #right_code) },
                "+" => quote! { (#left_code + #right_code) },
                "-" => quote! { (#left_code - #right_code) },
                "*" => quote! { (#left_code * #right_code) },
                "/" => quote! { (#left_code / #right_code) },
                _ => quote! { #left_code },
            }
        }
        scaffold_ir::ExprIR::Call { function, args } => {
            let func = format_ident!("{}", function);
            let arg_codes: Vec<_> = args
                .iter()
                .map(|a| gen_pipeline_expr(a, input_fields, local_vars))
                .collect();
            quote! { #func(#(#arg_codes),*) }
        }
        scaffold_ir::ExprIR::ForeignCall {
            module,
            function,
            args,
        } => {
            let module_ident = format_ident!("{}", module);
            let func_ident = format_ident!("{}", function);
            let arg_codes: Vec<_> = args
                .iter()
                .map(|a| gen_pipeline_expr(a, input_fields, local_vars))
                .collect();
            quote! { #module_ident::#func_ident(#(#arg_codes),*) }
        }
    }
}

/// Generate a literal value
fn gen_literal(lit: &scaffold_ir::LiteralIR) -> TokenStream {
    use scaffold_ir::LiteralIR;
    match lit {
        LiteralIR::Int { value } => quote! { #value },
        LiteralIR::Float { value } => quote! { #value },
        LiteralIR::String { value } => quote! { #value.to_string() },
        LiteralIR::Bool { value } => quote! { #value },
        LiteralIR::Null => quote! { None },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_ir::TypeIR;
    use std::collections::HashMap;

    #[test]
    fn test_gen_pipelines_mod() {
        let names = vec!["extract_and_summarize", "process_data"];
        let code = gen_pipelines_mod(&names).to_string();
        assert!(code.contains("extract_and_summarize"));
        assert!(code.contains("process_data"));
        assert!(code.contains("ExtractAndSummarizePipeline"));
        assert!(code.contains("ProcessDataPipeline"));
    }

    #[test]
    fn test_gen_simple_pipeline() {
        let pipeline = PipelineIR {
            name: "analyze".to_string(),
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
                    f.insert("summary".to_string(), TypeIR::String);
                    f
                },
            },
            steps: vec![
                PipelineStepIR {
                    binding: Some("extracted".to_string()),
                    call: PipelineCallIR::Tool {
                        name: "extract".to_string(),
                        args: vec![ToolExprIR::Ident {
                            name: "input".to_string(),
                        }],
                    },
                },
                PipelineStepIR {
                    binding: Some("result".to_string()),
                    call: PipelineCallIR::Prompt {
                        name: "summarize".to_string(),
                        args: vec![ToolExprIR::Ident {
                            name: "extracted".to_string(),
                        }],
                    },
                },
            ],
            reward: None,
        };

        let tools_index: std::collections::HashMap<String, ToolIR> =
            std::collections::HashMap::new();
        let prompts_index: std::collections::HashMap<String, PromptIR> =
            std::collections::HashMap::new();
        let agents_index: std::collections::HashMap<String, AgentIR> =
            std::collections::HashMap::new();
        let types_index: std::collections::HashMap<String, TypeDefIR> =
            std::collections::HashMap::new();
        let code = gen_pipeline_module(
            &pipeline,
            &tools_index,
            &prompts_index,
            &agents_index,
            &types_index,
        )
        .to_string();
        assert!(code.contains("AnalyzePipeline"));
        assert!(code.contains("execute"));
        assert!(code.contains("extracted"));
        assert!(code.contains("result"));
    }
}
