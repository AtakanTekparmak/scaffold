//! Tool code generation
//!
//! Generates Rust implementations for scaffold tools.
//! Tools implement the rig `Tool` trait for LLM tool calling.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use scaffold_ir::{
    LiteralIR, ToolExprIR, ToolIR, ToolImplIR, ToolSpecIR, ToolStatementIR, ToolVariantIR, TypeIR,
};

use crate::expr::gen_expr;
use crate::types::gen_type;
use crate::util::{to_ident, to_pascal_case, to_snake_case};

/// Generate input/output type - either use existing type or generate inline struct
use scaffold_ir::ExprIR;

/// Parse shell command template and generate code with variable interpolation
/// Returns (generated_code, has_variables)
fn interpolate_shell_command(command: &str) -> (TokenStream, bool) {
    use regex::Regex;

    // Find all {variable} patterns
    let re = Regex::new(r"\{(\w+)\}").unwrap();
    let mut vars: Vec<String> = Vec::new();

    for cap in re.captures_iter(command) {
        let var_name = cap[1].to_string();
        if !vars.contains(&var_name) {
            vars.push(var_name);
        }
    }

    if vars.is_empty() {
        return (quote! {}, false);
    }

    // Replace {var} with {} for format! macro
    let format_str = re.replace_all(command, "{}").to_string();

    // Generate the variable accessors
    let var_accesses: Vec<_> = vars
        .iter()
        .map(|v| {
            let var_ident = format_ident!("{}", v);
            quote! { input.#var_ident }
        })
        .collect();

    let code = quote! {
        scaffold_runtime::shell::execute(&format!(#format_str, #(#var_accesses),*))?
    };

    (code, true)
}

/// Generate expression, treating function calls as potential tool calls
fn gen_expr_or_tool_call(
    expr: &ExprIR,
    tools_index: &std::collections::HashMap<String, ToolIR>,
    input_fields: &std::collections::HashSet<String>,
    local_vars: &std::collections::HashSet<String>,
) -> TokenStream {
    match expr {
        ExprIR::Call { function, args } => {
            // Treat as a tool call - generate proper tool invocation
            let tool_struct = format_ident!("{}Tool", to_pascal_case(function));
            let tool_mod = format_ident!("{}", to_snake_case(function));
            let arg_codes: Vec<_> = args
                .iter()
                .map(|arg| gen_expr_or_tool_call(arg, tools_index, input_fields, local_vars))
                .collect();

            // Look up the tool to determine its input type
            if let Some(tool_ir) = tools_index.get(function) {
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
                                let __call_input = crate::tools::#tool_mod::Input { #(#field_inits),* };
                                crate::tools::#tool_struct::new().execute(__call_input)?
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
        ExprIR::FieldAccess { base, field } => {
            let base_code = gen_expr_or_tool_call(base, tools_index, input_fields, local_vars);
            let field_ident = format_ident!("{}", to_ident(field));
            quote! { #base_code.#field_ident }
        }
        ExprIR::Ident { name } => {
            // Handle identifiers - check if it's an input field or local variable
            let ident = format_ident!("{}", name);
            if local_vars.contains(name) {
                quote! { #ident }
            } else if input_fields.contains(name) {
                quote! { input.#ident }
            } else {
                quote! { #ident }
            }
        }
        ExprIR::Binary { left, op, right } => {
            let left_code = gen_expr_or_tool_call(left, tools_index, input_fields, local_vars);
            let right_code = gen_expr_or_tool_call(right, tools_index, input_fields, local_vars);
            match op.as_str() {
                "==" => quote! { (#left_code == #right_code) },
                "!=" => quote! { (#left_code != #right_code) },
                "<" => quote! { (#left_code < #right_code) },
                "<=" => quote! { (#left_code <= #right_code) },
                ">" => quote! { (#left_code > #right_code) },
                ">=" => quote! { (#left_code >= #right_code) },
                "&&" | "and" => quote! { (#left_code && #right_code) },
                "||" | "or" => quote! { (#left_code || #right_code) },
                "+" => quote! { (#left_code + #right_code) },
                "-" => quote! { (#left_code - #right_code) },
                "*" => quote! { (#left_code * #right_code) },
                "/" => quote! { (#left_code / #right_code) },
                "%" => quote! { (#left_code % #right_code) },
                other => {
                    let method = format_ident!("{}", to_ident(other));
                    quote! { #left_code.#method(#right_code) }
                }
            }
        }
        // For other expression types, delegate to gen_expr
        _ => gen_expr(expr),
    }
}

/// Generate input/output type - either use existing type or generate inline struct
/// Returns (struct_definition, type_token, needs_type_alias)
fn gen_io_type(ty: &TypeIR, type_name: &str) -> (TokenStream, TokenStream, bool) {
    match ty {
        TypeIR::Struct { fields } if !fields.is_empty() => {
            // Generate an inline struct definition
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

            // Struct was generated, no type alias needed
            (struct_def, quote! { #struct_ident }, false)
        }
        TypeIR::Struct { fields } if fields.is_empty() => {
            // Empty struct -> use unit type, need type alias
            (quote! {}, quote! { () }, true)
        }
        other => {
            // Use existing type directly, need type alias
            let ty = gen_type(other);
            (quote! {}, ty, true)
        }
    }
}

/// Generate the tools mod.rs
pub fn gen_tools_mod(tool_names: &[&str]) -> TokenStream {
    let mods: Vec<_> = tool_names
        .iter()
        .map(|name| {
            let mod_name = format_ident!("{}", to_snake_case(name));
            quote! {
                pub mod #mod_name;
            }
        })
        .collect();

    let uses: Vec<_> = tool_names
        .iter()
        .map(|name| {
            let mod_name = format_ident!("{}", to_snake_case(name));
            let struct_name = format_ident!("{}Tool", to_pascal_case(name));
            quote! {
                pub use #mod_name::#struct_name;
            }
        })
        .collect();

    quote! {
        //! Tool implementations

        #(#mods)*

        #(#uses)*
    }
}

/// Generate a tool module
pub fn gen_tool_module(
    tool: &ToolIR,
    tools_index: &std::collections::HashMap<String, ToolIR>,
) -> TokenStream {
    let tool_name = &tool.name;
    let struct_name = format_ident!("{}Tool", to_pascal_case(tool_name));

    // Generate input/output struct definitions if they are anonymous structs
    let (input_struct_def, input_type, needs_input_alias) = gen_io_type(&tool.input, "Input");
    let (output_struct_def, output_type, needs_output_alias) = gen_io_type(&tool.output, "Output");

    // Extract input field names for variable tracking
    let input_fields: std::collections::HashSet<String> = match &tool.input {
        TypeIR::Struct { fields } => fields.keys().cloned().collect(),
        _ => std::collections::HashSet::new(),
    };
    let empty_locals: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Generate the main execute method (with expected output type context)
    let execute_body = match &tool.implementation {
        Some(impl_) => gen_tool_impl_with_ctx(
            impl_,
            Some(&tool.output),
            tools_index,
            &input_fields,
            &empty_locals,
        ),
        None => quote! { todo!("Tool implementation not provided") },
    };

    // Generate spec validation if present
    let (pre_check, post_check, pure_attr) = match &tool.spec {
        Some(spec) => gen_spec_checks(spec),
        None => (quote! {}, quote! { let _ = &result; }, quote! {}),
    };

    // Generate variants as separate methods
    let variant_methods: Vec<_> = tool
        .variants
        .iter()
        .map(|v| gen_variant_method(v, &input_type, &output_type, tools_index, &input_fields))
        .collect();

    // Generate variant enum if there are variants
    let variant_enum = if !tool.variants.is_empty() {
        let variant_names: Vec<_> = tool
            .variants
            .iter()
            .map(|v| format_ident!("{}", to_pascal_case(&v.name)))
            .collect();

        quote! {
            /// Available implementation variants
            #[derive(Clone, Copy, Debug, Default)]
            pub enum Variant {
                #[default]
                Default,
                #(#variant_names,)*
            }
        }
    } else {
        quote! {}
    };

    // Generate execute_variant method if there are variants
    let execute_variant = if !tool.variants.is_empty() {
        let variant_arms: Vec<_> = tool
            .variants
            .iter()
            .map(|v| {
                let variant_name = format_ident!("{}", to_pascal_case(&v.name));
                let method_name = format_ident!("execute_{}", to_snake_case(&v.name));
                quote! {
                    Variant::#variant_name => self.#method_name(input),
                }
            })
            .collect();

        quote! {
            /// Execute with a specific variant
            pub fn execute_variant(&self, input: #input_type, variant: Variant) -> scaffold_runtime::Result<#output_type> {
                match variant {
                    Variant::Default => self.execute(input),
                    #(#variant_arms)*
                }
            }
        }
    } else {
        quote! {}
    };

    let doc = format!("Tool: {}", tool_name);

    // Generate the tool description
    let tool_description = format!("Execute the {} tool", tool_name);

    // Generate JSON schema for the input type (using schemars at compile time)
    let input_schema_code = gen_input_schema_fn(&tool.input);

    // Generate type aliases only when needed (when no struct was generated)
    let input_alias = if needs_input_alias {
        quote! {
            /// Input type alias for this tool
            pub type Input = #input_type;
        }
    } else {
        quote! {}
    };
    let output_alias = if needs_output_alias {
        quote! {
            /// Output type alias for this tool
            pub type Output = #output_type;
        }
    } else {
        quote! {}
    };

    quote! {
        #![doc = #doc]

        use crate::types::*;
        use scaffold_runtime::prelude::*;
        use scaffold_runtime::rig::tool::Tool;
        use scaffold_runtime::rig::completion::request::ToolDefinition;
        use schemars::JsonSchema;

        #input_struct_def

        #output_struct_def

        #input_alias

        #output_alias

        #variant_enum

        /// Tool implementation struct
        #[derive(Clone, Debug, Default)]
        pub struct #struct_name {
            // Tool can hold configuration or state if needed
        }

        impl #struct_name {
            /// Create a new tool instance
            pub fn new() -> Self {
                Self::default()
            }

            /// Get the tool name
            pub fn name(&self) -> &'static str {
                #tool_name
            }

            /// Get the JSON schema for the input type
            pub fn input_schema() -> serde_json::Value {
                #input_schema_code
            }

            /// Execute the tool with the given input (synchronous)
            #pure_attr
            pub fn execute(&self, input: Input) -> scaffold_runtime::Result<Output> {
                #pre_check
                let result = (|| -> scaffold_runtime::Result<Output> {
                    #execute_body
                })();
                #post_check
                result
            }

            #execute_variant

            #(#variant_methods)*
        }

        // Implement rig's Tool trait for LLM tool calling
        impl Tool for #struct_name {
            const NAME: &'static str = #tool_name;

            type Args = Input;
            type Output = Output;
            type Error = scaffold_runtime::ToolError;

            async fn definition(&self, _prompt: String) -> ToolDefinition {
                ToolDefinition {
                    name: #tool_name.to_string(),
                    description: #tool_description.to_string(),
                    parameters: Self::input_schema(),
                }
            }

            async fn call(&self, args: Self::Args) -> std::result::Result<Self::Output, Self::Error> {
                self.execute(args).map_err(|e| scaffold_runtime::ToolError::from(e))
            }
        }

        /// Run the tool with the given input (async wrapper for CLI compatibility)
        pub async fn run(input: Input) -> scaffold_runtime::Result<Output> {
            let tool = #struct_name::new();
            tool.execute(input)
        }
    }
}

/// Generate the body for a tool implementation
fn gen_tool_impl_with_ctx(
    impl_: &ToolImplIR,
    expected: Option<&TypeIR>,
    tools_index: &std::collections::HashMap<String, ToolIR>,
    input_fields: &std::collections::HashSet<String>,
    local_vars: &std::collections::HashSet<String>,
) -> TokenStream {
    match impl_ {
        ToolImplIR::Expr { expr } => {
            let expr_code =
                gen_tool_expr_with_ctx(expr, expected, tools_index, input_fields, local_vars);
            quote! { Ok(#expr_code) }
        }
        ToolImplIR::Sequence { statements } => {
            // Track local variables incrementally
            let mut current_locals = local_vars.clone();
            let mut stmts: Vec<TokenStream> = Vec::new();

            for stmt in statements {
                let stmt_code =
                    gen_tool_statement_with_ctx(stmt, tools_index, input_fields, &current_locals);
                stmts.push(stmt_code);
                // Add binding to local vars for subsequent statements
                if let Some(name) = &stmt.binding {
                    current_locals.insert(name.clone());
                }
            }

            // Collect all binding names for output construction
            let bindings: std::collections::HashSet<String> = statements
                .iter()
                .filter_map(|s| s.binding.clone())
                .collect();

            // Only construct output if ALL field names have matching bindings
            if let Some(TypeIR::Struct { fields }) = expected {
                if !fields.is_empty() {
                    let all_match = fields.keys().all(|k| bindings.contains(k));
                    if all_match {
                        let field_inits: Vec<_> = fields
                            .keys()
                            .map(|k| {
                                let ident = format_ident!("{}", to_ident(k));
                                quote! { #ident: #ident }
                            })
                            .collect();
                        return quote! {
                            #(#stmts)*
                            let __output = Output { #(#field_inits),* };
                            Ok(__output)
                        };
                    }
                }
            }

            // Fallback: return last binding or Default
            if let Some(last_stmt) = statements.last() {
                if let Some(name) = &last_stmt.binding {
                    let last_ident = format_ident!("{}", name);
                    return quote! {
                        #(#stmts)*
                        Ok(#last_ident)
                    };
                }
            }

            quote! {
                #(#stmts)*
                Ok(Default::default())
            }
        }
        ToolImplIR::Parallel { statements } => {
            // Track local variables incrementally
            let mut current_locals = local_vars.clone();
            let mut stmts: Vec<TokenStream> = Vec::new();

            for stmt in statements {
                let stmt_code =
                    gen_tool_statement_with_ctx(stmt, tools_index, input_fields, &current_locals);
                stmts.push(stmt_code);
                if let Some(name) = &stmt.binding {
                    current_locals.insert(name.clone());
                }
            }

            // Collect all binding names for output construction
            let bindings: std::collections::HashSet<String> = statements
                .iter()
                .filter_map(|s| s.binding.clone())
                .collect();

            // Only construct output if ALL field names have matching bindings
            if let Some(TypeIR::Struct { fields }) = expected {
                if !fields.is_empty() {
                    let all_match = fields.keys().all(|k| bindings.contains(k));
                    if all_match {
                        let field_inits: Vec<_> = fields
                            .keys()
                            .map(|k| {
                                let ident = format_ident!("{}", to_ident(k));
                                quote! { #ident: #ident }
                            })
                            .collect();
                        return quote! {
                            // TODO: Execute in parallel (currently sequential)
                            #(#stmts)*
                            let __output = Output { #(#field_inits),* };
                            Ok(__output)
                        };
                    }
                }
            }

            // Fallback: return last binding or Default
            if let Some(last_stmt) = statements.last() {
                if let Some(name) = &last_stmt.binding {
                    let last_ident = format_ident!("{}", name);
                    return quote! {
                        // TODO: Execute in parallel (currently sequential)
                        #(#stmts)*
                        Ok(#last_ident)
                    };
                }
            }

            quote! {
                // TODO: Execute in parallel (currently sequential)
                #(#stmts)*
                Ok(Default::default())
            }
        }
    }
}

/// Generate a tool expression with optional expected output type context and tool index
fn gen_tool_expr_with_ctx(
    expr: &ToolExprIR,
    expected: Option<&TypeIR>,
    tools_index: &std::collections::HashMap<String, ToolIR>,
    input_fields: &std::collections::HashSet<String>,
    local_vars: &std::collections::HashSet<String>,
) -> TokenStream {
    match expr {
        ToolExprIR::Ident { name } => {
            let ident = format_ident!("{}", name);
            if local_vars.contains(name) {
                // Local variable from previous statement
                quote! { #ident }
            } else if input_fields.contains(name) {
                // Input field - prefix with input.
                quote! { input.#ident }
            } else {
                // Unknown - might be external or defined later
                quote! { #ident }
            }
        }
        ToolExprIR::FieldAccess { base, field } => {
            let base_code =
                gen_tool_expr_with_ctx(base, None, tools_index, input_fields, local_vars);
            let field_ident = format_ident!("{}", field);
            quote! { #base_code.#field_ident }
        }
        ToolExprIR::ForeignCall {
            module,
            function,
            args,
        } => {
            let mod_ident = format_ident!("{}", to_snake_case(module));
            let fn_ident = format_ident!("{}", to_snake_case(function));
            let arg_codes: Vec<_> = args
                .iter()
                .map(|a| gen_tool_expr_with_ctx(a, None, tools_index, input_fields, local_vars))
                .collect();
            quote! {
                crate::foreign::#mod_ident::#fn_ident(#(#arg_codes),*)
            }
        }
        ToolExprIR::ToolCall { tool, args } => {
            let tool_struct = format_ident!("{}Tool", to_pascal_case(tool));
            let callee_mod = format_ident!("{}", to_snake_case(tool));
            let arg_codes: Vec<_> = args
                .iter()
                .map(|a| gen_tool_expr_with_ctx(a, None, tools_index, input_fields, local_vars))
                .collect();

            if let Some(callee) = tools_index.get(tool) {
                match &callee.input {
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
                                let __input = crate::tools::#callee_mod::Input { #(#field_inits),* };
                                crate::tools::#tool_struct::new().execute(__input)?
                            }
                        }
                    }
                    _ => {
                        if arg_codes.is_empty() {
                            quote! { crate::tools::#tool_struct::new().execute(())? }
                        } else if arg_codes.len() == 1 {
                            let arg = &arg_codes[0];
                            quote! { crate::tools::#tool_struct::new().execute(#arg)? }
                        } else {
                            quote! { crate::tools::#tool_struct::new().execute((#(#arg_codes),*))? }
                        }
                    }
                }
            } else {
                if arg_codes.is_empty() {
                    quote! { crate::tools::#tool_struct::new().execute(())? }
                } else if arg_codes.len() == 1 {
                    let arg = &arg_codes[0];
                    quote! { crate::tools::#tool_struct::new().execute(#arg)? }
                } else {
                    quote! { crate::tools::#tool_struct::new().execute((#(#arg_codes),*))? }
                }
            }
        }
        ToolExprIR::Shell { command } => {
            // Parse template variables like {text} and replace with input.text
            let (formatted_cmd, has_vars) = interpolate_shell_command(command);
            let base_exec = if has_vars {
                formatted_cmd
            } else {
                quote! { scaffold_runtime::shell::execute(#command)? }
            };
            if let Some(exp) = expected {
                match exp {
                    TypeIR::String => quote! {{ let __out = #base_exec; __out.trim().to_string() }},
                    TypeIR::Int => {
                        quote! {{ let __out = #base_exec; scaffold_runtime::parse::parse_i64(__out.trim())? }}
                    }
                    TypeIR::Float => {
                        quote! {{ let __out = #base_exec; scaffold_runtime::parse::parse_f64(__out.trim())? }}
                    }
                    TypeIR::Bool => {
                        quote! {{ let __out = #base_exec; scaffold_runtime::parse::parse_bool(__out.trim())? }}
                    }
                    TypeIR::Bytes => quote! { scaffold_runtime::shell::execute_bytes(#command)? },
                    TypeIR::Any => {
                        quote! {{ let __out = #base_exec; scaffold_runtime::Value::from(__out) }}
                    }
                    _ => {
                        quote! {{ let __out = #base_exec; serde_json::from_str::<Output>(__out.trim()).map_err(|e| scaffold_runtime::Error::ParseError(e.to_string()))? }}
                    }
                }
            } else {
                base_exec
            }
        }
        ToolExprIR::Pipe { left, right } => {
            let left_code =
                gen_tool_expr_with_ctx(left, None, tools_index, input_fields, local_vars);
            let right_code =
                gen_tool_expr_with_ctx(right, None, tools_index, input_fields, local_vars);
            // Pipe passes left result to right
            quote! {
                {
                    let __pipe_input = #left_code;
                    let __pipe_fn = #right_code;
                    __pipe_fn(__pipe_input)
                }
            }
        }
        ToolExprIR::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond_code = gen_expr(condition);
            let then_code = gen_tool_impl_with_ctx(
                then_branch,
                expected,
                tools_index,
                input_fields,
                local_vars,
            );
            let else_code = match else_branch {
                Some(branch) => {
                    gen_tool_impl_with_ctx(branch, expected, tools_index, input_fields, local_vars)
                }
                None => quote! { Ok(Default::default()) },
            };
            quote! {
                if #cond_code {
                    #then_code
                } else {
                    #else_code
                }?
            }
        }
        ToolExprIR::Match { scrutinee, arms } => {
            let scrutinee_code =
                gen_tool_expr_with_ctx(scrutinee, None, tools_index, input_fields, local_vars);
            let arm_codes: Vec<_> = arms
                .iter()
                .map(|arm| {
                    let pattern = gen_expr(&arm.pattern);
                    let body = gen_tool_impl_with_ctx(
                        &arm.body,
                        expected,
                        tools_index,
                        input_fields,
                        local_vars,
                    );
                    quote! {
                        #pattern => { #body }
                    }
                })
                .collect();
            quote! {
                match #scrutinee_code {
                    #(#arm_codes)*
                    _ => Ok(Default::default()),
                }?
            }
        }
        ToolExprIR::Literal { value } => gen_literal(value),
        ToolExprIR::For {
            variable,
            iterable,
            body,
        } => {
            let var_ident = format_ident!("{}", variable);
            let iterable_code =
                gen_tool_expr_with_ctx(iterable, None, tools_index, input_fields, local_vars);
            // Add loop variable to local vars for body
            let mut body_locals = local_vars.clone();
            body_locals.insert(variable.clone());
            let body_code =
                gen_tool_impl_with_ctx(body, None, tools_index, input_fields, &body_locals);
            quote! {
                {
                    let mut __for_result = Ok(Default::default());
                    for #var_ident in #iterable_code.into_iter() {
                        match (|| { #body_code })() {
                            Ok(val) => __for_result = Ok(val),
                            Err(scaffold_runtime::Error::LoopBreak) => break,
                            Err(scaffold_runtime::Error::LoopContinue) => continue,
                            Err(e) => return Err(e),
                        }
                    }
                    __for_result?
                }
            }
        }
        ToolExprIR::While { condition, body } => {
            let cond_code = gen_expr(condition);
            let body_code =
                gen_tool_impl_with_ctx(body, None, tools_index, input_fields, local_vars);
            quote! {
                {
                    let mut __while_result = Ok(Default::default());
                    while #cond_code {
                        match (|| { #body_code })() {
                            Ok(val) => __while_result = Ok(val),
                            Err(scaffold_runtime::Error::LoopBreak) => break,
                            Err(scaffold_runtime::Error::LoopContinue) => continue,
                            Err(e) => return Err(e),
                        }
                    }
                    __while_result?
                }
            }
        }
        ToolExprIR::Loop { body } => {
            let body_code =
                gen_tool_impl_with_ctx(body, None, tools_index, input_fields, local_vars);
            quote! {
                {
                    let mut __loop_result = Ok(Default::default());
                    loop {
                        match (|| { #body_code })() {
                            Ok(val) => __loop_result = Ok(val),
                            Err(scaffold_runtime::Error::LoopBreak) => break,
                            Err(scaffold_runtime::Error::LoopContinue) => continue,
                            Err(e) => return Err(e),
                        }
                    }
                    __loop_result?
                }
            }
        }
        ToolExprIR::Break => {
            quote! { return Err(scaffold_runtime::Error::LoopBreak) }
        }
        ToolExprIR::Continue => {
            quote! { return Err(scaffold_runtime::Error::LoopContinue) }
        }
        ToolExprIR::Expr { expr } => {
            // Check if this is a function call that should be treated as a tool call
            gen_expr_or_tool_call(expr, tools_index, input_fields, local_vars)
        }
    }
}

/// Generate a literal value
fn gen_literal(lit: &LiteralIR) -> TokenStream {
    match lit {
        LiteralIR::Int { value } => quote! { #value },
        LiteralIR::Float { value } => quote! { #value },
        LiteralIR::String { value } => quote! { #value.to_string() },
        LiteralIR::Bool { value } => quote! { #value },
        LiteralIR::Null => quote! { None },
    }
}

/// Generate a tool statement (for sequence/parallel blocks)
fn gen_tool_statement_with_ctx(
    stmt: &ToolStatementIR,
    tools_index: &std::collections::HashMap<String, ToolIR>,
    input_fields: &std::collections::HashSet<String>,
    local_vars: &std::collections::HashSet<String>,
) -> TokenStream {
    let expr_code = gen_tool_expr_with_ctx(&stmt.expr, None, tools_index, input_fields, local_vars);

    match &stmt.binding {
        Some(name) => {
            let binding_ident = format_ident!("{}", name);
            quote! {
                let #binding_ident = #expr_code;
            }
        }
        None => {
            quote! {
                let _ = #expr_code;
            }
        }
    }
}

/// Generate spec checks (preconditions, postconditions)
fn gen_spec_checks(spec: &ToolSpecIR) -> (TokenStream, TokenStream, TokenStream) {
    // Generate precondition checks
    let pre_checks: Vec<_> = spec
        .preconditions
        .iter()
        .map(|expr| {
            let expr_code = gen_expr(expr);
            let expr_str = format!("{:?}", expr);
            quote! {
                if !(#expr_code) {
                    return Err(scaffold_runtime::Error::PreconditionFailed(#expr_str.to_string()));
                }
            }
        })
        .collect();

    let pre_check = if pre_checks.is_empty() {
        quote! {}
    } else {
        quote! {
            // Precondition checks
            #(#pre_checks)*
        }
    };

    // Generate postcondition checks
    let post_checks: Vec<_> = spec.postconditions.iter().map(|expr| {
        let expr_code = gen_expr(expr);
        let expr_str = format!("{:?}", expr);
        quote! {
            if let Ok(ref output) = result {
                let _ = output; // Make output available
                if !(#expr_code) {
                    return Err(scaffold_runtime::Error::PostconditionFailed(#expr_str.to_string()));
                }
            }
        }
    }).collect();

    let post_check = if post_checks.is_empty() {
        quote! { let _ = &result; }
    } else {
        quote! {
            // Postcondition checks
            #(#post_checks)*
        }
    };

    // Pure attribute (just a doc comment for now)
    let pure_attr = if spec.pure {
        quote! {
            /// This tool is pure (no side effects)
        }
    } else {
        quote! {}
    };

    (pre_check, post_check, pure_attr)
}

/// Generate a variant method
fn gen_variant_method(
    variant: &ToolVariantIR,
    input_type: &TokenStream,
    output_type: &TokenStream,
    tools_index: &std::collections::HashMap<String, ToolIR>,
    input_fields: &std::collections::HashSet<String>,
) -> TokenStream {
    let method_name = format_ident!("execute_{}", to_snake_case(&variant.name));
    let empty_locals: std::collections::HashSet<String> = std::collections::HashSet::new();
    // No explicit expected type; rely on implementation and type checker
    let body = gen_tool_impl_with_ctx(
        &variant.implementation,
        None,
        tools_index,
        input_fields,
        &empty_locals,
    );

    let doc = format!("Execute using the '{}' variant", variant.name);

    quote! {
        #[doc = #doc]
        pub fn #method_name(&self, input: #input_type) -> scaffold_runtime::Result<#output_type> {
            #body
        }
    }
}

/// Generate code to create JSON schema for the input type
/// Uses schemars to generate schema at runtime from types that derive JsonSchema
fn gen_input_schema_fn(ty: &TypeIR) -> TokenStream {
    // For types that derive JsonSchema, we use schemars::schema_for!
    // This works because our generated types now derive JsonSchema
    let rust_type = gen_type(ty);
    quote! {
        {
            let schema = schemars::schema_for!(#rust_type);
            serde_json::to_value(schema).unwrap_or_default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_tools_mod() {
        let names = vec!["read_file", "parse_binary"];
        let code = gen_tools_mod(&names).to_string();
        assert!(code.contains("read_file"));
        assert!(code.contains("parse_binary"));
        assert!(code.contains("ReadFileTool"));
        assert!(code.contains("ParseBinaryTool"));
    }

    #[test]
    fn test_gen_simple_tool() {
        use scaffold_ir::TypeIR;

        let tool = ToolIR {
            name: "read_file".to_string(),
            input: TypeIR::Struct {
                fields: {
                    let mut f = std::collections::HashMap::new();
                    f.insert("path".to_string(), TypeIR::String);
                    f
                },
            },
            output: TypeIR::Bytes,
            implementation: Some(ToolImplIR::Expr {
                expr: ToolExprIR::Shell {
                    command: "cat {path}".to_string(),
                },
            }),
            spec: None,
            variants: vec![],
        };

        let mut idx = std::collections::HashMap::new();
        idx.insert(tool.name.clone(), tool.clone());
        let code = gen_tool_module(&tool, &idx).to_string();
        eprintln!("Generated code:\n{}", code);
        assert!(code.contains("ReadFileTool"));
        assert!(code.contains("execute"));
        assert!(code.contains("scaffold_runtime :: shell :: execute_bytes"));
    }
}
