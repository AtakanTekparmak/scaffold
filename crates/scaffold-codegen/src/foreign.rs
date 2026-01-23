//! Foreign module code generation
//!
//! Generates Rust bindings for foreign declarations.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use scaffold_ir::{ExternCrateIR, ForeignModuleIR, ForeignFnIR, ForeignTypeAliasIR, TypeIR};

use crate::types::gen_type;
use crate::util::{to_snake_case, to_type_name};

/// Generate extern crate entries for Cargo.toml
pub fn gen_extern_crate_deps(crates: &[ExternCrateIR]) -> String {
    let mut deps = String::new();

    for ext in crates {
        let name = &ext.name;
        let version = &ext.version;

        if ext.features.is_empty() {
            deps.push_str(&format!("{} = \"{}\"\n", name, version));
        } else {
            let features: Vec<_> = ext.features.iter().map(|f| format!("\"{}\"", f)).collect();
            deps.push_str(&format!(
                "{} = {{ version = \"{}\", features = [{}] }}\n",
                name,
                version,
                features.join(", ")
            ));
        }
    }

    deps
}

/// Generate the foreign modules mod.rs
pub fn gen_foreign_mod(module_names: &[&str]) -> TokenStream {
    let mods: Vec<_> = module_names
        .iter()
        .map(|name| {
            let mod_name = format_ident!("{}", to_snake_case(name));
            quote! {
                pub mod #mod_name;
            }
        })
        .collect();

    let uses: Vec<_> = module_names
        .iter()
        .map(|name| {
            let mod_name = format_ident!("{}", to_snake_case(name));
            quote! {
                pub use #mod_name::*;
            }
        })
        .collect();

    quote! {
        //! Foreign module bindings

        #(#mods)*

        #(#uses)*
    }
}

/// Parse a Rust type path string into tokens
#[allow(dead_code)]
fn parse_type_path(path: &str) -> TokenStream {
    // Parse path like "goblin::elf::Elf" into tokens
    let segments: Vec<_> = path.split("::").collect();

    if segments.is_empty() {
        return quote! { () };
    }

    let idents: Vec<_> = segments.iter().map(|s| {
        // Handle generic parameters like "Elf<'static>"
        if let Some(idx) = s.find('<') {
            let name = &s[..idx];
            // Strip generics - they'll be handled separately
            format_ident!("{}", name)
        } else {
            format_ident!("{}", *s)
        }
    }).collect();

    // Check if the original path has lifetime parameters
    let has_lifetime = path.contains("<'");

    if idents.len() == 1 {
        let ident = &idents[0];
        if has_lifetime {
            quote! { #ident<'static> }
        } else {
            quote! { #ident }
        }
    } else {
        let first = &idents[0];
        let rest = &idents[1..];
        if has_lifetime {
            quote! { #first #(::#rest)* <'static> }
        } else {
            quote! { #first #(::#rest)* }
        }
    }
}

/// Generate a foreign module with type aliases and function bindings
pub fn gen_foreign_module(module: &ForeignModuleIR) -> TokenStream {
    // Generate type re-exports that reference actual external types
    let type_reexports: Vec<_> = module.type_aliases.iter().map(|alias| {
        let scaffold_name = format_ident!("{}", to_type_name(&alias.name));
        let doc = format!("Type alias for `{}`", alias.external_type);

        quote! {
            #[doc = #doc]
            pub use crate::foreign_types::#scaffold_name;
        }
    }).collect();

    // Generate function bindings
    let functions: Vec<_> = module.functions.iter().map(|func| {
        gen_foreign_fn(&module.name, func, &module.type_aliases)
    }).collect();

    let mod_doc = format!("Foreign bindings for `{} {}`", module.language, module.name);

    quote! {
        #![doc = #mod_doc]
        #![allow(unused_imports)]

        use crate::types::*;
        use scaffold_runtime::prelude::*;

        // Type re-exports
        #(#type_reexports)*

        // Function bindings
        #(#functions)*
    }
}

/// Generate a foreign function binding that actually calls the external function
fn gen_foreign_fn(module_name: &str, func: &ForeignFnIR, type_aliases: &[ForeignTypeAliasIR]) -> TokenStream {
    let fn_name = format_ident!("{}", to_snake_case(&func.name));

    // Generate parameters
    let params: Vec<_> = func.params.iter().map(|p| {
        let name = format_ident!("{}", to_snake_case(&p.name));
        let ty = gen_type(&p.ty);
        quote! { #name: #ty }
    }).collect();

    let return_ty = gen_type(&func.return_type);

    // Generate parameter names for the call
    let param_names: Vec<_> = func.params.iter().map(|p| {
        format_ident!("{}", to_snake_case(&p.name))
    }).collect();

    // Generate the function body based on the return type
    let body = gen_foreign_fn_body(func, &param_names, type_aliases);

    let doc = format!("Calls foreign function `{}::{}`", module_name, func.name);

    quote! {
        #[doc = #doc]
        pub fn #fn_name(#(#params),*) -> #return_ty {
            #body
        }
    }
}

/// Generate the body of a foreign function
fn gen_foreign_fn_body(func: &ForeignFnIR, param_names: &[proc_macro2::Ident], _type_aliases: &[ForeignTypeAliasIR]) -> TokenStream {
    // Check if return type is Result
    let is_result = matches!(&func.return_type, TypeIR::Result { .. });

    // Generate parameter conversions
    let conversions: Vec<_> = func.params.iter().zip(param_names.iter()).map(|(param, name)| {
        gen_param_conversion(name, &param.ty)
    }).collect();

    let converted_names: Vec<_> = param_names.iter().map(|n| {
        format_ident!("{}_converted", n)
    }).collect();

    // The actual foreign call placeholder - users implement this in foreign_impl module
    let fn_name = to_snake_case(&func.name);
    let impl_fn = format_ident!("{}_impl", fn_name);

    if is_result {
        quote! {
            #(#conversions)*
            crate::foreign_impl::#impl_fn(#(#converted_names),*)
        }
    } else {
        quote! {
            #(#conversions)*
            crate::foreign_impl::#impl_fn(#(#converted_names),*)
        }
    }
}

/// Generate parameter conversion code
fn gen_param_conversion(name: &proc_macro2::Ident, ty: &TypeIR) -> TokenStream {
    let converted = format_ident!("{}_converted", name);

    match ty {
        TypeIR::Bytes => {
            // bytes -> &[u8]
            quote! { let #converted = &#name[..]; }
        }
        TypeIR::String => {
            // String -> &str
            quote! { let #converted = #name.as_str(); }
        }
        _ => {
            // Pass through
            quote! { let #converted = #name; }
        }
    }
}

/// Generate a foreign_types module with actual type wrappers
pub fn gen_foreign_types_module(modules: &[ForeignModuleIR]) -> TokenStream {
    let mut all_types = Vec::new();

    for module in modules {
        for alias in &module.type_aliases {
            let type_name = format_ident!("{}", to_type_name(&alias.name));
            let doc = format!("Wrapper for foreign type `{}`", alias.external_type);

            all_types.push(quote! {
                #[doc = #doc]
                #[derive(Debug)]
                pub struct #type_name {
                    inner: Box<dyn std::any::Any + Send + Sync>,
                }

                impl #type_name {
                    /// Create a new wrapper from the foreign type
                    pub fn new<T: std::any::Any + Send + Sync + 'static>(value: T) -> Self {
                        Self { inner: Box::new(value) }
                    }

                    /// Try to get a reference to the inner value
                    pub fn as_ref<T: 'static>(&self) -> Option<&T> {
                        self.inner.downcast_ref()
                    }

                    /// Try to get a mutable reference to the inner value
                    pub fn as_mut<T: 'static>(&mut self) -> Option<&mut T> {
                        self.inner.downcast_mut()
                    }

                    /// Consume and try to extract the inner value
                    pub fn into_inner<T: 'static>(self) -> Result<T, Self> {
                        match self.inner.downcast::<T>() {
                            Ok(boxed) => Ok(*boxed),
                            Err(inner) => Err(Self { inner }),
                        }
                    }
                }

                impl Clone for #type_name {
                    fn clone(&self) -> Self {
                        // Foreign types may not be clonable - return a placeholder
                        // Users should implement proper cloning if needed
                        panic!("Clone not implemented for foreign type {}", stringify!(#type_name))
                    }
                }

                impl Default for #type_name {
                    fn default() -> Self {
                        panic!("Default not implemented for foreign type {}", stringify!(#type_name))
                    }
                }
            });
        }
    }

    quote! {
        //! Foreign type wrappers
        //!
        //! These types wrap foreign types from external crates.
        //! Use `as_ref::<T>()` to access the underlying type.

        #(#all_types)*
    }
}

/// Generate a foreign_impl module with implementation stubs
pub fn gen_foreign_impl_module(modules: &[ForeignModuleIR]) -> TokenStream {
    let mut all_fns = Vec::new();

    for module in modules {
        for func in &module.functions {
            let fn_name = format_ident!("{}_impl", to_snake_case(&func.name));

            // Generate parameters with converted types
            let params: Vec<_> = func.params.iter().map(|p| {
                let name = format_ident!("{}_converted", to_snake_case(&p.name));
                let ty = gen_impl_param_type(&p.ty);
                quote! { #name: #ty }
            }).collect();

            let return_ty = gen_type(&func.return_type);

            let full_name = format!("{}::{}", module.name, func.name);
            let doc = format!("Implementation for `{}`\n\nTODO: Implement this function to call the actual foreign code.", full_name);

            // Generate example implementation comment
            let example = gen_impl_example(&func.name, &module.type_aliases, func);

            all_fns.push(quote! {
                #[doc = #doc]
                ///
                /// # Example implementation
                ///
                #[doc = #example]
                pub fn #fn_name(#(#params),*) -> #return_ty {
                    todo!("Implement foreign function call")
                }
            });
        }
    }

    quote! {
        //! Foreign function implementations
        //!
        //! This module contains stub implementations for foreign functions.
        //! Replace the `todo!()` calls with actual implementations.

        #![allow(unused_variables)]

        use crate::types::*;
        use crate::foreign_types::*;

        #(#all_fns)*
    }
}

/// Generate the parameter type for impl functions
fn gen_impl_param_type(ty: &TypeIR) -> TokenStream {
    match ty {
        TypeIR::Bytes => quote! { &[u8] },
        TypeIR::String => quote! { &str },
        _ => gen_type(ty),
    }
}

/// Generate an example implementation comment
fn gen_impl_example(fn_name: &str, type_aliases: &[ForeignTypeAliasIR], func: &ForeignFnIR) -> String {
    let mut example = String::from("```rust,ignore\n");

    // Check if this looks like a parse function
    if fn_name.starts_with("parse") {
        // Find the return type's external path
        if let TypeIR::Result { ok, .. } = &func.return_type {
            if let TypeIR::Named { name } = ok.as_ref() {
                if let Some(alias) = type_aliases.iter().find(|a| &a.name == name) {
                    example.push_str(&format!(
                        "// Example for parsing with {}\n",
                        alias.external_type
                    ));
                    example.push_str(&format!(
                        "let parsed = {}::parse(data)\n",
                        alias.external_type.split("::").take(2).collect::<Vec<_>>().join("::")
                    ));
                    example.push_str(&format!(
                        "    .map(|v| {}::new(v))\n",
                        name
                    ));
                    example.push_str("    .map_err(|e| e.to_string())\n");
                }
            }
        }
    }

    example.push_str("```");
    example
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_ir::{ForeignTypeAliasIR, ForeignParamIR, TypeIR};

    #[test]
    fn test_gen_extern_crate_deps() {
        let crates = vec![
            ExternCrateIR {
                name: "goblin".to_string(),
                version: "0.7".to_string(),
                features: vec![],
            },
            ExternCrateIR {
                name: "object".to_string(),
                version: "0.32".to_string(),
                features: vec!["read".to_string(), "write".to_string()],
            },
        ];

        let deps = gen_extern_crate_deps(&crates);
        assert!(deps.contains("goblin = \"0.7\""));
        assert!(deps.contains("object = { version = \"0.32\", features = [\"read\", \"write\"] }"));
    }

    #[test]
    fn test_gen_foreign_module() {
        let module = ForeignModuleIR {
            language: "rust".to_string(),
            name: "parsing".to_string(),
            type_aliases: vec![
                ForeignTypeAliasIR {
                    name: "ElfBinary".to_string(),
                    external_type: "goblin::elf::Elf".to_string(),
                }
            ],
            functions: vec![
                ForeignFnIR {
                    name: "parse_elf".to_string(),
                    params: vec![
                        ForeignParamIR {
                            name: "data".to_string(),
                            ty: TypeIR::Bytes,
                        }
                    ],
                    return_type: TypeIR::Result {
                        ok: Box::new(TypeIR::Named { name: "ElfBinary".to_string() }),
                        err: Box::new(TypeIR::String),
                    },
                }
            ],
        };

        let code = gen_foreign_module(&module).to_string();
        assert!(code.contains("parse_elf"));
        assert!(code.contains("ElfBinary"));
    }

    #[test]
    fn test_parse_type_path() {
        let path = parse_type_path("goblin::elf::Elf");
        let code = path.to_string();
        assert!(code.contains("goblin"));
        assert!(code.contains("elf"));
        assert!(code.contains("Elf"));
    }

    #[test]
    fn test_gen_foreign_impl_module() {
        let modules = vec![
            ForeignModuleIR {
                language: "rust".to_string(),
                name: "parsing".to_string(),
                type_aliases: vec![],
                functions: vec![
                    ForeignFnIR {
                        name: "parse_data".to_string(),
                        params: vec![
                            ForeignParamIR {
                                name: "input".to_string(),
                                ty: TypeIR::Bytes,
                            }
                        ],
                        return_type: TypeIR::String,
                    }
                ],
            }
        ];

        let code = gen_foreign_impl_module(&modules).to_string();
        assert!(code.contains("parse_data_impl"));
    }
}
