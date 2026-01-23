//! Type IR to Rust type generation

use crate::util::{to_ident, to_type_name};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use scaffold_ir::{TypeDefIR, TypeIR, TypeRefIR};
use std::collections::HashMap;

/// Generate Rust type token stream from TypeIR
pub fn gen_type(ty: &TypeIR) -> TokenStream {
    match ty {
        TypeIR::Bool => quote! { bool },
        TypeIR::Int => quote! { i64 },
        TypeIR::Float => quote! { f64 },
        TypeIR::String => quote! { String },
        TypeIR::Bytes => quote! { Vec<u8> },
        TypeIR::Any => quote! { scaffold_runtime::Value },
        TypeIR::List { element } => {
            let elem_ty = gen_type(element);
            quote! { Vec<#elem_ty> }
        }
        TypeIR::Map { key, value } => {
            let key_ty = gen_type(key);
            let val_ty = gen_type(value);
            quote! { std::collections::HashMap<#key_ty, #val_ty> }
        }
        TypeIR::Option { inner } => {
            let inner_ty = gen_type(inner);
            quote! { Option<#inner_ty> }
        }
        TypeIR::Result { ok, err } => {
            let ok_ty = gen_type(ok);
            let err_ty = gen_type(err);
            quote! { Result<#ok_ty, #err_ty> }
        }
        TypeIR::Struct { fields } => {
            // Anonymous struct - generate inline struct type
            // This is rare; usually structs are named via TypeDefIR
            let field_tokens: Vec<_> = fields
                .iter()
                .map(|(name, ty)| {
                    let field_name = format_ident!("{}", to_ident(name));
                    let field_ty = gen_type(ty);
                    quote! { #field_name: #field_ty }
                })
                .collect();
            // Return a tuple for anonymous inline structs
            if field_tokens.is_empty() {
                quote! { () }
            } else {
                // Can't create anonymous struct in Rust, use a comment placeholder
                // This case should be avoided by using named types
                quote! { /* inline struct */ () }
            }
        }
        TypeIR::Named { name } => {
            let type_name = format_ident!("{}", to_type_name(name));
            quote! { #type_name }
        }
    }
}

/// Generate Rust type from TypeRefIR
pub fn gen_type_ref(type_ref: &TypeRefIR) -> TokenStream {
    match type_ref {
        TypeRefIR::Named { ref_name } => {
            let type_name = format_ident!("{}", to_type_name(ref_name));
            quote! { #type_name }
        }
        TypeRefIR::Inline(ty) => gen_type(ty),
    }
}

/// Generate a struct definition from TypeDefIR
pub fn gen_struct_def(type_def: &TypeDefIR) -> TokenStream {
    let type_name = format_ident!("{}", to_type_name(&type_def.name));

    match &type_def.definition {
        TypeIR::Struct { fields } => {
            let field_defs: Vec<_> = fields
                .iter()
                .map(|(name, ty)| {
                    let field_name = format_ident!("{}", to_ident(name));
                    let field_ty = gen_type(ty);
                    quote! {
                        pub #field_name: #field_ty
                    }
                })
                .collect();

            quote! {
                #[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
                pub struct #type_name {
                    #(#field_defs),*
                }
            }
        }
        // For non-struct types, create a type alias
        other => {
            let aliased_ty = gen_type(other);
            quote! {
                pub type #type_name = #aliased_ty;
            }
        }
    }
}

/// Generate all type definitions for a module
pub fn gen_types_module(types: &[TypeDefIR]) -> TokenStream {
    let type_defs: Vec<_> = types.iter().map(gen_struct_def).collect();

    quote! {
        //! Type definitions generated from scaffold IR

        use serde::{Deserialize, Serialize};
        use schemars::JsonSchema;

        #(#type_defs)*
    }
}

/// Collect all types referenced in a task (for input/output structs)
pub fn collect_task_types(
    input_type: &TypeRefIR,
    output_type: &TypeRefIR,
    type_defs: &HashMap<String, TypeDefIR>,
) -> Vec<TypeDefIR> {
    let mut result = Vec::new();
    let mut visited = std::collections::HashSet::new();

    fn collect_from_type(
        ty: &TypeIR,
        type_defs: &HashMap<String, TypeDefIR>,
        result: &mut Vec<TypeDefIR>,
        visited: &mut std::collections::HashSet<String>,
    ) {
        match ty {
            TypeIR::Named { name } => {
                if !visited.contains(name) {
                    visited.insert(name.clone());
                    if let Some(def) = type_defs.get(name) {
                        // Recursively collect types from fields
                        if let TypeIR::Struct { fields } = &def.definition {
                            for (_, field_ty) in fields {
                                collect_from_type(field_ty, type_defs, result, visited);
                            }
                        }
                        result.push(def.clone());
                    }
                }
            }
            TypeIR::List { element } => {
                collect_from_type(element, type_defs, result, visited);
            }
            TypeIR::Map { key, value } => {
                collect_from_type(key, type_defs, result, visited);
                collect_from_type(value, type_defs, result, visited);
            }
            TypeIR::Option { inner } => {
                collect_from_type(inner, type_defs, result, visited);
            }
            TypeIR::Result { ok, err } => {
                collect_from_type(ok, type_defs, result, visited);
                collect_from_type(err, type_defs, result, visited);
            }
            TypeIR::Struct { fields } => {
                for (_, field_ty) in fields {
                    collect_from_type(field_ty, type_defs, result, visited);
                }
            }
            _ => {}
        }
    }

    fn collect_from_ref(
        type_ref: &TypeRefIR,
        type_defs: &HashMap<String, TypeDefIR>,
        result: &mut Vec<TypeDefIR>,
        visited: &mut std::collections::HashSet<String>,
    ) {
        match type_ref {
            TypeRefIR::Named { ref_name } => {
                if !visited.contains(ref_name) {
                    visited.insert(ref_name.clone());
                    if let Some(def) = type_defs.get(ref_name) {
                        collect_from_type(&def.definition, type_defs, result, visited);
                        result.push(def.clone());
                    }
                }
            }
            TypeRefIR::Inline(ty) => {
                collect_from_type(ty, type_defs, result, visited);
            }
        }
    }

    collect_from_ref(input_type, type_defs, &mut result, &mut visited);
    collect_from_ref(output_type, type_defs, &mut result, &mut visited);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_primitive_types() {
        assert_eq!(gen_type(&TypeIR::Bool).to_string(), "bool");
        assert_eq!(gen_type(&TypeIR::Int).to_string(), "i64");
        assert_eq!(gen_type(&TypeIR::Float).to_string(), "f64");
        assert_eq!(gen_type(&TypeIR::String).to_string(), "String");
    }

    #[test]
    fn test_gen_container_types() {
        let list_ty = TypeIR::List {
            element: Box::new(TypeIR::Int),
        };
        assert_eq!(gen_type(&list_ty).to_string(), "Vec < i64 >");

        let option_ty = TypeIR::Option {
            inner: Box::new(TypeIR::String),
        };
        assert_eq!(gen_type(&option_ty).to_string(), "Option < String >");
    }

    #[test]
    fn test_gen_named_type() {
        let named = TypeIR::Named {
            name: "position".to_string(),
        };
        assert_eq!(gen_type(&named).to_string(), "Position");
    }
}
