//! Derive macros for `dynars` keyword schemas.
//!
//! `#[derive(Keyword)]` and `#[derive(Card)]` let you declare a keyword's
//! layout as a struct — the Rust mirror of the Python `@keyword` class — which
//! lowers to the exact same `dynars::schema::Schema` the builder produces.
//!
//! ```ignore
//! use dynars::{Keyword, Card};
//!
//! #[derive(Keyword)]
//! #[keyword("NODE")]                       // repeat defaults to true
//! struct Node {
//!     #[field(8)]  nid: i64,               // i64 -> Int, f64 -> Float, String -> Str
//!     #[field(16)] x: f64,
//!     #[field(16)] y: f64,
//!     #[field(16)] z: f64,
//! }
//!
//! #[derive(Card)] struct Heading  { #[field(80)] title: String }
//! #[derive(Card)] struct PartData { #[field(8)] pid: i64, #[field(8)] secid: i64, #[field(8)] mid: i64 }
//!
//! #[derive(Keyword)]
//! #[keyword("PART")]
//! #[cards(Heading, PartData)]              // multi-card by composition
//! struct Part;
//! ```

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    parse_macro_input, Data, DeriveInput, Field, Fields, Ident, LitBool, LitInt, Path, Token, Type,
};

/// `#[derive(Card)]` — implement `CardLayout` for a struct of `#[field]`s.
#[proc_macro_derive(Card, attributes(field))]
pub fn derive_card(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let card_expr = match build_card_expr(&input) {
        Ok(e) => e,
        Err(e) => return e.to_compile_error().into(),
    };
    quote! {
        impl ::dynars::schema::CardLayout for #name {
            fn card() -> ::dynars::schema::Card {
                #card_expr
            }
        }
    }
    .into()
}

/// `#[derive(Keyword)]` — implement `KeywordSchema`. The keyword name comes from
/// `#[keyword("NAME", repeat = true)]`; fields on the struct form a single card,
/// or `#[cards(A, B, ...)]` composes several `Card` types.
#[proc_macro_derive(Keyword, attributes(field, keyword, cards))]
pub fn derive_keyword(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let (kw_name, repeat) = match parse_keyword_attr(&input) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };

    let build = match parse_cards_attr(&input) {
        Ok(Some(paths)) => {
            let card_calls = paths.iter().map(|p| {
                quote! { .card(<#p as ::dynars::schema::CardLayout>::card()) }
            });
            quote! { ::dynars::schema::Schema::new(#kw_name) #(#card_calls)* }
        }
        Ok(None) => match build_card_expr(&input) {
            Ok(card_expr) => quote! { ::dynars::schema::Schema::new(#kw_name).card(#card_expr) },
            Err(e) => return e.to_compile_error().into(),
        },
        Err(e) => return e.to_compile_error().into(),
    };

    let finish = if repeat {
        quote! { __s }
    } else {
        quote! { __s.once() }
    };

    quote! {
        impl ::dynars::schema::KeywordSchema for #name {
            fn schema() -> ::dynars::schema::Schema {
                let __s = #build;
                #finish
            }
        }
    }
    .into()
}

/// Build a `Card::new().int(..).float(..)...` expression from a struct's fields.
fn build_card_expr(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) => &n.named,
            Fields::Unit => {
                return Err(syn::Error::new_spanned(
                    &input.ident,
                    "keyword struct has no fields and no #[cards(...)] — add fields or a cards list",
                ));
            }
            Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(&input.ident, "expected named fields"));
            }
        },
        _ => return Err(syn::Error::new_spanned(&input.ident, "expected a struct")),
    };

    let mut expr = quote! { ::dynars::schema::Card::new() };
    for f in fields {
        expr = {
            let call = field_call(f)?;
            quote! { #expr #call }
        };
    }
    Ok(expr)
}

/// One `.int(name, w)` / `.float_array(name, n, w)` / ... call for a field.
fn field_call(f: &Field) -> syn::Result<TokenStream2> {
    let fname = f.ident.as_ref().unwrap().to_string();
    let width = field_width(f)?;

    Ok(match &f.ty {
        Type::Array(arr) => {
            let count = &arr.len;
            match base_kind(&arr.elem)? {
                Kind::Int => quote! { .int_array(#fname, #count, #width) },
                Kind::Float => quote! { .float_array(#fname, #count, #width) },
                Kind::Str => {
                    return Err(syn::Error::new_spanned(&f.ty, "string arrays are not supported"));
                }
            }
        }
        ty => match base_kind(ty)? {
            Kind::Int => quote! { .int(#fname, #width) },
            Kind::Float => quote! { .float(#fname, #width) },
            Kind::Str => quote! { .str(#fname, #width) },
        },
    })
}

enum Kind {
    Int,
    Float,
    Str,
}

fn base_kind(ty: &Type) -> syn::Result<Kind> {
    if let Type::Path(p) = ty {
        let seg = p.path.segments.last().unwrap().ident.to_string();
        return match seg.as_str() {
            "i64" | "i32" | "i16" | "i8" | "u64" | "u32" | "u16" | "u8" | "usize" | "isize" => {
                Ok(Kind::Int)
            }
            "f64" | "f32" => Ok(Kind::Float),
            "String" => Ok(Kind::Str),
            _ => Err(syn::Error::new_spanned(
                ty,
                "unsupported field type (use an integer, f32/f64, String, or [T; N])",
            )),
        };
    }
    Err(syn::Error::new_spanned(ty, "unsupported field type"))
}

/// Read the width from `#[field(WIDTH)]`.
fn field_width(f: &Field) -> syn::Result<LitInt> {
    for attr in &f.attrs {
        if attr.path().is_ident("field") {
            return attr.parse_args::<LitInt>();
        }
    }
    Err(syn::Error::new_spanned(
        f.ident.as_ref().unwrap(),
        "missing #[field(width)] attribute",
    ))
}

/// Parsed `#[keyword("NAME", repeat = <bool>)]`.
struct KeywordArgs {
    name: syn::LitStr,
    repeat: bool,
}

impl Parse for KeywordArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: syn::LitStr = input.parse()?;
        let mut repeat = true;
        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            if key == "repeat" {
                repeat = input.parse::<LitBool>()?.value;
            } else {
                return Err(syn::Error::new_spanned(key, "expected `repeat`"));
            }
        }
        Ok(KeywordArgs { name, repeat })
    }
}

fn parse_keyword_attr(input: &DeriveInput) -> syn::Result<(String, bool)> {
    for attr in &input.attrs {
        if attr.path().is_ident("keyword") {
            let args: KeywordArgs = attr.parse_args()?;
            return Ok((args.name.value(), args.repeat));
        }
    }
    Err(syn::Error::new_spanned(
        &input.ident,
        "missing #[keyword(\"NAME\")] attribute",
    ))
}

fn parse_cards_attr(input: &DeriveInput) -> syn::Result<Option<Vec<Path>>> {
    for attr in &input.attrs {
        if attr.path().is_ident("cards") {
            let paths =
                attr.parse_args_with(Punctuated::<Path, Token![,]>::parse_terminated)?;
            return Ok(Some(paths.into_iter().collect()));
        }
    }
    Ok(None)
}
