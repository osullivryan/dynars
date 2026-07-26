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
use quote::{format_ident, quote};
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

    // The single parse entry point on the struct. For a single-card keyword it
    // is specialized (monomorphized) code; for multi-card it interprets the
    // schema (those are low volume). Either way it returns a `Table`.
    let parse_body = match single_card_fields(&input) {
        Some(fields) => match specialized_parse(&kw_name, fields) {
            Ok(body) => body,
            Err(e) => return e.to_compile_error().into(),
        },
        None => quote! {
            ::dynars::schema::parse_schema(
                parsed,
                &<Self as ::dynars::schema::KeywordSchema>::schema(),
            )
        },
    };

    quote! {
        impl ::dynars::schema::KeywordSchema for #name {
            fn schema() -> ::dynars::schema::Schema {
                let __s = #build;
                #finish
            }
        }

        impl #name {
            /// Parse this keyword from a file into a columnar `Table`.
            pub fn parse(parsed: &::dynars::file::ParsedFile) -> ::dynars::schema::Table {
                #parse_body
            }
        }
    }
    .into()
}

/// The named fields for a single-card keyword (no `#[cards(...)]`), or `None`
/// for the multi-card / interpreted case.
fn single_card_fields(input: &DeriveInput) -> Option<&Punctuated<Field, Token![,]>> {
    if input.attrs.iter().any(|a| a.path().is_ident("cards")) {
        return None;
    }
    match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) => Some(&n.named),
            _ => None,
        },
        _ => None,
    }
}

/// Generate the specialized single-card parse body: typed column vecs filled by
/// a per-line loop with compile-time-known field layout (no enum dispatch),
/// driven in parallel by the shared chunk driver.
fn specialized_parse(
    keyword: &str,
    fields: &Punctuated<Field, Token![,]>,
) -> syn::Result<TokenStream2> {
    let mut decls = Vec::new();
    let mut width_decls = Vec::new();
    let mut free_pushes = Vec::new();
    let mut fixed_pushes = Vec::new();
    let mut wraps = Vec::new();
    let mut names = Vec::new();

    for (i, f) in fields.iter().enumerate() {
        let fname = f.ident.as_ref().unwrap().to_string();
        let var = format_ident!("__c_{}", f.ident.as_ref().unwrap());
        let wvar = format_ident!("__w_{}", i);
        let width = field_width(f)?;
        // Hoist effective width (base * long-scale) out of the per-line loop.
        width_decls.push(quote! { let #wvar: usize = #width * __scale; });

        let (elem_ty, to_fn, variant, count, is_str_array) = match &f.ty {
            Type::Array(arr) => {
                let count = &arr.len;
                match base_kind(&arr.elem)? {
                    Kind::Int => (quote!(i64), quote!(__to_int), quote!(Int), quote!(#count), false),
                    Kind::Float => {
                        (quote!(f64), quote!(__to_float), quote!(Float), quote!(#count), false)
                    }
                    Kind::Str => (quote!(String), quote!(__to_str), quote!(Str), quote!(#count), true),
                }
            }
            ty => match base_kind(ty)? {
                Kind::Int => (quote!(i64), quote!(__to_int), quote!(Int), quote!(1usize), false),
                Kind::Float => (quote!(f64), quote!(__to_float), quote!(Float), quote!(1usize), false),
                Kind::Str => (quote!(String), quote!(__to_str), quote!(Str), quote!(1usize), false),
            },
        };
        if is_str_array {
            return Err(syn::Error::new_spanned(&f.ty, "string arrays are not supported"));
        }

        decls.push(quote! { let mut #var: ::std::vec::Vec<#elem_ty> = ::std::vec::Vec::new(); });
        free_pushes.push(quote! {
            for _ in 0..#count {
                #var.push(::dynars::schema::#to_fn(__t.next().unwrap_or(&b""[..])));
            }
        });
        fixed_pushes.push(quote! {
            for _ in 0..#count {
                #var.push(::dynars::schema::#to_fn(::dynars::schema::__slice(__line, __off, #wvar)));
                __off += #wvar;
            }
        });
        wraps.push(quote! { ::dynars::schema::Column::#variant { data: #var, ncols: #count } });
        names.push(quote! { #fname });
    }

    Ok(quote! {
        let __cols = ::dynars::schema::__drive_single_card(parsed, #keyword, |__chunk, __fmt| {
            let __scale: usize =
                if __fmt == ::dynars::file::CardFormat::Long { 2 } else { 1 };
            #(#width_decls)*
            #(#decls)*
            for __line in __chunk.split(|&__b| __b == b'\n') {
                if ::dynars::schema::__is_skippable(__line) { continue; }
                let __line = ::dynars::schema::__strip_eol(__line);
                if ::dynars::schema::__is_free(__line, __fmt) {
                    let mut __t = __line.split(|&__b| __b == b',');
                    #(#free_pushes)*
                } else {
                    let mut __off = 0usize;
                    #(#fixed_pushes)*
                }
            }
            ::std::vec![ #(#wraps),* ]
        });
        ::dynars::schema::__table(::std::vec![ #(#names),* ], __cols)
    })
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
