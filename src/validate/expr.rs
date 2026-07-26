//! Validation vocabulary: comparison operators ([`Cmp`]) and the boolean
//! predicate tree ([`Expr`]) evaluated against a [`Keyword`](crate::model::Keyword).
//! Values are the core [`Value`](crate::model::Value).

use crate::model::{Keyword, Value};

/// A comparison operator — used instead of a stringly `"eq"`/`"ne"`.
#[cfg_attr(feature = "python", pyo3::pyclass(eq, eq_int, from_py_object, name = "Cmp"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    fn test_num(self, a: f64, b: f64) -> bool {
        match self {
            Cmp::Eq => a == b,
            Cmp::Ne => a != b,
            Cmp::Lt => a < b,
            Cmp::Le => a <= b,
            Cmp::Gt => a > b,
            Cmp::Ge => a >= b,
        }
    }
    /// Compare two [`Value`]s: numeric when both coerce to a number, else
    /// equality/inequality on the raw value (ordering of non-numbers is `false`).
    pub(crate) fn test(self, a: &Value, b: &Value) -> bool {
        match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => self.test_num(x, y),
            _ => match self {
                Cmp::Eq => a == b,
                Cmp::Ne => a != b,
                _ => false,
            },
        }
    }
    fn symbol(self) -> &'static str {
        match self {
            Cmp::Eq => "==",
            Cmp::Ne => "!=",
            Cmp::Lt => "<",
            Cmp::Le => "<=",
            Cmp::Gt => ">",
            Cmp::Ge => ">=",
        }
    }
}

/// A predicate on one card field: `field <cmp> value`.
#[derive(Debug, Clone)]
pub struct FieldPredicate {
    pub field: String,
    pub cmp: Cmp,
    pub value: Value,
}

/// A boolean expression tree over field predicates — the composable
/// ("tier 2") layer. Evaluated entirely in Rust.
#[derive(Debug, Clone)]
pub enum Expr {
    Field(FieldPredicate),
    All(Vec<Expr>),
    Any(Vec<Expr>),
    Not(Box<Expr>),
}

impl Expr {
    /// `field <cmp> value`.
    pub fn field(field: impl Into<String>, cmp: Cmp, value: Value) -> Expr {
        Expr::Field(FieldPredicate { field: field.into(), cmp, value })
    }
    pub fn all(exprs: impl IntoIterator<Item = Expr>) -> Expr {
        Expr::All(exprs.into_iter().collect())
    }
    pub fn any(exprs: impl IntoIterator<Item = Expr>) -> Expr {
        Expr::Any(exprs.into_iter().collect())
    }
    pub fn not(expr: Expr) -> Expr {
        Expr::Not(Box::new(expr))
    }

    pub(crate) fn eval(&self, kw: &Keyword) -> bool {
        match self {
            Expr::Field(p) => kw.field(&p.field).map(|f| p.cmp.test(&f.value(), &p.value)).unwrap_or(false),
            Expr::All(v) => v.iter().all(|e| e.eval(kw)),
            Expr::Any(v) => v.iter().any(|e| e.eval(kw)),
            Expr::Not(e) => !e.eval(kw),
        }
    }

    pub(crate) fn describe(&self) -> String {
        match self {
            Expr::Field(p) => format!("{} {} {}", p.field, p.cmp.symbol(), p.value.display()),
            Expr::All(v) => format!("all({})", v.iter().map(Expr::describe).collect::<Vec<_>>().join(", ")),
            Expr::Any(v) => format!("any({})", v.iter().map(Expr::describe).collect::<Vec<_>>().join(", ")),
            Expr::Not(e) => format!("not({})", e.describe()),
        }
    }
}

/// Convenience: a single-field predicate as an [`Expr`].
pub fn pred(field: impl Into<String>, cmp: Cmp, value: Value) -> Expr {
    Expr::field(field, cmp, value)
}
