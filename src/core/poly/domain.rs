use crate::core::hlir::{Dim, Symbol};
use crate::core::llir::affine::{AffineExpr, Var};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Aff {
    pub constant: i64,
    pub terms: Vec<(i64, PolyVar)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PolyVar {
    Iter(String),
    Param(Symbol),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Constraint {
    Eq(Aff),
    Ineq(Aff),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain {
    pub iters: Vec<String>,
    pub params: Vec<Symbol>,
    pub constraints: Vec<Constraint>,
}

// ---------------------------------------------------------------------------
// Aff: arithmetic, canonicalization, substitution
// ---------------------------------------------------------------------------

impl Aff {
    pub fn constant(v: i64) -> Self {
        Self {
            constant: v,
            terms: Vec::new(),
        }
    }

    pub fn iter_var(name: impl Into<String>) -> Self {
        Self {
            constant: 0,
            terms: vec![(1, PolyVar::Iter(name.into()))],
        }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mut terms = self.terms.clone();
        terms.extend(rhs.terms.clone());
        Self {
            constant: self.constant + rhs.constant,
            terms,
        }
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let mut terms = self.terms.clone();
        terms.extend(rhs.terms.iter().map(|(c, v)| (-*c, v.clone())));
        Self {
            constant: self.constant - rhs.constant,
            terms,
        }
    }

    pub fn scale(&self, factor: i64) -> Self {
        Self {
            constant: self.constant * factor,
            terms: self
                .terms
                .iter()
                .map(|(c, v)| (c * factor, v.clone()))
                .collect(),
        }
    }

    /// Canonicalize in-place: combine like terms, drop zeros, sort.
    pub fn canonicalize(&mut self) {
        let mut merged: Vec<(i64, PolyVar)> = Vec::new();
        for (coeff, var) in self.terms.drain(..) {
            if let Some(entry) = merged.iter_mut().find(|(_, v)| *v == var) {
                entry.0 += coeff;
            } else {
                merged.push((coeff, var));
            }
        }
        merged.retain(|(c, _)| *c != 0);
        merged.sort_by(|(_, a), (_, b)| a.cmp(b));
        self.terms = merged;
    }

    /// Return a canonicalized copy.
    pub fn canonicalized(&self) -> Self {
        let mut copy = self.clone();
        copy.canonicalize();
        copy
    }

    /// Coefficient for a given variable, summing duplicates.
    pub fn coefficient_of(&self, var: &PolyVar) -> i64 {
        self.terms
            .iter()
            .filter(|(_, v)| v == var)
            .map(|(c, _)| c)
            .sum()
    }

    /// Substitute all occurrences of `var` with `replacement`.
    pub fn substitute(&self, var: &PolyVar, replacement: &Aff) -> Self {
        let coeff = self.coefficient_of(var);
        if coeff == 0 {
            return self.clone();
        }
        let remaining: Vec<(i64, PolyVar)> = self
            .terms
            .iter()
            .filter(|(_, v)| v != var)
            .cloned()
            .collect();
        let scaled = replacement.scale(coeff);
        let mut result = Self {
            constant: self.constant + scaled.constant,
            terms: remaining,
        };
        result.terms.extend(scaled.terms);
        result
    }

    /// Returns true when the expression is a pure constant (no variables).
    pub fn is_constant(&self) -> bool {
        self.terms.iter().all(|(c, _)| *c == 0)
    }
}

// ---------------------------------------------------------------------------
// Conversions between LLIR AffineExpr and poly Aff
// ---------------------------------------------------------------------------

impl From<&AffineExpr> for Aff {
    fn from(expr: &AffineExpr) -> Self {
        Self {
            constant: expr.constant,
            terms: expr
                .terms
                .iter()
                .map(|(c, v)| {
                    let pv = match v {
                        Var::Loop(name) => PolyVar::Iter(name.clone()),
                        Var::Param(sym) => PolyVar::Param(*sym),
                    };
                    (*c, pv)
                })
                .collect(),
        }
    }
}

impl From<&Aff> for AffineExpr {
    fn from(aff: &Aff) -> Self {
        Self {
            constant: aff.constant,
            terms: aff
                .terms
                .iter()
                .map(|(c, v)| {
                    let var = match v {
                        PolyVar::Iter(name) => Var::Loop(name.clone()),
                        PolyVar::Param(sym) => Var::Param(*sym),
                    };
                    (*c, var)
                })
                .collect(),
        }
    }
}

/// Try to convert an HLIR `Dim` into a poly `Aff`.
///
/// Returns `None` for non-affine forms (Div, Mod).
pub fn dim_to_aff(dim: &Dim) -> Option<Aff> {
    match dim {
        Dim::Const(v) => Some(Aff::constant(*v)),
        Dim::Sym(sym) => Some(Aff {
            constant: 0,
            terms: vec![(1, PolyVar::Param(*sym))],
        }),
        Dim::Add(a, b) => {
            let a = dim_to_aff(a)?;
            let b = dim_to_aff(b)?;
            Some(a.add(&b))
        }
        Dim::Mul(a, b) => {
            let a = dim_to_aff(a)?;
            let b = dim_to_aff(b)?;
            // Affine only if one side is constant.
            if a.is_constant() {
                Some(b.scale(a.constant))
            } else if b.is_constant() {
                Some(a.scale(b.constant))
            } else {
                None
            }
        }
        Dim::Div(..) | Dim::Mod(..) => None,
    }
}

// ---------------------------------------------------------------------------
// Domain construction
// ---------------------------------------------------------------------------

pub fn shape_to_domain(shape: &[Dim]) -> Domain {
    shape_to_domain_checked(shape).expect("shape_to_domain requires affine dimensions")
}

pub fn shape_to_domain_checked(shape: &[Dim]) -> Result<Domain, String> {
    let mut iters = Vec::with_capacity(shape.len());
    let mut params = Vec::new();
    let mut constraints = Vec::with_capacity(shape.len() * 2);

    for (i, dim) in shape.iter().enumerate() {
        let iter = format!("i{i}");
        iters.push(iter.clone());

        // Lower bound: iter >= 0  ⟹  iter >= 0  (Ineq means expr >= 0)
        constraints.push(Constraint::Ineq(Aff {
            constant: 0,
            terms: vec![(1, PolyVar::Iter(iter.clone()))],
        }));

        // Upper bound: try converting dim to affine first.
        if let Some(bound_aff) = dim_to_aff(dim) {
            // iter < bound  ⟹  bound - 1 - iter >= 0
            let ub = bound_aff.sub(&Aff::iter_var(&iter)).add(&Aff::constant(-1));
            // Collect any params from the affine bound.
            for (_, pv) in &ub.terms {
                if let PolyVar::Param(sym) = pv
                    && !params.contains(sym)
                {
                    params.push(*sym);
                }
            }
            constraints.push(Constraint::Ineq(ub));
        } else {
            return Err(format!(
                "non-affine dimension at axis {i} is not supported by native poly domain"
            ));
        }
    }

    Ok(Domain {
        iters,
        params,
        constraints,
    })
}
