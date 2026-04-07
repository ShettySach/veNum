use std::fmt;

use crate::core::hlir::Symbol;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineExpr {
    pub constant: i64,
    pub terms: Vec<(i64, Var)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Var {
    Loop(String),
    Param(Symbol),
}

impl AffineExpr {
    pub fn constant(v: i64) -> Self {
        Self {
            constant: v,
            terms: Vec::new(),
        }
    }

    pub fn with_term(mut self, coeff: i64, var: Var) -> Self {
        self.terms.push((coeff, var));
        self
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
            terms: self.terms.iter().map(|(c, v)| (c * factor, v.clone())).collect(),
        }
    }

    /// Canonicalize: combine like terms, drop zeros, sort deterministically.
    pub fn canonicalize(&mut self) {
        // Combine like terms.
        let mut merged: Vec<(i64, Var)> = Vec::new();
        for (coeff, var) in self.terms.drain(..) {
            if let Some(entry) = merged.iter_mut().find(|(_, v)| *v == var) {
                entry.0 += coeff;
            } else {
                merged.push((coeff, var));
            }
        }
        // Drop zero coefficients.
        merged.retain(|(c, _)| *c != 0);
        // Sort deterministically.
        merged.sort_by(|(_, a), (_, b)| a.cmp(b));
        self.terms = merged;
    }

    /// Return a canonicalized copy.
    pub fn canonicalized(&self) -> Self {
        let mut copy = self.clone();
        copy.canonicalize();
        copy
    }

    /// Return the coefficient for a given variable, or 0 if absent.
    pub fn coefficient_of(&self, var: &Var) -> i64 {
        self.terms
            .iter()
            .filter(|(_, v)| v == var)
            .map(|(c, _)| c)
            .sum()
    }

    /// If this expression is a pure constant (no variable terms), return it.
    pub fn as_const_value(&self) -> Option<i64> {
        if self.terms.is_empty() {
            Some(self.constant)
        } else {
            None
        }
    }

    /// Substitute all occurrences of `var` with the expression `replacement`.
    pub fn substitute(&self, var: &Var, replacement: &AffineExpr) -> Self {
        let coeff = self.coefficient_of(var);
        if coeff == 0 {
            return self.clone();
        }
        let remaining: Vec<(i64, Var)> = self
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
}

impl fmt::Display for AffineExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "{}", self.constant);
        }
        let mut first = true;
        for (coeff, var) in &self.terms {
            if first {
                if *coeff == 1 {
                    write!(f, "{var}")?;
                } else if *coeff == -1 {
                    write!(f, "-{var}")?;
                } else {
                    write!(f, "{coeff}*{var}")?;
                }
                first = false;
            } else if *coeff == 1 {
                write!(f, " + {var}")?;
            } else if *coeff == -1 {
                write!(f, " - {var}")?;
            } else if *coeff > 0 {
                write!(f, " + {coeff}*{var}")?;
            } else {
                write!(f, " - {}*{var}", -coeff)?;
            }
        }
        if self.constant > 0 {
            write!(f, " + {}", self.constant)?;
        } else if self.constant < 0 {
            write!(f, " - {}", -self.constant)?;
        }
        Ok(())
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Var::Loop(name) => write!(f, "{name}"),
            Var::Param(sym) => write!(f, "p{}", sym.0),
        }
    }
}
