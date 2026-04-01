use crate::core::hlir::Symbol;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineExpr {
    pub constant: i64,
    pub terms: Vec<(i64, Var)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
}
