use crate::core::hlir::{Dim, Symbol};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Aff {
    pub constant: i64,
    pub terms: Vec<(i64, PolyVar)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

pub fn shape_to_domain(shape: &[Dim]) -> Domain {
    let mut iters = Vec::with_capacity(shape.len());
    let mut params = Vec::new();
    let mut constraints = Vec::with_capacity(shape.len() * 2);

    for (i, dim) in shape.iter().enumerate() {
        let iter = format!("i{i}");
        iters.push(iter.clone());

        constraints.push(Constraint::Ineq(Aff {
            constant: 0,
            terms: vec![(1, PolyVar::Iter(iter.clone()))],
        }));

        match dim {
            Dim::Const(v) => constraints.push(Constraint::Ineq(Aff {
                constant: v - 1,
                terms: vec![(-1, PolyVar::Iter(iter))],
            })),
            Dim::Sym(sym) => {
                params.push(*sym);
                constraints.push(Constraint::Ineq(Aff {
                    constant: -1,
                    terms: vec![(1, PolyVar::Param(*sym)), (-1, PolyVar::Iter(iter))],
                }));
            }
            _ => {
                constraints.push(Constraint::Ineq(Aff {
                    constant: -1,
                    terms: vec![(-1, PolyVar::Iter(iter))],
                }));
            }
        }
    }

    Domain {
        iters,
        params,
        constraints,
    }
}
