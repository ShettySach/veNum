use anyhow::{bail, Result};

use crate::core::hlir::Dim;
use crate::core::llir::MemoryAccess;

use super::domain::{Aff, PolyVar};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessMap {
    pub domain_iters: Vec<String>,
    pub mapping: Vec<Aff>,
}

pub fn strides_to_access(strides: &[Dim], domain_iters: &[String]) -> Result<AccessMap> {
    if strides.len() != domain_iters.len() {
        bail!(
            "strides/domain rank mismatch: {} vs {}",
            strides.len(),
            domain_iters.len()
        );
    }

    let mut aff = Aff {
        constant: 0,
        terms: Vec::with_capacity(strides.len()),
    };

    for (iter, stride) in domain_iters.iter().zip(strides.iter()) {
        let c = stride
            .as_const()
            .ok_or_else(|| anyhow::anyhow!("symbolic stride is not affine-lowerable"))?;
        aff.terms.push((c, PolyVar::Iter(iter.clone())));
    }

    Ok(AccessMap {
        domain_iters: domain_iters.to_vec(),
        mapping: vec![aff],
    })
}

pub fn memory_access_to_access_map(access: &MemoryAccess, loop_vars: &[String]) -> AccessMap {
    let mapping = access
        .indices
        .iter()
        .map(|idx| Aff {
            constant: idx.constant,
            terms: idx
                .terms
                .iter()
                .filter_map(|(coeff, v)| match v {
                    crate::core::llir::Var::Loop(name) => {
                        Some((*coeff, PolyVar::Iter(name.clone())))
                    }
                    crate::core::llir::Var::Param(sym) => Some((*coeff, PolyVar::Param(*sym))),
                })
                .collect(),
        })
        .collect();

    AccessMap {
        domain_iters: loop_vars.to_vec(),
        mapping,
    }
}
