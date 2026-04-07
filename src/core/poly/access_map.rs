use anyhow::{bail, Result};

use crate::core::hlir::{BufferId, Dim};
use crate::core::llir::MemoryAccess;

use super::domain::{Aff, IterName, PolyVar};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessMap {
    pub buffer: BufferId,
    pub domain_iters: Vec<IterName>,
    pub mapping: Vec<Aff>,
}

pub fn strides_to_access(strides: &[Dim], domain_iters: &[IterName]) -> Result<AccessMap> {
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
        let c = if let Dim::Const(v) = stride {
            *v
        } else {
            return Err(anyhow::anyhow!("symbolic stride is not affine-lowerable"));
        };
        aff.terms.push((c, PolyVar::Iter(iter.clone())));
    }

    Ok(AccessMap {
        buffer: BufferId(0),
        domain_iters: domain_iters.to_vec(),
        mapping: vec![aff],
    })
}

pub fn memory_access_to_access_map(access: &MemoryAccess, loop_vars: &[IterName]) -> AccessMap {
    let mapping = access
        .indices
        .iter()
        .map(|idx| Aff {
            constant: idx.constant,
            terms: idx
                .terms
                .iter()
                .map(|(coeff, v)| match v {
                    crate::core::llir::Var::Loop(name) => {
                        (*coeff, PolyVar::Iter(name.clone().into()))
                    }
                    crate::core::llir::Var::Param(sym) => (*coeff, PolyVar::Param(*sym)),
                })
                .collect(),
        })
        .collect();

    AccessMap {
        buffer: access.buffer,
        domain_iters: loop_vars.to_vec(),
        mapping,
    }
}
