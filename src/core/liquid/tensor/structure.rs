use anyhow::{bail, Result};
use std::sync::Arc;

use crate::core::liquid::context::Context;
use crate::core::shared::dtype::DType;
use crate::core::shared::graph::{Graph, NodeId, Op};

#[derive(Clone)]
pub struct Tensor {
    pub(super) cx: Context,
    pub(super) id: NodeId,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: DType,
}

// Ops helpers

impl Tensor {
    pub(super) fn with_graph_mut<R>(&self, f: impl FnOnce(&mut Graph) -> R) -> R {
        f(&mut self.cx.graph().lock().unwrap())
    }

    pub(super) fn derived(&self, id: NodeId, shape: Vec<usize>) -> Self {
        Self {
            cx: self.cx.clone(),
            id,
            shape,
            dtype: self.dtype,
        }
    }

    pub(super) fn binary_op(&self, rhs: &Tensor, op: Op) -> Result<Tensor> {
        if !Arc::ptr_eq(&self.cx.graph(), &rhs.cx.graph()) {
            bail!("binary_op requires both tensors to share the same Context")
        } else if self.dtype != rhs.dtype {
            bail!(
                "binary_op requires matching dtypes, got {:?} and {:?}",
                self.dtype,
                rhs.dtype
            )
        } else {
            let id = self.with_graph_mut(|g| g.binary(op, self.id, rhs.id));
            Ok(self.derived(id, self.shape.clone()))
        }
    }

    pub(super) fn unary_op(&self, op: Op) -> Tensor {
        let id = self.with_graph_mut(|g| g.unary(op, self.id));
        self.derived(id, self.shape.clone())
    }
}
