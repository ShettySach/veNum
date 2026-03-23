//! Tensor structure for Solid mode.

use anyhow::{bail, Result};
use std::sync::Arc;

use crate::core::shared::dtype::DType;
use crate::core::shared::graph::{Graph, NodeId, Op};
use crate::core::solid::context::SolidContext;

/// A tensor in Solid (AOT) execution mode.
///
/// Unlike Liquid tensors which can be immediately realized, Solid tensors:
/// - Are purely symbolic (graph nodes)
/// - Cannot be executed individually
/// - Must be compiled as part of a whole program via `compile()`
///
/// Solid tensors support the same operations as Liquid tensors but defer
/// all execution until the entire graph is compiled.
#[derive(Clone)]
pub struct Tensor {
    pub(super) cx: SolidContext,
    pub(super) id: NodeId,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: DType,
}

// Public accessors
impl Tensor {
    /// Get the node ID in the computation graph.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the data type of the tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the context this tensor belongs to.
    pub fn context(&self) -> &SolidContext {
        &self.cx
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// Internal helpers for operations
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
        if !Arc::ptr_eq(self.cx.graph(), rhs.cx.graph()) {
            bail!("binary_op requires both tensors to share the same SolidContext")
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
