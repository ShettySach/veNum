//! Tensor structure.

use crate::core::hlir::{DType, Dim, HLIRGraph, NodeId};

use super::context::Context;

/// A tensor in Venum's computation graph.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) cx: Context,
    pub(crate) id: NodeId,
    pub(crate) shape: Vec<Dim>,
    pub(crate) dtype: DType,
}

impl Tensor {
    /// Get the node ID in the computation graph.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[Dim] {
        &self.shape
    }

    /// Get the data type of the tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the context this tensor belongs to.
    pub fn context(&self) -> &Context {
        &self.cx
    }

    /// Get the total number of elements.
    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Create a new tensor (internal use).
    pub(crate) fn new(cx: Context, id: NodeId, shape: Vec<Dim>, dtype: DType) -> Self {
        Self {
            cx,
            id,
            shape,
            dtype,
        }
    }

    /// Execute a function with mutable access to the graph.
    pub(crate) fn with_graph_mut<R>(&self, f: impl FnOnce(&mut HLIRGraph) -> R) -> R {
        f(&mut self.cx.graph().lock().unwrap())
    }

    /// Create a derived tensor with the same context and dtype.
    pub(crate) fn derived(&self, id: NodeId, shape: Vec<Dim>) -> Self {
        Self {
            cx: self.cx.clone(),
            id,
            shape,
            dtype: self.dtype,
        }
    }
}
