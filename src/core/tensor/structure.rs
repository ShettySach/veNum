//! Tensor structure.

use crate::core::dtype::DType;
use crate::core::graph::{Graph, NodeId};

use super::context::Context;

/// A tensor in Venum's computation graph.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) cx: Context,
    pub(crate) id: NodeId,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
}

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
    pub fn context(&self) -> &Context {
        &self.cx
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Create a new tensor (internal use).
    pub(crate) fn new(cx: Context, id: NodeId, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            cx,
            id,
            shape,
            dtype,
        }
    }

    /// Execute a function with mutable access to the graph.
    pub(crate) fn with_graph_mut<R>(&self, f: impl FnOnce(&mut Graph) -> R) -> R {
        f(&mut self.cx.graph().lock().unwrap())
    }

    /// Create a derived tensor with the same context and dtype.
    pub(crate) fn derived(&self, id: NodeId, shape: Vec<usize>) -> Self {
        Self {
            cx: self.cx.clone(),
            id,
            shape,
            dtype: self.dtype,
        }
    }
}
