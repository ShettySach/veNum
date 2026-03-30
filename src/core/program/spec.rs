//! Tensor specification for compiled programs.

use crate::core::dtype::DType;
use crate::core::graph::NodeId;

/// Specification of a tensor in a compiled program.
///
/// This describes the shape, dtype, and optional name of a tensor
/// without storing its actual data. Used for program inputs/outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Node ID in the computation graph
    pub node_id: NodeId,

    /// Shape of the tensor
    pub shape: Vec<usize>,

    /// Data type of the tensor elements
    pub dtype: DType,

    /// Optional name for debugging/visualization
    pub name: Option<String>,
}

impl TensorSpec {
    /// Create a new tensor specification.
    pub fn new(node_id: NodeId, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            node_id,
            shape,
            dtype,
            name: None,
        }
    }

    /// Create a tensor specification with a name.
    pub fn named(node_id: NodeId, shape: Vec<usize>, dtype: DType, name: String) -> Self {
        Self {
            node_id,
            shape,
            dtype,
            name: Some(name),
        }
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }
}
