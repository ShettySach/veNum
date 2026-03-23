//! Solid-specific tensor constructors.

use crate::core::shared::dtype::DType;
use crate::core::shared::graph::{Node, Op};
use crate::core::shared::tensor::Tensor;
use crate::core::solid::context::SolidContext;

/// Solid-specific constructor extensions.
impl Tensor<SolidContext> {
    /// Create a symbolic placeholder tensor (no concrete data).
    ///
    /// Placeholders represent inputs that will be provided at execution time.
    /// This is the primary way to define inputs in Solid mode.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cx = SolidContext::new();
    /// let input = Tensor::placeholder(&cx, vec![batch, seq_len, hidden], DType::F32);
    /// ```
    pub fn placeholder(cx: &SolidContext, shape: Vec<usize>, dtype: DType) -> Self {
        let graph = cx.graph();
        let id = graph.lock().unwrap().add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: shape.clone(),
            dtype,
            buffer: None, // No data - symbolic!
        });
        cx.register_input(id);
        Self::new(cx.clone(), id, shape, dtype)
    }
}
