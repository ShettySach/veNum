//! Placeholder tensor constructors.

use crate::core::dtype::DType;
use crate::core::graph::{Node, Op};
use crate::core::tensor::{Context, Tensor};

impl Tensor {
    /// Create a symbolic placeholder tensor (no concrete data).
    ///
    /// Placeholders represent inputs that will be provided at execution time.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cx = Context::new();
    /// let input = Tensor::placeholder(&cx, DType::F32, vec![batch, seq_len, hidden]);
    /// ```
    pub fn placeholder(cx: &Context, dtype: DType, shape: Vec<usize>) -> Self {
        let graph = cx.graph();
        let id = graph.lock().unwrap().add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: shape.clone(),
            dtype,
            buffer: None,
        });
        cx.register_input(id);
        Self::new(cx.clone(), id, shape, dtype)
    }
}
