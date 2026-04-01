//! Placeholder tensor constructors.

use crate::core::hlir::{DType, Dim, TensorType};
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
    pub fn placeholder(cx: &Context, dtype: DType, shape: Vec<i64>) -> Self {
        let dim_shape: Vec<Dim> = shape.into_iter().map(Dim::constant).collect();
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(
            cx.alloc_buffer_id(),
            TensorType::contiguous(dim_shape.clone(), dtype),
        );
        cx.register_input(id);
        Self::new(cx.clone(), id, dim_shape, dtype)
    }
}
