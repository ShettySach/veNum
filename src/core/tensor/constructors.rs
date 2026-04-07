//! Tensor constructors.

use crate::core::hlir::{Dim, Scalar};

use super::context::Context;
use super::structure::Tensor;

impl Tensor {
    // ==================== Constants ====================

    /// Create a constant tensor with a scalar repeated over shape.
    pub fn constant(cx: &Context, value: f32, shape: Vec<i64>) -> Self {
        Self::constant_scalar(cx, Scalar::F32(value), shape)
    }

    /// Create a constant tensor from a scalar value.
    pub fn constant_scalar(cx: &Context, value: Scalar, shape: Vec<i64>) -> Self {
        let dim_shape: Vec<Dim> = shape.into_iter().map(Dim::Const).collect();
        let dtype = value.dtype();
        let graph = cx.graph();
        let id = graph
            .lock()
            .unwrap()
            .constant(value, dim_shape.clone(), dtype);
        Self::new(cx.clone(), id, dim_shape, dtype)
    }
}
