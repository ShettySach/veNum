//! Tensor constructors for Solid mode.

use crate::core::shared::dtype::{Buffer, DType, Scalar};
use crate::core::solid::context::SolidContext;
use crate::core::solid::tensor::Tensor;

impl Tensor {
    /// Create a symbolic placeholder tensor (no concrete data).
    ///
    /// Placeholders represent inputs that will be provided at execution time.
    /// This is the primary way to define inputs in Solid mode.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let cx = SolidContext::new();
    /// let input = Tensor::placeholder(&cx, vec![batch, seq_len, hidden], DType::F32);
    /// ```
    pub fn placeholder(cx: &SolidContext, shape: Vec<usize>, dtype: DType) -> Self {
        let graph = cx.graph();
        let id = graph
            .lock()
            .unwrap()
            .add_node(crate::core::shared::graph::Node {
                op: crate::core::shared::graph::Op::Load,
                inputs: vec![],
                shape: shape.clone(),
                dtype,
                buffer: None, // No data - symbolic!
            });
        cx.register_input(id);
        Self {
            cx: cx.clone(),
            id,
            shape,
            dtype,
        }
    }

    /// Create a tensor from a slice of f32 data.
    pub fn from_slice(cx: &SolidContext, data: &[f32], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_f32_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self {
            cx: cx.clone(),
            id,
            shape,
            dtype: DType::F32,
        }
    }

    /// Create a tensor from a slice of f64 data.
    pub fn from_slice_f64(cx: &SolidContext, data: &[f64], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_f64_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self {
            cx: cx.clone(),
            id,
            shape,
            dtype: DType::F64,
        }
    }

    /// Create a tensor from a slice of i32 data.
    pub fn from_slice_i32(cx: &SolidContext, data: &[i32], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_i32_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self {
            cx: cx.clone(),
            id,
            shape,
            dtype: DType::I32,
        }
    }

    /// Create a tensor from a slice of i64 data.
    pub fn from_slice_i64(cx: &SolidContext, data: &[i64], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_i64_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self {
            cx: cx.clone(),
            id,
            shape,
            dtype: DType::I64,
        }
    }

    /// Create a constant tensor with a single value repeated.
    pub fn constant(cx: &SolidContext, value: f32, shape: Vec<usize>) -> Self {
        let graph = cx.graph();
        let id = graph
            .lock()
            .unwrap()
            .constant(Scalar::F32(value), shape.clone());
        Self {
            cx: cx.clone(),
            id,
            shape,
            dtype: DType::F32,
        }
    }

    /// Create a constant tensor from a scalar value.
    pub fn constant_scalar(cx: &SolidContext, value: Scalar, shape: Vec<usize>) -> Self {
        let dtype = value.dtype();
        let graph = cx.graph();
        let id = graph.lock().unwrap().constant(value, shape.clone());
        Self {
            cx: cx.clone(),
            id,
            shape,
            dtype,
        }
    }
}
