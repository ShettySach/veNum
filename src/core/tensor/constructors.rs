//! Tensor constructors.

use crate::core::dtype::{Buffer, DType, Scalar};

use super::context::Context;
use super::structure::Tensor;

impl Tensor {
    // ==================== From Slice ====================

    /// Create a tensor from a slice of f32 data.
    pub fn from_slice(cx: &Context, data: &[f32], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_f32_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self::new(cx.clone(), id, shape, DType::F32)
    }

    /// Create a tensor from a slice of f64 data.
    pub fn from_slice_f64(cx: &Context, data: &[f64], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_f64_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self::new(cx.clone(), id, shape, DType::F64)
    }

    /// Create a tensor from a slice of i32 data.
    pub fn from_slice_i32(cx: &Context, data: &[i32], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_i32_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self::new(cx.clone(), id, shape, DType::I32)
    }

    /// Create a tensor from a slice of i64 data.
    pub fn from_slice_i64(cx: &Context, data: &[i64], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_i64_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self::new(cx.clone(), id, shape, DType::I64)
    }

    // ==================== 1D Convenience ====================

    /// Create a 1D tensor from a slice of f32 data.
    pub fn from_slice_f32_1d(cx: &Context, data: &[f32]) -> Self {
        Self::from_slice(cx, data, vec![data.len()])
    }

    /// Create a 1D tensor from a slice of f64 data.
    pub fn from_slice_f64_1d(cx: &Context, data: &[f64]) -> Self {
        Self::from_slice_f64(cx, data, vec![data.len()])
    }

    /// Create a 1D tensor from a slice of i32 data.
    pub fn from_slice_i32_1d(cx: &Context, data: &[i32]) -> Self {
        Self::from_slice_i32(cx, data, vec![data.len()])
    }

    /// Create a 1D tensor from a slice of i64 data.
    pub fn from_slice_i64_1d(cx: &Context, data: &[i64]) -> Self {
        Self::from_slice_i64(cx, data, vec![data.len()])
    }

    // ==================== Constants ====================

    /// Create a constant tensor with a single f32 value repeated.
    pub fn constant(cx: &Context, value: f32, shape: Vec<usize>) -> Self {
        let graph = cx.graph();
        let id = graph
            .lock()
            .unwrap()
            .constant(Scalar::F32(value), shape.clone());
        Self::new(cx.clone(), id, shape, DType::F32)
    }

    /// Create a constant tensor from a scalar value.
    pub fn constant_scalar(cx: &Context, value: Scalar, shape: Vec<usize>) -> Self {
        let dtype = value.dtype();
        let graph = cx.graph();
        let id = graph.lock().unwrap().constant(value, shape.clone());
        Self::new(cx.clone(), id, shape, dtype)
    }
}
