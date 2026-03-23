use anyhow::Result;

use crate::core::errors::*;

use crate::core::lazy::context::Context;
use crate::core::lazy::tensor::Tensor;
use crate::core::shared::dtype::{Buffer, DType, Scalar};

impl Tensor {
    pub fn from_slice(cx: &Context, data: &[f32], shape: Vec<usize>) -> Self {
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

    pub fn from_slice_f64(cx: &Context, data: &[f64], shape: Vec<usize>) -> Self {
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

    pub fn from_slice_i32(cx: &Context, data: &[i32], shape: Vec<usize>) -> Self {
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

    pub fn from_slice_i64(cx: &Context, data: &[i64], shape: Vec<usize>) -> Self {
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

    pub fn from_slice_f32_1d(cx: &Context, data: &[f32]) -> Self {
        Self::from_slice(cx, data, vec![data.len()])
    }

    pub fn from_slice_f64_1d(cx: &Context, data: &[f64]) -> Self {
        Self::from_slice_f64(cx, data, vec![data.len()])
    }

    pub fn from_slice_i32_1d(cx: &Context, data: &[i32]) -> Self {
        Self::from_slice_i32(cx, data, vec![data.len()])
    }

    pub fn from_slice_i64_1d(cx: &Context, data: &[i64]) -> Self {
        Self::from_slice_i64(cx, data, vec![data.len()])
    }

    pub fn constant(cx: &Context, value: f32, shape: Vec<usize>) -> Self {
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

    pub fn constant_scalar(cx: &Context, value: Scalar, shape: Vec<usize>) -> Self {
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

    pub fn arange(cx: &Context, start: f32, end: f32, step: f32) -> Result<Self> {
        use std::cmp::Ordering;
        use std::iter::successors;

        let ascending = match step.partial_cmp(&0.0).ok_or(ArangeError::Comparison)? {
            Ordering::Greater if end > start => Ok(true),
            Ordering::Less if start > end => Ok(false),
            Ordering::Greater => Err(ArangeError::Positive),
            Ordering::Less => Err(ArangeError::Negative),
            Ordering::Equal => Err(ArangeError::Zero),
        }?;

        let data: Vec<_> = successors(Some(start), |&prev| {
            let curr = prev + step;
            let cond = end > curr;
            (ascending == cond).then_some(curr)
        })
        .collect();

        let shape = vec![data.len()];
        Ok(Self::from_slice(cx, &data, shape))
    }
}
