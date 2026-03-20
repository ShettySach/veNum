use anyhow::{anyhow, Result};

use super::super::context::Context;
use super::super::dtype::{Buffer, DType, Scalar};
use super::Tensor;

impl Tensor {
    pub fn from_slice(cx: &Context, data: &[f32], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_f32_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self {
            graph,
            kernel_cache: cx.kernel_cache(),
            plan_cache: cx.plan_cache(),
            buffer_pool: cx.buffer_pool(),
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
            graph,
            kernel_cache: cx.kernel_cache(),
            plan_cache: cx.plan_cache(),
            buffer_pool: cx.buffer_pool(),
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
            graph,
            kernel_cache: cx.kernel_cache(),
            plan_cache: cx.plan_cache(),
            buffer_pool: cx.buffer_pool(),
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
            graph,
            kernel_cache: cx.kernel_cache(),
            plan_cache: cx.plan_cache(),
            buffer_pool: cx.buffer_pool(),
            id,
            shape,
            dtype: DType::I64,
        }
    }

    pub fn constant(cx: &Context, value: f32, shape: Vec<usize>) -> Self {
        let graph = cx.graph();
        let id = graph
            .lock()
            .unwrap()
            .constant(Scalar::F32(value), shape.clone());
        Self {
            graph,
            kernel_cache: cx.kernel_cache(),
            plan_cache: cx.plan_cache(),
            buffer_pool: cx.buffer_pool(),
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
            graph,
            kernel_cache: cx.kernel_cache(),
            plan_cache: cx.plan_cache(),
            buffer_pool: cx.buffer_pool(),
            id,
            shape,
            dtype,
        }
    }

    pub fn arange(cx: &Context, start: f32, end: f32, step: f32) -> Result<Self> {
        use std::cmp::Ordering;
        use std::iter::successors;

        let ascending = match step
            .partial_cmp(&0.0)
            .ok_or(anyhow!("Cannot compare step value"))?
        {
            Ordering::Greater if end > start => Ok(true),
            Ordering::Less if start > end => Ok(false),
            Ordering::Greater => Err(anyhow!("step is positive but end <= start")),
            Ordering::Less => Err(anyhow!("step is negative but start <= end")),
            Ordering::Equal => Err(anyhow!("step cannot be zero")),
        }?;

        let sign = if ascending { 1.0 } else { -1.0 };
        let scaled_step = step * sign;
        let scaled_start = start * sign;
        let scaled_end = end * sign;

        let data: Vec<f32> = successors(Some(scaled_start), |&prev| {
            let curr = prev + scaled_step;
            let cond = scaled_end > curr;
            (ascending == cond).then_some(curr)
        })
        .map(|v| v * sign)
        .collect();

        let shape = vec![data.len()];
        Ok(Self::from_slice(cx, &data, shape))
    }
}
