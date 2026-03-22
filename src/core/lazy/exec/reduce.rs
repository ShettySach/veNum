use anyhow::{bail, Result};
use std::cmp::Ordering;

use crate::core::iters::Indexer;
use crate::core::lazy::dtype::Buffer;
use crate::core::lazy::graph::Op;

pub(crate) fn execute_reduce_op_typed(
    op: &Op,
    input_buf: &Buffer,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Buffer> {
    match input_buf {
        Buffer::F32(v) => Ok(Buffer::from_f32_vec(execute_reduce_op_f(
            op,
            v,
            input_shape,
            output_shape,
        )?)),
        Buffer::F64(v) => Ok(Buffer::from_f64_vec(execute_reduce_op_f(
            op,
            v,
            input_shape,
            output_shape,
        )?)),
        Buffer::I32(v) => Ok(Buffer::from_i32_vec(execute_reduce_op_i(
            op,
            v,
            input_shape,
            output_shape,
        )?)),
        Buffer::I64(v) => Ok(Buffer::from_i64_vec(execute_reduce_op_i(
            op,
            v,
            input_shape,
            output_shape,
        )?)),
    }
}

pub(crate) fn reduced_shape(
    shape: &[usize],
    dimensions: &[usize],
    keepdims: bool,
) -> Result<Vec<usize>> {
    validate_reduce_dims(shape, dimensions)?;

    let mut is_reduce_dim = vec![false; shape.len()];
    for &d in dimensions {
        is_reduce_dim[d] = true;
    }

    let mut out = Vec::new();
    for (d, &size) in shape.iter().enumerate() {
        if is_reduce_dim[d] {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(size);
        }
    }

    if out.is_empty() {
        out.push(1);
    }

    Ok(out)
}

pub(crate) fn validate_reduce_dims(shape: &[usize], dimensions: &[usize]) -> Result<()> {
    let mut seen = vec![false; shape.len()];
    for &d in dimensions {
        if d >= shape.len() {
            bail!(
                "reduce dimension {} out of bounds for rank {}",
                d,
                shape.len()
            );
        }
        if seen[d] {
            bail!("duplicate reduce dimension {}", d);
        }
        seen[d] = true;
    }
    Ok(())
}

fn execute_reduce_op_f<T>(
    op: &Op,
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + PartialOrd,
    T: std::iter::Sum + std::iter::Product,
{
    match op {
        Op::Sum(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().sum(),
        ),
        Op::Prod(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().product(),
        ),
        Op::Max(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| {
                slice
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(T::default())
            },
        ),
        Op::Min(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| {
                slice
                    .iter()
                    .copied()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(T::default())
            },
        ),
        _ => bail!("execute_reduce_op called with non-reduce op: {:?}", op),
    }
}

fn execute_reduce_op_i<T>(
    op: &Op,
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Ord,
    T: std::iter::Sum + std::iter::Product,
{
    match op {
        Op::Sum(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().sum(),
        ),
        Op::Prod(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().product(),
        ),
        Op::Max(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().max().unwrap_or(T::default()),
        ),
        Op::Min(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().min().unwrap_or(T::default()),
        ),
        _ => bail!("execute_reduce_op called with non-reduce op: {:?}", op),
    }
}

fn execute_reduce<T: Copy>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    dimensions: &[usize],
    keepdims: bool,
    reducer: impl Fn(&[T]) -> T,
) -> Result<Vec<T>> {
    validate_reduce_dims(input_shape, dimensions)?;

    let expected_shape = reduced_shape(input_shape, dimensions, keepdims)?;
    if expected_shape != output_shape {
        bail!(
            "reduce output shape mismatch: expected {:?}, got {:?}",
            expected_shape,
            output_shape
        );
    }

    let rank = input_shape.len();
    let mut is_reduce_dim = vec![false; rank];
    for &d in dimensions {
        is_reduce_dim[d] = true;
    }

    let reduce_numel: usize = dimensions.iter().map(|&d| input_shape[d]).product();
    let mut out = Vec::with_capacity(output_shape.iter().product());

    let mut fixed = vec![0usize; rank];
    let mut current = vec![0usize; rank];

    // Buffer reused per reduction to avoid reallocating vectors.
    let mut reduced_values: Vec<T> = Vec::with_capacity(reduce_numel);

    for out_idx in Indexer::new(output_shape) {
        if keepdims {
            for d in 0..rank {
                if !is_reduce_dim[d] {
                    fixed[d] = out_idx[d];
                }
            }
        } else {
            let mut out_pos = 0usize;
            for d in 0..rank {
                if !is_reduce_dim[d] {
                    fixed[d] = out_idx[out_pos];
                    out_pos += 1;
                }
            }
        }

        reduced_values.clear();

        // Iteratively walk the reduced dimensions instead of recursive collection.
        current.copy_from_slice(&fixed);

        let reduce_dims: Vec<usize> = (0..rank).filter(|&d| is_reduce_dim[d]).collect();

        if reduce_dims.is_empty() {
            let off = idx_to_offset(&current, input_shape);
            reduced_values.push(input_data[off]);
        } else {
            loop {
                let off = idx_to_offset(&current, input_shape);
                reduced_values.push(input_data[off]);

                // Increment the last reduced dimension, carrying over as needed.
                let mut carry = true;
                for &d in reduce_dims.iter().rev() {
                    if carry {
                        current[d] += 1;
                        if current[d] >= input_shape[d] {
                            current[d] = 0;
                            carry = true;
                        } else {
                            carry = false;
                            break;
                        }
                    }
                }

                if carry {
                    break;
                }
            }
        }

        out.push(reducer(&reduced_values));
    }

    Ok(out)
}

fn idx_to_offset(index: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut off = 0usize;
    for d in (0..shape.len()).rev() {
        off += index[d] * stride;
        stride *= shape[d];
    }
    off
}
