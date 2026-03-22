use anyhow::{bail, Result};

use super::super::dtype::Buffer;
use super::super::graph::Op;
use crate::core::iters::Indexer;

pub(crate) fn execute_shape_op_typed(
    op: &Op,
    input_buf: &Buffer,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Buffer> {
    // Handle Pad here to avoid unsafe generic scalar casting.
    if let Op::Pad(constant, padding) = op {
        return match input_buf {
            Buffer::F32(v) => {
                let fill = constant.to_f64() as f32;
                Ok(Buffer::from_f32_vec(execute_pad(
                    v,
                    input_shape,
                    output_shape,
                    fill,
                    padding,
                )?))
            }
            Buffer::F64(v) => {
                let fill = constant.to_f64();
                Ok(Buffer::from_f64_vec(execute_pad(
                    v,
                    input_shape,
                    output_shape,
                    fill,
                    padding,
                )?))
            }
            Buffer::I32(v) => {
                let fill = constant.to_f64() as i32;
                Ok(Buffer::from_i32_vec(execute_pad(
                    v,
                    input_shape,
                    output_shape,
                    fill,
                    padding,
                )?))
            }
            Buffer::I64(v) => {
                let fill = constant.to_f64() as i64;
                Ok(Buffer::from_i64_vec(execute_pad(
                    v,
                    input_shape,
                    output_shape,
                    fill,
                    padding,
                )?))
            }
        };
    }

    match input_buf {
        Buffer::F32(v) => Ok(Buffer::from_f32_vec(execute_shape_op(
            op,
            v,
            input_shape,
            output_shape,
        )?)),
        Buffer::F64(v) => Ok(Buffer::from_f64_vec(execute_shape_op(
            op,
            v,
            input_shape,
            output_shape,
        )?)),
        Buffer::I32(v) => Ok(Buffer::from_i32_vec(execute_shape_op(
            op,
            v,
            input_shape,
            output_shape,
        )?)),
        Buffer::I64(v) => Ok(Buffer::from_i64_vec(execute_shape_op(
            op,
            v,
            input_shape,
            output_shape,
        )?)),
    }
}

fn execute_shape_op<T: Copy + Default>(
    op: &Op,
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>> {
    match op {
        Op::Reshape => execute_reshape(input_data, input_shape, output_shape),
        Op::Permute(perm) => execute_permute(input_data, input_shape, output_shape, perm),
        Op::Transpose(d1, d2) => execute_transpose(input_data, input_shape, output_shape, *d1, *d2),
        Op::Expand => execute_expand(input_data, input_shape, output_shape),
        Op::Slice(ranges) => execute_slice(input_data, input_shape, output_shape, ranges),
        Op::Flip(flips) => execute_flip(input_data, input_shape, output_shape, flips),
        Op::Squeeze => execute_reshape(input_data, input_shape, output_shape),
        Op::Unsqueeze(_) => execute_reshape(input_data, input_shape, output_shape),
        Op::Pad(_, _) => bail!("execute_shape_op Pad should be handled typed"),
        _ => bail!("execute_shape_op called with non-shape op: {:?}", op),
    }
}

fn execute_reshape<T: Copy>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>> {
    let in_numel: usize = input_shape.iter().product();
    let out_numel: usize = output_shape.iter().product();
    if in_numel != out_numel || input_data.len() != in_numel {
        bail!(
            "invalid reshape numel: input {:?}, output {:?}",
            input_shape,
            output_shape
        );
    }
    Ok(input_data.to_vec())
}

fn execute_permute<T: Copy + Default>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    perm: &[usize],
) -> Result<Vec<T>> {
    let rank = input_shape.len();
    if perm.len() != rank || output_shape.len() != rank {
        bail!("invalid permute rank");
    }
    let mut out = vec![T::default(); output_shape.iter().product()];

    let mut in_idx = vec![0usize; rank];
    for out_idx in Indexer::new(output_shape) {
        for (out_dim, &in_dim) in perm.iter().enumerate() {
            in_idx[in_dim] = out_idx[out_dim];
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_transpose<T: Copy + Default>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    dim_1: usize,
    dim_2: usize,
) -> Result<Vec<T>> {
    if dim_1 >= input_shape.len()
        || dim_2 >= input_shape.len()
        || input_shape.len() != output_shape.len()
    {
        bail!("invalid transpose dims");
    }

    let mut perm: Vec<usize> = (0..input_shape.len()).collect();
    perm.swap(dim_1, dim_2);
    execute_permute(input_data, input_shape, output_shape, &perm)
}

fn execute_expand<T: Copy + Default>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>> {
    let rank = input_shape.len();
    if rank != output_shape.len() {
        bail!("expand rank mismatch");
    }

    let mut out = vec![T::default(); output_shape.iter().product()];
    let mut in_idx = vec![0usize; rank];

    for out_idx in Indexer::new(output_shape) {
        for d in 0..rank {
            if input_shape[d] == output_shape[d] {
                in_idx[d] = out_idx[d];
            } else if input_shape[d] == 1 {
                in_idx[d] = 0;
            } else {
                bail!(
                    "cannot expand dim {} from {} to {}",
                    d,
                    input_shape[d],
                    output_shape[d]
                );
            }
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_slice<T: Copy + Default>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<T>> {
    let rank = input_shape.len();
    if ranges.len() != rank || output_shape.len() != rank {
        bail!("slice rank mismatch");
    }

    let mut out = vec![T::default(); output_shape.iter().product()];
    let mut in_idx = vec![0usize; rank];

    for out_idx in Indexer::new(output_shape) {
        for d in 0..rank {
            let (start, end_raw) = ranges[d];
            let end = if end_raw == 0 {
                input_shape[d]
            } else {
                end_raw
            };
            if start > end || end > input_shape[d] {
                bail!("invalid slice range {:?} for dim {}", ranges[d], d);
            }
            in_idx[d] = start + out_idx[d];
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_flip<T: Copy + Default>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    flips: &[usize],
) -> Result<Vec<T>> {
    if input_shape != output_shape {
        bail!("flip must preserve shape");
    }

    let rank = input_shape.len();
    let mut flip_mask = vec![false; rank];
    for &d in flips {
        if d >= rank {
            bail!("flip dim {} out of range", d);
        }
        flip_mask[d] = true;
    }

    let mut out = vec![T::default(); output_shape.iter().product()];
    let mut in_idx = vec![0usize; rank];

    for out_idx in Indexer::new(output_shape) {
        for d in 0..rank {
            in_idx[d] = if flip_mask[d] {
                input_shape[d] - 1 - out_idx[d]
            } else {
                out_idx[d]
            };
        }

        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_pad<T: Copy>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    constant: T,
    padding: &[(usize, usize)],
) -> Result<Vec<T>> {
    let rank = input_shape.len();
    if rank != output_shape.len() {
        bail!("pad rank mismatch");
    }

    let mut pad = padding.to_vec();
    pad.resize(rank, (0, 0));

    let mut expected = Vec::with_capacity(rank);
    for d in 0..rank {
        expected.push(pad[d].0 + input_shape[d] + pad[d].1);
    }
    if expected != output_shape {
        bail!(
            "pad output shape mismatch: expected {:?}, got {:?}",
            expected,
            output_shape
        );
    }

    let mut out = vec![constant; output_shape.iter().product()];
    let mut out_idx = vec![0usize; rank];

    for in_idx in Indexer::new(input_shape) {
        for d in 0..rank {
            out_idx[d] = pad[d].0 + in_idx[d];
        }

        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
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
