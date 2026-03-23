//! ShapeTracker index computation for Cranelift IR.

use cranelift::prelude::{types, FunctionBuilder, InstBuilder, Value};

use crate::core::shared::shape_tracker::ShapeTracker;

/// Decompose a flat index `i` into per-dimension indices for `output_shape`.
///
/// For shape [s0, s1, s2]:
///   d2 = i % s2
///   d1 = (i / s2) % s1
///   d0 = (i / (s2 * s1)) % s0
pub fn decompose_flat_index(
    builder: &mut FunctionBuilder,
    flat_idx: Value,
    shape: &[usize],
) -> Vec<Value> {
    let rank = shape.len();
    let mut indices = vec![flat_idx; rank]; // placeholder
    let mut remaining = flat_idx;

    for d in (0..rank).rev() {
        let size = builder.ins().iconst(types::I64, shape[d] as i64);
        let idx = builder.ins().urem(remaining, size);
        indices[d] = idx;
        if d > 0 {
            remaining = builder.ins().udiv(remaining, size);
        }
    }

    indices
}

/// Compute the byte offset into a source buffer using a ShapeTracker.
///
/// flat_offset = sum(dim_indices[d] * strides[d]) + offset
/// byte_offset = flat_offset * elem_size
pub fn compute_tracker_byte_offset(
    builder: &mut FunctionBuilder,
    dim_indices: &[Value],
    tracker: &ShapeTracker,
    elem_size: i64,
) -> Value {
    let mut sum = builder.ins().iconst(types::I64, tracker.offset as i64);
    let tracker_rank = tracker.strides.len();
    let idx_rank = dim_indices.len();
    let pad_front = tracker_rank.saturating_sub(idx_rank);
    let drop_front = idx_rank.saturating_sub(tracker_rank);
    let zero = builder.ins().iconst(types::I64, 0);

    for (d, &stride) in tracker.strides.iter().enumerate() {
        if stride == 0 {
            // Broadcast dimension, contributes nothing.
            continue;
        }
        let logical_idx = if tracker_rank >= idx_rank {
            if d < pad_front {
                zero
            } else {
                dim_indices[d - pad_front]
            }
        } else {
            dim_indices[d + drop_front]
        };
        let stride_val = builder.ins().iconst(types::I64, stride as i64);
        let contribution = builder.ins().imul(logical_idx, stride_val);
        sum = builder.ins().iadd(sum, contribution);
    }

    let size_val = builder.ins().iconst(types::I64, elem_size);
    builder.ins().imul(sum, size_val)
}

/// Compute a flat index from per-dimension indices and shape.
///
/// For shape [s0, s1, s2] and indices [d0, d1, d2]:
///   flat = d0 * (s1 * s2) + d1 * s2 + d2
pub fn flatten_multi_index(
    builder: &mut FunctionBuilder,
    indices: &[Value],
    shape: &[usize],
) -> Value {
    let rank = shape.len();
    let mut result = builder.ins().iconst(types::I64, 0);

    let mut stride = 1i64;
    for d in (0..rank).rev() {
        let stride_val = builder.ins().iconst(types::I64, stride);
        let contribution = builder.ins().imul(indices[d], stride_val);
        result = builder.ins().iadd(result, contribution);
        stride *= shape[d] as i64;
    }

    result
}
