//! Kernel loop emission for Cranelift IR.
//!
//! This module provides functions to emit elementwise and reduce kernel loops.

use std::collections::HashMap;

use anyhow::Result;
use cranelift::prelude::{types, FunctionBuilder, InstBuilder, IntCC, MemFlags, Value};

use crate::core::codegen::expr::{build_expression, ExprBuildContext};
use crate::core::codegen::math::MathFuncRefs;
use crate::core::codegen::tracker::{
    compute_tracker_byte_offset, decompose_flat_index, flatten_multi_index,
};
use crate::core::dtype::DType;
use crate::core::graph::{Graph, NodeId};

use super::reduce::{emit_reduce_combine, emit_reduce_identity, ReduceKind};

/// Context for building kernel loops.
pub struct KernelBuilderContext<'a> {
    pub graph: &'a Graph,
    pub input_ptrs: &'a [Value],
    pub input_index: &'a HashMap<NodeId, usize>,
    pub out_ptr: Value,
    pub n_param: Value,
    pub math_refs: &'a MathFuncRefs,
    pub has_trackers: bool,
    pub dtype: DType,
    pub cl_type: types::Type,
    pub elem_size: i64,
}

/// Kernel metadata extracted from a FusedKernel.
pub struct KernelEmitInfo<'a> {
    pub expr_root: NodeId,
    pub output_shape: &'a [usize],
    pub iter_shape: &'a [usize],
    pub input_trackers: &'a HashMap<NodeId, crate::core::shape_tracker::ShapeTracker>,
    pub shape_source_map: &'a HashMap<NodeId, NodeId>,
    pub has_noncontiguous_trackers: bool,
    pub output_tracker: Option<&'a crate::core::shape_tracker::ShapeTracker>,
    pub reduce: Option<ReduceSpec<'a>>,
}

/// Reduce specification for emit functions.
pub struct ReduceSpec<'a> {
    pub op: &'a ReduceKind,
    pub dims: &'a [usize],
    pub keepdims: bool,
}

/// Emit the body of a pure elementwise kernel.
pub fn emit_elementwise_kernel(
    kernel_ctx: &KernelBuilderContext<'_>,
    emit_info: &KernelEmitInfo<'_>,
    builder: &mut FunctionBuilder,
) -> Result<()> {
    let graph = kernel_ctx.graph;

    // Loop: for i in 0..n
    let loop_header = builder.create_block();
    builder.append_block_param(loop_header, types::I64);
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    // i = 0
    let zero = builder.ins().iconst(types::I64, 0);
    builder.ins().jump(loop_header, &[zero]);

    // Loop header: check i < n
    builder.switch_to_block(loop_header);
    let i = builder.block_params(loop_header)[0];
    let cmp = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, i, kernel_ctx.n_param);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    // Loop body
    builder.switch_to_block(loop_body);
    builder.seal_block(loop_body);

    // Byte offset for output = i * elem_size.
    let elem_size_val = builder.ins().iconst(types::I64, kernel_ctx.elem_size);
    let byte_offset = builder.ins().imul(i, elem_size_val);

    // Check if any tracker is non-contiguous and actually needs index
    // decomposition. Contiguous trackers can just use the flat byte offset.
    let output_tracker_noncontig = emit_info
        .output_tracker
        .map(|t| !t.is_contiguous())
        .unwrap_or(false);
    let needs_decompose = (kernel_ctx.has_trackers && emit_info.has_noncontiguous_trackers)
        || output_tracker_noncontig;

    // If we have non-contiguous trackers, decompose flat index `i` into
    // multi-dim indices. Contiguous trackers reuse the flat byte offset.
    let dim_indices = if needs_decompose {
        Some(decompose_flat_index(builder, i, emit_info.output_shape))
    } else {
        None
    };

    // Pre-compute per-input byte offsets from trackers.
    let mut tracked_byte_offsets: HashMap<NodeId, Value> = HashMap::new();
    for (&node_id, tracker) in emit_info.input_trackers {
        if tracker.is_contiguous() {
            // Contiguous tracker: flat offset is equivalent, no decomposition needed.
            tracked_byte_offsets.insert(node_id, byte_offset);
        } else if let Some(ref dims) = dim_indices {
            let tracked_offset =
                compute_tracker_byte_offset(builder, dims, tracker, kernel_ctx.elem_size);
            tracked_byte_offsets.insert(node_id, tracked_offset);
        }
    }

    let expr_ctx = ExprBuildContext {
        graph,
        input_ptrs: kernel_ctx.input_ptrs,
        input_index: kernel_ctx.input_index,
        math: kernel_ctx.math_refs,
        tracked_byte_offsets: &tracked_byte_offsets,
        shape_source_map: emit_info.shape_source_map,
        dtype: kernel_ctx.dtype,
        cl_type: kernel_ctx.cl_type,
    };

    let result = build_expression(&expr_ctx, emit_info.expr_root, builder, byte_offset)?;

    // Store result to out[...], applying any forward-fused output tracker.
    let out_byte_offset = if let Some(tracker) = emit_info.output_tracker {
        if tracker.is_contiguous() {
            byte_offset
        } else if let Some(ref dims) = dim_indices {
            compute_tracker_byte_offset(builder, dims, tracker, kernel_ctx.elem_size)
        } else {
            let dims = decompose_flat_index(builder, i, emit_info.output_shape);
            compute_tracker_byte_offset(builder, &dims, tracker, kernel_ctx.elem_size)
        }
    } else {
        byte_offset
    };

    let out_addr = builder.ins().iadd(kernel_ctx.out_ptr, out_byte_offset);
    builder.ins().store(MemFlags::new(), result, out_addr, 0);

    // i += 1, jump back to header
    let one = builder.ins().iconst(types::I64, 1);
    let i_next = builder.ins().iadd(i, one);
    builder.ins().jump(loop_header, &[i_next]);

    // Now seal loop_header - both predecessors (entry, loop_body) are complete.
    builder.seal_block(loop_header);

    // Exit
    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_exit);
    builder.ins().return_(&[]);

    Ok(())
}

/// Emit the body of a reduce-fused kernel.
///
/// Structure:
///   for out_i in 0..output_numel:
///     acc = identity
///     for red_i in 0..reduce_numel:
///       iter_idx = compose(out_idx, red_idx, reduce_dims)
///       val = eval(expr_root at iter_idx)
///       acc = combine(acc, val)
///     out[out_i] = acc
pub fn emit_reduce_kernel(
    kernel_ctx: &KernelBuilderContext<'_>,
    emit_info: &KernelEmitInfo<'_>,
    output_shape: &[usize],
    builder: &mut FunctionBuilder,
) -> Result<()> {
    let graph = kernel_ctx.graph;

    let reduce_spec = emit_info.reduce.as_ref().unwrap();

    // Compute reduce extents: the sizes of the reduced dimensions.
    let iter_shape = emit_info.iter_shape;
    let iter_rank = iter_shape.len();

    let mut is_reduce_dim = vec![false; iter_rank];
    for &d in reduce_spec.dims {
        is_reduce_dim[d] = true;
    }

    let reduce_extents: Vec<usize> = reduce_spec.dims.iter().map(|&d| iter_shape[d]).collect();
    let reduce_numel: usize = reduce_extents.iter().product();
    let reduce_numel_val = builder.ins().iconst(types::I64, reduce_numel as i64);

    let identity = emit_reduce_identity(
        builder,
        reduce_spec.op,
        kernel_ctx.dtype,
        kernel_ctx.cl_type,
    );

    // Define constants in the entry block so they dominate all subsequent blocks.
    let zero = builder.ins().iconst(types::I64, 0);
    let one = builder.ins().iconst(types::I64, 1);
    let elem_size_val = builder.ins().iconst(types::I64, kernel_ctx.elem_size);

    // --- Outer loop: for out_i in 0..output_numel ---
    let out_header = builder.create_block();
    builder.append_block_param(out_header, types::I64);
    let out_body = builder.create_block();
    let out_exit = builder.create_block();

    builder.ins().jump(out_header, &[zero]);

    builder.switch_to_block(out_header);
    let out_i = builder.block_params(out_header)[0];
    let out_cmp = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, out_i, kernel_ctx.n_param);
    builder.ins().brif(out_cmp, out_body, &[], out_exit, &[]);

    builder.switch_to_block(out_body);
    builder.seal_block(out_body);

    // Decompose out_i into output dimension indices.
    let out_idx = decompose_flat_index(builder, out_i, output_shape);

    // --- Inner loop: for red_i in 0..reduce_numel ---
    let red_header = builder.create_block();
    builder.append_block_param(red_header, types::I64); // red_i
    builder.append_block_param(red_header, kernel_ctx.cl_type); // acc
    let red_body = builder.create_block();
    let red_exit = builder.create_block();
    builder.append_block_param(red_exit, kernel_ctx.cl_type); // acc_final

    builder.ins().jump(red_header, &[zero, identity]);

    builder.switch_to_block(red_header);
    let red_i = builder.block_params(red_header)[0];
    let acc = builder.block_params(red_header)[1];
    let red_cmp = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, red_i, reduce_numel_val);
    builder.ins().brif(red_cmp, red_body, &[], red_exit, &[acc]);

    builder.switch_to_block(red_body);
    builder.seal_block(red_body);

    // Decompose red_i into reduce dimension indices.
    let red_idx = decompose_flat_index(builder, red_i, &reduce_extents);

    // Compose full iteration index from output and reduce indices.
    let iter_idx = compose_iter_index(
        builder,
        &out_idx,
        &red_idx,
        output_shape,
        reduce_spec.dims,
        reduce_spec.keepdims,
        iter_rank,
    );

    // Compute flat byte offset for the iteration index.
    let iter_flat = flatten_multi_index(builder, &iter_idx, iter_shape);
    let iter_byte_offset = builder.ins().imul(iter_flat, elem_size_val);

    // Compute per-input tracked byte offsets from iter_idx.
    let mut tracked_byte_offsets: HashMap<NodeId, Value> = HashMap::new();
    for (&node_id, tracker) in emit_info.input_trackers {
        let tracked_offset =
            compute_tracker_byte_offset(builder, &iter_idx, tracker, kernel_ctx.elem_size);
        tracked_byte_offsets.insert(node_id, tracked_offset);
    }

    // Evaluate the expression tree at the current iteration index.
    let expr_ctx = ExprBuildContext {
        graph,
        input_ptrs: kernel_ctx.input_ptrs,
        input_index: kernel_ctx.input_index,
        math: kernel_ctx.math_refs,
        tracked_byte_offsets: &tracked_byte_offsets,
        shape_source_map: emit_info.shape_source_map,
        dtype: kernel_ctx.dtype,
        cl_type: kernel_ctx.cl_type,
    };
    let val = build_expression(&expr_ctx, emit_info.expr_root, builder, iter_byte_offset)?;

    // Combine accumulator with new value.
    let acc_next = emit_reduce_combine(builder, reduce_spec.op, kernel_ctx.dtype, acc, val);

    // red_i += 1, jump back to red_header
    let red_i_next = builder.ins().iadd(red_i, one);
    builder.ins().jump(red_header, &[red_i_next, acc_next]);

    builder.seal_block(red_header);

    // --- After inner loop: store result ---
    builder.switch_to_block(red_exit);
    builder.seal_block(red_exit);
    let acc_final = builder.block_params(red_exit)[0];

    let out_byte_offset = if let Some(tracker) = emit_info.output_tracker {
        if tracker.is_contiguous() {
            builder.ins().imul(out_i, elem_size_val)
        } else {
            compute_tracker_byte_offset(builder, &out_idx, tracker, kernel_ctx.elem_size)
        }
    } else {
        builder.ins().imul(out_i, elem_size_val)
    };
    let out_addr = builder.ins().iadd(kernel_ctx.out_ptr, out_byte_offset);
    builder.ins().store(MemFlags::new(), acc_final, out_addr, 0);

    // out_i += 1
    let out_i_next = builder.ins().iadd(out_i, one);
    builder.ins().jump(out_header, &[out_i_next]);

    builder.seal_block(out_header);

    // Exit
    builder.switch_to_block(out_exit);
    builder.seal_block(out_exit);
    builder.ins().return_(&[]);

    Ok(())
}

/// Compose a full iteration index from output indices and reduce indices.
///
/// For keepdims=true: output shape has 1 at reduced dims, so output indices
/// at those positions are 0; we replace them with the reduce indices.
///
/// For keepdims=false: output shape has reduced dims removed; we interleave
/// reduce indices at the positions of the reduced dims.
fn compose_iter_index(
    builder: &mut FunctionBuilder,
    out_idx: &[Value],
    red_idx: &[Value],
    output_shape: &[usize],
    reduce_dims: &[usize],
    keepdims: bool,
    iter_rank: usize,
) -> Vec<Value> {
    let mut result = vec![builder.ins().iconst(types::I64, 0); iter_rank];

    if keepdims {
        // Output rank == iter_rank. Reduce dims have size 1 in output.
        let mut red_pos = 0;
        let is_reduce: Vec<bool> = (0..iter_rank).map(|d| reduce_dims.contains(&d)).collect();
        for d in 0..iter_rank {
            if is_reduce[d] {
                result[d] = red_idx[red_pos];
                red_pos += 1;
            } else {
                result[d] = out_idx[d];
            }
        }
    } else {
        // Output rank < iter_rank. Reduced dims are absent from output.
        let is_reduce: Vec<bool> = (0..iter_rank).map(|d| reduce_dims.contains(&d)).collect();
        let mut out_pos = 0;
        let mut red_pos = 0;
        for d in 0..iter_rank {
            if is_reduce[d] {
                result[d] = red_idx[red_pos];
                red_pos += 1;
            } else {
                if out_pos < out_idx.len() {
                    result[d] = out_idx[out_pos];
                }
                out_pos += 1;
            }
        }

        // Handle scalar output: if all dims are reduced, output shape is [1]
        // and out_idx has one element (always 0), but we don't use it.
        let _ = output_shape;
    }

    result
}
