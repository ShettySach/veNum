use anyhow::Result;
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::super::dtype::DType;
use super::super::graph::Graph;
use super::super::schedule::{FusedKernel, ReduceKind};
use super::compiled::CompiledKernel;
use super::expr::build_expression;
use super::math::{declare_math_funcs, declare_math_refs, register_math_symbols};
use super::tracker::{compute_tracker_byte_offset, decompose_flat_index, flatten_multi_index};

/// Map a DType to the corresponding Cranelift IR type.
fn dtype_to_cl_type(dtype: DType) -> types::Type {
    match dtype {
        DType::F32 => types::F32,
        DType::F64 => types::F64,
        DType::I32 => types::I32,
        DType::I64 => types::I64,
    }
}

/// Compile a fused elementwise kernel into native code via Cranelift.
///
/// When `capture_ir` is true, the Cranelift IR text is stored in the returned
/// `CompiledKernel` for debug/visualization purposes.
pub fn compile_kernel(
    graph: &Graph,
    kernel: &FusedKernel,
    capture_ir: bool,
) -> Result<CompiledKernel> {
    let num_inputs = kernel.input_buffers.len();
    let has_trackers = !kernel.input_trackers.iter().all(|t| t.is_none());
    let dtype = graph.node(kernel.root).dtype;
    let cl_type = dtype_to_cl_type(dtype);
    let elem_size = dtype.size_bytes() as i64;

    let input_index = &kernel.input_index_map;

    // --- Cranelift setup ---
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().map_err(|e| anyhow::anyhow!("{}", e))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    register_math_symbols(&mut builder);
    let mut module = JITModule::new(builder);
    let ptr_type = module.target_config().pointer_type();

    // --- Declare math function signatures ---
    let math_ids = declare_math_funcs(&mut module)?;

    // --- Build kernel function signature ---
    // ABI:
    //   fn(inputs: *const *const u8, out: *mut u8, n: u64)
    // The first parameter points to an array of `num_inputs` pointers.
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type)); // inputs pointer array
    sig.params.push(AbiParam::new(ptr_type)); // output pointer
    sig.params.push(AbiParam::new(types::I64)); // n

    let func_id = module.declare_function("kernel", Linkage::Local, &sig)?;

    // --- Build function body ---
    let mut func =
        Function::with_name_signature(cranelift_codegen::ir::UserFuncName::user(0, 0), sig.clone());

    let mut func_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

    // Declare references to math functions.
    let math_refs = declare_math_refs(&mut module, &mut builder, &math_ids);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Extract parameters.
    let block_params: Vec<Value> = builder.block_params(entry_block).to_vec();
    let inputs_ptr = block_params[0];
    let out_ptr = block_params[1];
    let n_param = block_params[2];

    // Load input pointers from the inputs array.
    let ptr_size = module.target_config().pointer_bytes() as i64;
    let mut input_ptrs = Vec::with_capacity(num_inputs);
    for i in 0..num_inputs {
        let off = builder
            .ins()
            .iconst(ptr_type, (i as i64).wrapping_mul(ptr_size));
        let addr = builder.ins().iadd(inputs_ptr, off);
        let p = builder.ins().load(ptr_type, MemFlags::new(), addr, 0);
        input_ptrs.push(p);
    }

    if kernel.reduce.is_some() {
        emit_reduce_kernel(
            graph,
            kernel,
            &mut builder,
            &input_ptrs,
            &input_index,
            out_ptr,
            n_param,
            &math_refs,
            has_trackers,
            dtype,
            cl_type,
            elem_size,
        )?;
    } else {
        emit_elementwise_kernel(
            graph,
            kernel,
            &mut builder,
            &input_ptrs,
            &input_index,
            out_ptr,
            n_param,
            &math_refs,
            has_trackers,
            dtype,
            cl_type,
            elem_size,
        )?;
    }

    builder.finalize();

    let clif_ir = if capture_ir {
        Some(func.display().to_string())
    } else {
        None
    };

    // --- Compile ---
    let mut ctx = cranelift_codegen::Context::for_function(func);
    module.define_function(func_id, &mut ctx)?;

    module.finalize_definitions()?;

    let fn_ptr = module.get_finalized_function(func_id);

    Ok(CompiledKernel {
        num_inputs,
        _module: module,
        fn_ptr,
        clif_ir,
    })
}

/// Emit the body of a pure elementwise kernel (existing logic).
#[allow(clippy::too_many_arguments)]
fn emit_elementwise_kernel(
    graph: &Graph,
    kernel: &FusedKernel,
    builder: &mut FunctionBuilder,
    input_ptrs: &[Value],
    input_index: &[Option<usize>],
    out_ptr: Value,
    n_param: Value,
    math_refs: &super::math::MathFuncRefs,
    has_trackers: bool,
    dtype: DType,
    cl_type: types::Type,
    elem_size: i64,
) -> Result<()> {
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
    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, i, n_param);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    // Loop body
    builder.switch_to_block(loop_body);
    builder.seal_block(loop_body);

    // Byte offset for output = i * elem_size.
    let elem_size_val = builder.ins().iconst(types::I64, elem_size);
    let byte_offset = builder.ins().imul(i, elem_size_val);

    // Check if any tracker is non-contiguous and actually needs index
    // decomposition. Contiguous trackers can just use the flat byte offset.
    let output_tracker_noncontig = kernel
        .output_tracker
        .as_ref()
        .map(|t| !t.is_contiguous())
        .unwrap_or(false);
    let needs_decompose =
        (has_trackers && kernel.has_noncontiguous_trackers) || output_tracker_noncontig;

    // If we have non-contiguous trackers, decompose flat index `i` into
    // multi-dim indices. Contiguous trackers reuse the flat byte offset.
    let dim_indices = if needs_decompose {
        Some(decompose_flat_index(builder, i, &kernel.output_shape))
    } else {
        None
    };

    // Pre-compute per-input byte offsets from trackers.
    let mut tracked_byte_offsets: Vec<Option<Value>> = vec![None; graph.nodes.len()];
    for (buf_idx, tracker_opt) in kernel.input_trackers.iter().enumerate() {
        let Some(tracker) = tracker_opt else {
            continue;
        };
        if tracker.is_contiguous() {
            // Contiguous tracker: flat offset is equivalent, no decomposition needed.
            tracked_byte_offsets[buf_idx] = Some(byte_offset);
        } else if let Some(ref dims) = dim_indices {
            let tracked_offset = compute_tracker_byte_offset(builder, dims, tracker, elem_size);
            tracked_byte_offsets[buf_idx] = Some(tracked_offset);
        }
    }

    let result = build_expression(
        graph,
        kernel.expr_root,
        builder,
        input_ptrs,
        input_index,
        byte_offset,
        math_refs,
        &tracked_byte_offsets,
        &kernel.shape_source_map,
        dtype,
        cl_type,
    )?;

    // Store result to out[...], applying any forward-fused output tracker.
    let out_byte_offset = if let Some(tracker) = kernel.output_tracker.as_ref() {
        if tracker.is_contiguous() {
            byte_offset
        } else if let Some(ref dims) = dim_indices {
            compute_tracker_byte_offset(builder, dims, tracker, elem_size)
        } else {
            let dims = decompose_flat_index(builder, i, &kernel.output_shape);
            compute_tracker_byte_offset(builder, &dims, tracker, elem_size)
        }
    } else {
        byte_offset
    };

    let out_addr = builder.ins().iadd(out_ptr, out_byte_offset);
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
#[allow(clippy::too_many_arguments)]
fn emit_reduce_kernel(
    graph: &Graph,
    kernel: &FusedKernel,
    builder: &mut FunctionBuilder,
    input_ptrs: &[Value],
    input_index: &[Option<usize>],
    out_ptr: Value,
    n_param: Value,
    math_refs: &super::math::MathFuncRefs,
    _has_trackers: bool,
    dtype: DType,
    cl_type: types::Type,
    elem_size: i64,
) -> Result<()> {
    let reduce_spec = kernel.reduce.as_ref().unwrap();

    // Compute reduce extents: the sizes of the reduced dimensions.
    let iter_shape = &kernel.iter_shape;
    let iter_rank = iter_shape.len();

    let mut is_reduce_dim = vec![false; iter_rank];
    for &d in &reduce_spec.dims {
        is_reduce_dim[d] = true;
    }

    let reduce_extents: Vec<usize> = reduce_spec.dims.iter().map(|&d| iter_shape[d]).collect();
    let reduce_numel: usize = reduce_extents.iter().product();
    let reduce_numel_val = builder.ins().iconst(types::I64, reduce_numel as i64);

    let identity = emit_reduce_identity(builder, &reduce_spec.op, dtype, cl_type);

    // Define constants in the entry block so they dominate all subsequent blocks.
    let zero = builder.ins().iconst(types::I64, 0);
    let one = builder.ins().iconst(types::I64, 1);
    let elem_size_val = builder.ins().iconst(types::I64, elem_size);

    // --- Outer loop: for out_i in 0..output_numel ---
    let out_header = builder.create_block();
    builder.append_block_param(out_header, types::I64);
    let out_body = builder.create_block();
    let out_exit = builder.create_block();

    builder.ins().jump(out_header, &[zero]);

    builder.switch_to_block(out_header);
    let out_i = builder.block_params(out_header)[0];
    let out_cmp = builder.ins().icmp(IntCC::UnsignedLessThan, out_i, n_param);
    builder.ins().brif(out_cmp, out_body, &[], out_exit, &[]);

    builder.switch_to_block(out_body);
    builder.seal_block(out_body);

    // Decompose out_i into output dimension indices.
    let out_idx = decompose_flat_index(builder, out_i, &kernel.output_shape);

    // --- Inner loop: for red_i in 0..reduce_numel ---
    let red_header = builder.create_block();
    builder.append_block_param(red_header, types::I64); // red_i
    builder.append_block_param(red_header, cl_type); // acc
    let red_body = builder.create_block();
    let red_exit = builder.create_block();
    builder.append_block_param(red_exit, cl_type); // acc_final

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
        &kernel.output_shape,
        &reduce_spec.dims,
        reduce_spec.keepdims,
        iter_rank,
    );

    // Compute flat byte offset for the iteration index.
    let iter_flat = flatten_multi_index(builder, &iter_idx, iter_shape);
    let iter_byte_offset = builder.ins().imul(iter_flat, elem_size_val);

    // Compute per-input tracked byte offsets from iter_idx.
    let mut tracked_byte_offsets: Vec<Option<Value>> = vec![None; graph.nodes.len()];
    for (buf_idx, tracker_opt) in kernel.input_trackers.iter().enumerate() {
        let Some(tracker) = tracker_opt else {
            continue;
        };
        let tracked_offset = compute_tracker_byte_offset(builder, &iter_idx, tracker, elem_size);
        tracked_byte_offsets[buf_idx] = Some(tracked_offset);
    }

    // Evaluate the expression tree at the current iteration index.
    let val = build_expression(
        graph,
        kernel.expr_root,
        builder,
        input_ptrs,
        input_index,
        iter_byte_offset,
        math_refs,
        &tracked_byte_offsets,
        &kernel.shape_source_map,
        dtype,
        cl_type,
    )?;

    // Combine accumulator with new value.
    let acc_next = emit_reduce_combine(builder, &reduce_spec.op, dtype, acc, val);

    // red_i += 1, jump back to red_header
    let red_i_next = builder.ins().iadd(red_i, one);
    builder.ins().jump(red_header, &[red_i_next, acc_next]);

    builder.seal_block(red_header);

    // --- After inner loop: store result ---
    builder.switch_to_block(red_exit);
    builder.seal_block(red_exit);
    let acc_final = builder.block_params(red_exit)[0];

    let out_byte_offset = if let Some(tracker) = kernel.output_tracker.as_ref() {
        if tracker.is_contiguous() {
            builder.ins().imul(out_i, elem_size_val)
        } else {
            compute_tracker_byte_offset(builder, &out_idx, tracker, elem_size)
        }
    } else {
        builder.ins().imul(out_i, elem_size_val)
    };
    let out_addr = builder.ins().iadd(out_ptr, out_byte_offset);
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

/// Emit the identity value for a reduction operation.
fn emit_reduce_identity(
    builder: &mut FunctionBuilder,
    op: &ReduceKind,
    dtype: DType,
    cl_type: types::Type,
) -> Value {
    match (op, dtype.is_float()) {
        (ReduceKind::Sum, true) => {
            if dtype == DType::F32 {
                builder.ins().f32const(0.0)
            } else {
                builder.ins().f64const(0.0)
            }
        }
        (ReduceKind::Sum, false) => builder.ins().iconst(cl_type, 0),
        (ReduceKind::Prod, true) => {
            if dtype == DType::F32 {
                builder.ins().f32const(1.0)
            } else {
                builder.ins().f64const(1.0)
            }
        }
        (ReduceKind::Prod, false) => builder.ins().iconst(cl_type, 1),
        (ReduceKind::Max, true) => {
            if dtype == DType::F32 {
                builder.ins().f32const(f32::NEG_INFINITY)
            } else {
                builder.ins().f64const(f64::NEG_INFINITY)
            }
        }
        (ReduceKind::Max, false) => {
            if dtype == DType::I32 {
                builder.ins().iconst(cl_type, i32::MIN as i64)
            } else {
                builder.ins().iconst(cl_type, i64::MIN)
            }
        }
        (ReduceKind::Min, true) => {
            if dtype == DType::F32 {
                builder.ins().f32const(f32::INFINITY)
            } else {
                builder.ins().f64const(f64::INFINITY)
            }
        }
        (ReduceKind::Min, false) => {
            if dtype == DType::I32 {
                builder.ins().iconst(cl_type, i32::MAX as i64)
            } else {
                builder.ins().iconst(cl_type, i64::MAX)
            }
        }
    }
}

/// Emit the combine operation for a reduction.
fn emit_reduce_combine(
    builder: &mut FunctionBuilder,
    op: &ReduceKind,
    dtype: DType,
    acc: Value,
    val: Value,
) -> Value {
    let is_float = dtype.is_float();
    match (op, is_float) {
        (ReduceKind::Sum, true) => builder.ins().fadd(acc, val),
        (ReduceKind::Sum, false) => builder.ins().iadd(acc, val),
        (ReduceKind::Prod, true) => builder.ins().fmul(acc, val),
        (ReduceKind::Prod, false) => builder.ins().imul(acc, val),
        (ReduceKind::Max, true) => builder.ins().fmax(acc, val),
        (ReduceKind::Max, false) => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, acc, val);
            builder.ins().select(cmp, acc, val)
        }
        (ReduceKind::Min, true) => builder.ins().fmin(acc, val),
        (ReduceKind::Min, false) => {
            let cmp = builder.ins().icmp(IntCC::SignedLessThan, acc, val);
            builder.ins().select(cmp, acc, val)
        }
    }
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
