use std::collections::HashMap;

use anyhow::Result;
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::super::dtype::DType;
use super::super::graph::{Graph, NodeId};
use super::super::schedule::FusedKernel;
use super::compiled::CompiledKernel;
use super::expr::build_expression;
use super::math::{declare_math_funcs, declare_math_refs, register_math_symbols};
use super::tracker::{compute_tracker_byte_offset, decompose_flat_index};

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
    let has_trackers = !kernel.input_trackers.is_empty();
    let dtype = graph.node(kernel.root).dtype;
    let cl_type = dtype_to_cl_type(dtype);
    let elem_size = dtype.size_bytes() as i64;

    // Map each input buffer NodeId to its parameter index.
    let mut input_index: HashMap<NodeId, usize> = HashMap::new();
    for (i, &buf_id) in kernel.input_buffers.iter().enumerate() {
        input_index.insert(buf_id, i);
    }

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
    // fn(in0: *u8, in1: *u8, ..., out: *mut u8, n: u64)
    let mut sig = module.make_signature();
    for _ in 0..num_inputs {
        sig.params.push(AbiParam::new(ptr_type));
    }
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
    let input_ptrs: Vec<Value> = block_params[..num_inputs].to_vec();
    let out_ptr = block_params[num_inputs];
    let n_param = block_params[num_inputs + 1];

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

    // If we have trackers, decompose flat index `i` into multi-dim indices.
    let dim_indices = if has_trackers {
        Some(decompose_flat_index(&mut builder, i, &kernel.output_shape))
    } else {
        None
    };

    // Pre-compute per-input byte offsets from trackers.
    let mut tracked_byte_offsets: HashMap<NodeId, Value> = HashMap::new();
    if let Some(ref dims) = dim_indices {
        for (&buf_id, tracker) in &kernel.input_trackers {
            let tracked_offset =
                compute_tracker_byte_offset(&mut builder, dims, tracker, elem_size);
            tracked_byte_offsets.insert(buf_id, tracked_offset);
        }
    }

    let result = build_expression(
        graph,
        kernel.root,
        &mut builder,
        &input_ptrs,
        &input_index,
        byte_offset,
        &math_refs,
        &tracked_byte_offsets,
        &kernel.shape_source_map,
        dtype,
        cl_type,
    )?;

    // Store result to out[i]
    let out_addr = builder.ins().iadd(out_ptr, byte_offset);
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
