//! CPU code generator implementation.

use anyhow::Result;
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_jit::JITModule;
use cranelift_module::Module;

use crate::core::liquid::schedule::ReduceKind as LazyReduceKind;
use crate::core::shared::codegen::cranelift_setup::dtype_to_cl_type;
use crate::core::shared::codegen::emit::{
    emit_elementwise_kernel, emit_reduce_kernel, KernelBuilderContext, KernelEmitInfo, ReduceSpec,
};
use crate::core::shared::codegen::generator::{CodeGenerator, FusedKernel, GeneratedKernel};
use crate::core::shared::codegen::math::{declare_math_funcs, declare_math_refs};
use crate::core::shared::codegen::reduce::ReduceKind;
use crate::core::shared::graph::Graph;

/// CPU code generator using Cranelift.
///
/// This implements the shared `CodeGenerator` trait for CPU targets.
/// It generates Cranelift IR that can be:
/// - Immediately JIT compiled (Liquid mode)
/// - Collected and linked into a program (Solid mode)
pub struct CpuCodeGenerator;

impl CodeGenerator for CpuCodeGenerator {
    fn generate_kernel(
        &self,
        module: &mut JITModule,
        graph: &Graph,
        kernel: &FusedKernel,
        capture_ir: bool,
    ) -> Result<GeneratedKernel> {
        let num_inputs = kernel.input_buffers.len();
        let has_trackers = !kernel.input_trackers.is_empty();
        let dtype = graph.node(kernel.root).dtype;
        let cl_type = dtype_to_cl_type(dtype);
        let elem_size = dtype.size_bytes() as i64;

        let input_index = &kernel.input_index_map;

        let ptr_type = module.target_config().pointer_type();

        // --- Declare math function signatures ---
        let math_ids = declare_math_funcs(module)?;

        // --- Build kernel function signature ---
        // ABI:
        //   fn(inputs: *const *const u8, out: *mut u8, n: u64)
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_type)); // inputs pointer array
        sig.params.push(AbiParam::new(ptr_type)); // output pointer
        sig.params.push(AbiParam::new(types::I64)); // n

        // --- Build function body ---
        let mut func = Function::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, 0),
            sig.clone(),
        );

        let mut func_ctx = FunctionBuilderContext::new();
        let mut func_builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        // Declare references to math functions.
        let math_refs = declare_math_refs(module, &mut func_builder, &math_ids);

        let entry_block = func_builder.create_block();
        func_builder.append_block_params_for_function_params(entry_block);
        func_builder.switch_to_block(entry_block);
        func_builder.seal_block(entry_block);

        // Extract parameters.
        let block_params: Vec<Value> = func_builder.block_params(entry_block).to_vec();
        let inputs_ptr = block_params[0];
        let out_ptr = block_params[1];
        let n_param = block_params[2];

        // Load input pointers from the inputs array.
        let ptr_size = module.target_config().pointer_bytes() as i64;
        let mut input_ptrs = Vec::with_capacity(num_inputs);
        for i in 0..num_inputs {
            let off = func_builder
                .ins()
                .iconst(ptr_type, (i as i64).wrapping_mul(ptr_size));
            let addr = func_builder.ins().iadd(inputs_ptr, off);
            let p = func_builder.ins().load(ptr_type, MemFlags::new(), addr, 0);
            input_ptrs.push(p);
        }

        let kernel_ctx = KernelBuilderContext {
            graph,
            input_ptrs: &input_ptrs,
            input_index,
            out_ptr,
            n_param,
            math_refs: &math_refs,
            has_trackers,
            dtype,
            cl_type,
            elem_size,
        };

        let reduce_spec = kernel.reduce.as_ref().map(|r| {
            let kind = match r.op {
                LazyReduceKind::Sum => ReduceKind::Sum,
                LazyReduceKind::Prod => ReduceKind::Prod,
                LazyReduceKind::Max => ReduceKind::Max,
                LazyReduceKind::Min => ReduceKind::Min,
            };
            (kind, r.dims.clone(), r.keepdims)
        });

        let emit_info = KernelEmitInfo {
            expr_root: kernel.expr_root,
            output_shape: &kernel.output_shape,
            iter_shape: &kernel.iter_shape,
            input_trackers: &kernel.input_trackers,
            shape_source_map: &kernel.shape_source_map,
            has_noncontiguous_trackers: kernel.has_noncontiguous_trackers,
            output_tracker: kernel.output_tracker.as_ref(),
            reduce: reduce_spec
                .as_ref()
                .map(|(kind, dims, keepdims)| ReduceSpec {
                    op: kind,
                    dims: dims.as_slice(),
                    keepdims: *keepdims,
                }),
        };

        if kernel.reduce.is_some() {
            emit_reduce_kernel(
                &kernel_ctx,
                &emit_info,
                &kernel.output_shape,
                &mut func_builder,
            )?;
        } else {
            emit_elementwise_kernel(&kernel_ctx, &emit_info, &mut func_builder)?;
        }

        func_builder.finalize();

        let debug_ir = if capture_ir {
            Some(func.display().to_string())
        } else {
            None
        };

        Ok(GeneratedKernel {
            function: func,
            num_inputs,
            debug_ir,
        })
    }
}
