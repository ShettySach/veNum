use anyhow::Result;
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;

use super::graph::{Graph, NodeId, Op};
use super::schedule::FusedKernel;

/// A compiled kernel ready to execute.
pub struct CompiledKernel {
    /// Number of input buffer pointers.
    pub num_inputs: usize,
    /// The JIT module that owns the compiled code.
    _module: JITModule,
    /// Raw function pointer to the compiled kernel.
    fn_ptr: *const u8,
}

// Safety: The compiled code is immutable once created and the function pointer
// is valid for the lifetime of _module.
unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

/// Extern "C" functions we link into the JIT for math ops.
extern "C" fn jit_expf(x: f32) -> f32 {
    x.exp()
}
extern "C" fn jit_logf(x: f32) -> f32 {
    x.ln()
}
extern "C" fn jit_sqrtf(x: f32) -> f32 {
    x.sqrt()
}

impl CompiledKernel {
    /// Execute the kernel with the given input buffer pointers and output pointer.
    ///
    /// # Safety
    /// - All pointers must be valid and point to buffers of at least `numel` f32 elements.
    /// - `inputs` must have exactly `self.num_inputs` elements.
    pub unsafe fn execute(&self, inputs: &[*const f32], output: *mut f32, numel: usize) {
        // ABI: fn(in0: *const f32, in1: *const f32, ..., out: *mut f32, n: u64)
        match self.num_inputs {
            0 => {
                let f: extern "C" fn(*mut f32, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(output, numel as u64);
            }
            1 => {
                let f: extern "C" fn(*const f32, *mut f32, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(inputs[0], output, numel as u64);
            }
            2 => {
                let f: extern "C" fn(*const f32, *const f32, *mut f32, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(inputs[0], inputs[1], output, numel as u64);
            }
            3 => {
                let f: extern "C" fn(*const f32, *const f32, *const f32, *mut f32, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(inputs[0], inputs[1], inputs[2], output, numel as u64);
            }
            _ => {
                panic!("CompiledKernel: too many inputs ({}), max 3 supported in dispatch", self.num_inputs);
            }
        }
    }
}

/// Compile a fused elementwise kernel into native code via Cranelift.
pub fn compile_kernel(graph: &Graph, kernel: &FusedKernel) -> Result<CompiledKernel> {
    let num_inputs = kernel.input_buffers.len();

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

    // Register math symbols.
    builder.symbol("jit_expf", jit_expf as *const u8);
    builder.symbol("jit_logf", jit_logf as *const u8);
    builder.symbol("jit_sqrtf", jit_sqrtf as *const u8);

    let mut module = JITModule::new(builder);
    let ptr_type = module.target_config().pointer_type();

    // --- Declare math function signatures ---
    let mut math_sig = module.make_signature();
    math_sig.params.push(AbiParam::new(types::F32));
    math_sig.returns.push(AbiParam::new(types::F32));

    let expf_id = module.declare_function("jit_expf", Linkage::Import, &math_sig)?;
    let logf_id = module.declare_function("jit_logf", Linkage::Import, &math_sig)?;
    let sqrtf_id = module.declare_function("jit_sqrtf", Linkage::Import, &math_sig)?;

    // --- Build kernel function signature ---
    // fn(in0: *f32, in1: *f32, ..., out: *mut f32, n: u64)
    let mut sig = module.make_signature();
    for _ in 0..num_inputs {
        sig.params.push(AbiParam::new(ptr_type));
    }
    sig.params.push(AbiParam::new(ptr_type)); // output pointer
    sig.params.push(AbiParam::new(types::I64)); // n

    let func_id = module.declare_function("kernel", Linkage::Local, &sig)?;

    // --- Build function body ---
    let mut func = Function::with_name_signature(
        cranelift_codegen::ir::UserFuncName::user(0, 0),
        sig.clone(),
    );

    let mut func_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

    // Declare references to math functions.
    let expf_ref = module.declare_func_in_func(expf_id, builder.func);
    let logf_ref = module.declare_func_in_func(logf_id, builder.func);
    let sqrtf_ref = module.declare_func_in_func(sqrtf_id, builder.func);

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
    // Do NOT seal loop_header yet — loop_body back-edge not created yet.

    // Loop body
    builder.switch_to_block(loop_body);
    builder.seal_block(loop_body); // Only predecessor is loop_header (already defined).

    // Byte offset = i * 4 (sizeof f32)
    let four = builder.ins().iconst(types::I64, 4);
    let byte_offset = builder.ins().imul(i, four);

    let math_refs = MathFuncRefs {
        expf: expf_ref,
        logf: logf_ref,
        sqrtf: sqrtf_ref,
    };

    let result = build_expression(
        graph,
        kernel.root,
        &mut builder,
        &input_ptrs,
        &input_index,
        byte_offset,
        &math_refs,
    )?;

    // Store result to out[i]
    let out_addr = builder.ins().iadd(out_ptr, byte_offset);
    builder.ins().store(MemFlags::new(), result, out_addr, 0);

    // i += 1, jump back to header
    let one = builder.ins().iconst(types::I64, 1);
    let i_next = builder.ins().iadd(i, one);
    builder.ins().jump(loop_header, &[i_next]);

    // Now seal loop_header — both predecessors (entry, loop_body) are complete.
    builder.seal_block(loop_header);

    // Exit
    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_exit); // Only predecessor is loop_header.
    builder.ins().return_(&[]);

    builder.finalize();

    // --- Compile ---
    let mut ctx = cranelift_codegen::Context::for_function(func);
    module.define_function(func_id, &mut ctx)?;
    module.finalize_definitions()?;

    let fn_ptr = module.get_finalized_function(func_id);

    Ok(CompiledKernel {
        num_inputs,
        _module: module,
        fn_ptr,
    })
}

struct MathFuncRefs {
    expf: cranelift_codegen::ir::FuncRef,
    logf: cranelift_codegen::ir::FuncRef,
    sqrtf: cranelift_codegen::ir::FuncRef,
}

/// Recursively build the Cranelift IR for the expression tree rooted at `id`.
fn build_expression(
    graph: &Graph,
    id: NodeId,
    builder: &mut FunctionBuilder,
    input_ptrs: &[Value],
    input_index: &HashMap<NodeId, usize>,
    byte_offset: Value,
    math: &MathFuncRefs,
) -> Result<Value> {
    let node = graph.node(id);

    match &node.op {
        Op::Load => {
            let idx = input_index.get(&id).ok_or_else(|| {
                anyhow::anyhow!("Load node {:?} not found in input_index", id)
            })?;
            let ptr = input_ptrs[*idx];
            let addr = builder.ins().iadd(ptr, byte_offset);
            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
        }
        Op::Const(val) => Ok(builder.ins().f32const(*val)),
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            let lhs = build_expression(
                graph, node.inputs[0], builder, input_ptrs, input_index, byte_offset, math,
            )?;
            let rhs = build_expression(
                graph, node.inputs[1], builder, input_ptrs, input_index, byte_offset, math,
            )?;
            Ok(match node.op {
                Op::Add => builder.ins().fadd(lhs, rhs),
                Op::Sub => builder.ins().fsub(lhs, rhs),
                Op::Mul => builder.ins().fmul(lhs, rhs),
                Op::Div => builder.ins().fdiv(lhs, rhs),
                _ => unreachable!(),
            })
        }
        Op::Neg => {
            let val = build_expression(
                graph, node.inputs[0], builder, input_ptrs, input_index, byte_offset, math,
            )?;
            Ok(builder.ins().fneg(val))
        }
        Op::Exp => {
            let val = build_expression(
                graph, node.inputs[0], builder, input_ptrs, input_index, byte_offset, math,
            )?;
            let call = builder.ins().call(math.expf, &[val]);
            Ok(builder.inst_results(call)[0])
        }
        Op::Ln => {
            let val = build_expression(
                graph, node.inputs[0], builder, input_ptrs, input_index, byte_offset, math,
            )?;
            let call = builder.ins().call(math.logf, &[val]);
            Ok(builder.inst_results(call)[0])
        }
        Op::Sqrt => {
            let val = build_expression(
                graph, node.inputs[0], builder, input_ptrs, input_index, byte_offset, math,
            )?;
            Ok(builder.ins().sqrt(val))
        }
    }
}
