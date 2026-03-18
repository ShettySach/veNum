use anyhow::Result;
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::dtype::{DType, Scalar};
use super::graph::{Graph, NodeId, Op};
use super::schedule::FusedKernel;
use super::shape_tracker::ShapeTracker;

// -------- kernel signature (structural identity for caching) --------

/// Structural identity of a fused kernel, used as a cache key.
///
/// Two kernels with the same op tree, shapes, and tracker layouts will produce
/// identical machine code, so they can share a single `CompiledKernel`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelSignature(u64);

impl KernelSignature {
    /// Compute the structural signature of a fused kernel by hashing its
    /// expression tree (ops + shapes + trackers), ignoring concrete data.
    pub fn from_kernel(graph: &Graph, kernel: &FusedKernel) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash output shape, numel, and dtype.
        kernel.output_shape.hash(&mut hasher);
        kernel.numel.hash(&mut hasher);
        graph.node(kernel.root).dtype.hash(&mut hasher);

        // Build input_index the same way compile_kernel does.
        let mut input_index: HashMap<NodeId, usize> = HashMap::new();
        for (i, &buf_id) in kernel.input_buffers.iter().enumerate() {
            input_index.insert(buf_id, i);
        }

        // Hash the expression tree structure.
        hash_expr(
            graph,
            kernel.root,
            &input_index,
            &kernel.input_trackers,
            &kernel.shape_source_map,
            &mut hasher,
        );

        KernelSignature(hasher.finish())
    }
}

/// Recursively hash the expression tree rooted at `id`.
fn hash_expr(
    graph: &Graph,
    id: NodeId,
    input_index: &HashMap<NodeId, usize>,
    trackers: &HashMap<NodeId, ShapeTracker>,
    source_map: &HashMap<NodeId, NodeId>,
    hasher: &mut impl Hasher,
) {
    let node = graph.node(id);

    // Hash a discriminant tag for the op.
    std::mem::discriminant(&node.op).hash(hasher);

    match &node.op {
        Op::Const(v) => v.hash(hasher),
        Op::Load => {
            // Leaf — hash its input index and any tracker.
            let resolved = source_map.get(&id).copied().unwrap_or(id);
            if let Some(&idx) = input_index.get(&resolved) {
                0u8.hash(hasher); // tag: indexed input
                idx.hash(hasher);
                if let Some(tracker) = trackers.get(&resolved) {
                    hash_tracker(tracker, hasher);
                }
            }
        }
        op if !op.is_elementwise() => {
            // Inlined shape op resolved to a source buffer.
            let resolved = source_map.get(&id).copied().unwrap_or(id);
            if let Some(&idx) = input_index.get(&resolved) {
                1u8.hash(hasher); // tag: resolved shape op
                idx.hash(hasher);
                if let Some(tracker) = trackers.get(&resolved) {
                    hash_tracker(tracker, hasher);
                }
            }
        }
        _ => {
            // Elementwise ops — recurse into children.
            for &input_id in &node.inputs {
                hash_expr(graph, input_id, input_index, trackers, source_map, hasher);
            }
        }
    }
}

fn hash_tracker(tracker: &ShapeTracker, hasher: &mut impl Hasher) {
    tracker.shape.hash(hasher);
    tracker.strides.hash(hasher);
    tracker.offset.hash(hasher);
}

// -------- compiled kernel --------

/// A compiled kernel ready to execute.
pub struct CompiledKernel {
    /// Number of input buffer pointers.
    pub num_inputs: usize,
    /// The JIT module that owns the compiled code.
    _module: JITModule,
    /// Raw function pointer to the compiled kernel.
    fn_ptr: *const u8,
    /// Cranelift IR (CLIF) text, captured only when requested.
    pub clif_ir: Option<String>,
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
extern "C" fn jit_exp(x: f64) -> f64 {
    x.exp()
}
extern "C" fn jit_log(x: f64) -> f64 {
    x.ln()
}

impl CompiledKernel {
    /// Execute the kernel with the given input buffer pointers and output pointer.
    ///
    /// # Safety
    /// - All pointers must be valid and point to buffers of sufficient size.
    /// - `inputs` must have exactly `self.num_inputs` elements.
    pub unsafe fn execute(&self, inputs: &[*const u8], output: *mut u8, numel: usize) {
        // ABI: fn(in0: *const u8, in1: *const u8, ..., out: *mut u8, n: u64)
        // All pointer types have the same ABI representation.
        match self.num_inputs {
            0 => {
                let f: extern "C" fn(*mut u8, u64) = std::mem::transmute(self.fn_ptr);
                f(output, numel as u64);
            }
            1 => {
                let f: extern "C" fn(*const u8, *mut u8, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(inputs[0], output, numel as u64);
            }
            2 => {
                let f: extern "C" fn(*const u8, *const u8, *mut u8, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(inputs[0], inputs[1], output, numel as u64);
            }
            3 => {
                let f: extern "C" fn(*const u8, *const u8, *const u8, *mut u8, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(inputs[0], inputs[1], inputs[2], output, numel as u64);
            }
            4 => {
                let f: extern "C" fn(
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *mut u8,
                    u64,
                ) = std::mem::transmute(self.fn_ptr);
                f(
                    inputs[0], inputs[1], inputs[2], inputs[3], output,
                    numel as u64,
                );
            }
            5 => {
                let f: extern "C" fn(
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *mut u8,
                    u64,
                ) = std::mem::transmute(self.fn_ptr);
                f(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                    output, numel as u64,
                );
            }
            6 => {
                let f: extern "C" fn(
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *mut u8,
                    u64,
                ) = std::mem::transmute(self.fn_ptr);
                f(
                    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                    inputs[5], output, numel as u64,
                );
            }
            _ => {
                panic!(
                    "CompiledKernel: too many inputs ({}), max 6 supported in dispatch",
                    self.num_inputs
                );
            }
        }
    }
}

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

    // Register math symbols.
    builder.symbol("jit_expf", jit_expf as *const u8);
    builder.symbol("jit_logf", jit_logf as *const u8);
    builder.symbol("jit_exp", jit_exp as *const u8);
    builder.symbol("jit_log", jit_log as *const u8);
    let mut module = JITModule::new(builder);
    let ptr_type = module.target_config().pointer_type();

    // --- Declare math function signatures ---
    let mut math_sig_f32 = module.make_signature();
    math_sig_f32.params.push(AbiParam::new(types::F32));
    math_sig_f32.returns.push(AbiParam::new(types::F32));

    let mut math_sig_f64 = module.make_signature();
    math_sig_f64.params.push(AbiParam::new(types::F64));
    math_sig_f64.returns.push(AbiParam::new(types::F64));

    let expf_id = module.declare_function("jit_expf", Linkage::Import, &math_sig_f32)?;
    let logf_id = module.declare_function("jit_logf", Linkage::Import, &math_sig_f32)?;
    let exp_id = module.declare_function("jit_exp", Linkage::Import, &math_sig_f64)?;
    let log_id = module.declare_function("jit_log", Linkage::Import, &math_sig_f64)?;

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
    let expf_ref = module.declare_func_in_func(expf_id, builder.func);
    let logf_ref = module.declare_func_in_func(logf_id, builder.func);
    let exp_ref = module.declare_func_in_func(exp_id, builder.func);
    let log_ref = module.declare_func_in_func(log_id, builder.func);

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

    let math_refs = MathFuncRefs {
        expf: expf_ref,
        logf: logf_ref,
        exp: exp_ref,
        log: log_ref,
    };

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

    // Now seal loop_header — both predecessors (entry, loop_body) are complete.
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

/// Decompose a flat index `i` into per-dimension indices for `output_shape`.
///
/// For shape [s0, s1, s2]:
///   d2 = i % s2
///   d1 = (i / s2) % s1
///   d0 = (i / (s2 * s1)) % s0
fn decompose_flat_index(
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
fn compute_tracker_byte_offset(
    builder: &mut FunctionBuilder,
    dim_indices: &[Value],
    tracker: &ShapeTracker,
    elem_size: i64,
) -> Value {
    let mut sum = builder.ins().iconst(types::I64, tracker.offset as i64);

    for (d, &stride) in tracker.strides.iter().enumerate() {
        if stride == 0 {
            // Broadcast dimension, contributes nothing.
            continue;
        }
        if d >= dim_indices.len() {
            break;
        }
        let stride_val = builder.ins().iconst(types::I64, stride as i64);
        let contribution = builder.ins().imul(dim_indices[d], stride_val);
        sum = builder.ins().iadd(sum, contribution);
    }

    let size_val = builder.ins().iconst(types::I64, elem_size);
    builder.ins().imul(sum, size_val)
}

struct MathFuncRefs {
    expf: cranelift_codegen::ir::FuncRef,
    logf: cranelift_codegen::ir::FuncRef,
    exp: cranelift_codegen::ir::FuncRef,
    log: cranelift_codegen::ir::FuncRef,
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
    tracked_byte_offsets: &HashMap<NodeId, Value>,
    shape_source_map: &HashMap<NodeId, NodeId>,
    dtype: DType,
    cl_type: types::Type,
) -> Result<Value> {
    let node = graph.node(id);

    match &node.op {
        op if !op.is_elementwise() && !matches!(op, Op::Const(_)) => {
            let resolved_id = shape_source_map.get(&id).copied().unwrap_or(id);
            let idx = input_index
                .get(&resolved_id)
                .ok_or_else(|| anyhow::anyhow!("Barrier node {:?} not found in input_index", id))?;
            let ptr = input_ptrs[*idx];
            let offset = tracked_byte_offsets
                .get(&resolved_id)
                .copied()
                .unwrap_or(byte_offset);
            let addr = builder.ins().iadd(ptr, offset);
            Ok(builder.ins().load(cl_type, MemFlags::new(), addr, 0))
        }
        Op::Load => {
            let idx = input_index
                .get(&id)
                .ok_or_else(|| anyhow::anyhow!("Load node {:?} not found in input_index", id))?;
            let ptr = input_ptrs[*idx];
            let offset = tracked_byte_offsets
                .get(&id)
                .copied()
                .unwrap_or(byte_offset);
            let addr = builder.ins().iadd(ptr, offset);
            Ok(builder.ins().load(cl_type, MemFlags::new(), addr, 0))
        }
        Op::Const(scalar) => emit_const(builder, *scalar, dtype),
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            let lhs = build_expression(
                graph, node.inputs[0], builder, input_ptrs, input_index,
                byte_offset, math, tracked_byte_offsets, shape_source_map,
                dtype, cl_type,
            )?;
            let rhs = build_expression(
                graph, node.inputs[1], builder, input_ptrs, input_index,
                byte_offset, math, tracked_byte_offsets, shape_source_map,
                dtype, cl_type,
            )?;
            Ok(match (node.op.clone(), dtype.is_float()) {
                (Op::Add, true) => builder.ins().fadd(lhs, rhs),
                (Op::Sub, true) => builder.ins().fsub(lhs, rhs),
                (Op::Mul, true) => builder.ins().fmul(lhs, rhs),
                (Op::Div, true) => builder.ins().fdiv(lhs, rhs),
                (Op::Add, false) => builder.ins().iadd(lhs, rhs),
                (Op::Sub, false) => builder.ins().isub(lhs, rhs),
                (Op::Mul, false) => builder.ins().imul(lhs, rhs),
                (Op::Div, false) => builder.ins().sdiv(lhs, rhs),
                _ => unreachable!(),
            })
        }
        Op::Neg => {
            let val = build_expression(
                graph, node.inputs[0], builder, input_ptrs, input_index,
                byte_offset, math, tracked_byte_offsets, shape_source_map,
                dtype, cl_type,
            )?;
            if dtype.is_float() {
                Ok(builder.ins().fneg(val))
            } else {
                let zero = builder.ins().iconst(cl_type, 0);
                Ok(builder.ins().isub(zero, val))
            }
        }
        Op::Exp => {
            let val = build_expression(
                graph, node.inputs[0], builder, input_ptrs, input_index,
                byte_offset, math, tracked_byte_offsets, shape_source_map,
                dtype, cl_type,
            )?;
            let func_ref = match dtype {
                DType::F32 => math.expf,
                DType::F64 => math.exp,
                _ => return Err(anyhow::anyhow!("exp not supported for {:?}", dtype)),
            };
            let call = builder.ins().call(func_ref, &[val]);
            Ok(builder.inst_results(call)[0])
        }
        Op::Ln => {
            let val = build_expression(
                graph, node.inputs[0], builder, input_ptrs, input_index,
                byte_offset, math, tracked_byte_offsets, shape_source_map,
                dtype, cl_type,
            )?;
            let func_ref = match dtype {
                DType::F32 => math.logf,
                DType::F64 => math.log,
                _ => return Err(anyhow::anyhow!("ln not supported for {:?}", dtype)),
            };
            let call = builder.ins().call(func_ref, &[val]);
            Ok(builder.inst_results(call)[0])
        }
        Op::Sqrt => {
            let val = build_expression(
                graph, node.inputs[0], builder, input_ptrs, input_index,
                byte_offset, math, tracked_byte_offsets, shape_source_map,
                dtype, cl_type,
            )?;
            if !dtype.is_float() {
                return Err(anyhow::anyhow!("sqrt not supported for {:?}", dtype));
            }
            Ok(builder.ins().sqrt(val))
        }
        _ => Err(anyhow::anyhow!(
            "Unsupported op in JIT expression builder for node {:?}: {:?}",
            id,
            node.op
        )),
    }
}

/// Emit a constant value instruction for the given Scalar and DType.
fn emit_const(builder: &mut FunctionBuilder, scalar: Scalar, dtype: DType) -> Result<Value> {
    Ok(match (scalar, dtype) {
        (Scalar::F32(v), DType::F32) => builder.ins().f32const(v),
        (Scalar::F64(v), DType::F64) => builder.ins().f64const(v),
        (Scalar::I32(v), DType::I32) => builder.ins().iconst(types::I32, v as i64),
        (Scalar::I64(v), DType::I64) => builder.ins().iconst(types::I64, v),
        // Cross-dtype const (e.g. Scalar::F32 used in I32 kernel) — convert
        _ => {
            let v = scalar.to_f64();
            match dtype {
                DType::F32 => builder.ins().f32const(v as f32),
                DType::F64 => builder.ins().f64const(v),
                DType::I32 => builder.ins().iconst(types::I32, v as i64),
                DType::I64 => builder.ins().iconst(types::I64, v as i64),
            }
        }
    })
}
