//! Math intrinsic declarations for Cranelift JIT.

use anyhow::Result;
use cranelift::prelude::{types, AbiParam, FunctionBuilder};
use cranelift_codegen::ir::FuncRef;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use crate::core::dtype::DType;

// -------- JIT-linked math functions --------

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

/// Function IDs for declared math functions in a JIT module.
pub struct MathFuncIds {
    pub expf: FuncId,
    pub logf: FuncId,
    pub exp: FuncId,
    pub log: FuncId,
}

/// Function references for use within a Cranelift function.
pub struct MathFuncRefs {
    pub expf: FuncRef,
    pub logf: FuncRef,
    pub exp: FuncRef,
    pub log: FuncRef,
}

/// Register math function symbols with the JIT builder.
pub fn register_math_symbols(builder: &mut JITBuilder) {
    builder.symbol("jit_expf", jit_expf as *const u8);
    builder.symbol("jit_logf", jit_logf as *const u8);
    builder.symbol("jit_exp", jit_exp as *const u8);
    builder.symbol("jit_log", jit_log as *const u8);
}

/// Declare math function signatures in a JIT module.
pub fn declare_math_funcs(module: &mut JITModule) -> Result<MathFuncIds> {
    let mut math_sig_f32 = module.make_signature();
    math_sig_f32.params.push(AbiParam::new(types::F32));
    math_sig_f32.returns.push(AbiParam::new(types::F32));

    let mut math_sig_f64 = module.make_signature();
    math_sig_f64.params.push(AbiParam::new(types::F64));
    math_sig_f64.returns.push(AbiParam::new(types::F64));

    Ok(MathFuncIds {
        expf: module.declare_function("jit_expf", Linkage::Import, &math_sig_f32)?,
        logf: module.declare_function("jit_logf", Linkage::Import, &math_sig_f32)?,
        exp: module.declare_function("jit_exp", Linkage::Import, &math_sig_f64)?,
        log: module.declare_function("jit_log", Linkage::Import, &math_sig_f64)?,
    })
}

/// Declare function references within a Cranelift function builder.
pub fn declare_math_refs(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    ids: &MathFuncIds,
) -> MathFuncRefs {
    MathFuncRefs {
        expf: module.declare_func_in_func(ids.expf, builder.func),
        logf: module.declare_func_in_func(ids.logf, builder.func),
        exp: module.declare_func_in_func(ids.exp, builder.func),
        log: module.declare_func_in_func(ids.log, builder.func),
    }
}

/// Get the function reference for a math operation given the dtype.
pub fn math_func_ref_for_op(dtype: DType, math: &MathFuncRefs, op: &str) -> Result<FuncRef> {
    match (op, dtype) {
        ("exp", DType::F32) => Ok(math.expf),
        ("exp", DType::F64) => Ok(math.exp),
        ("ln", DType::F32) => Ok(math.logf),
        ("ln", DType::F64) => Ok(math.log),
        ("exp", _) => Err(anyhow::anyhow!("exp not supported for {:?}", dtype)),
        ("ln", _) => Err(anyhow::anyhow!("ln not supported for {:?}", dtype)),
        _ => Err(anyhow::anyhow!("unknown math op '{}'", op)),
    }
}
