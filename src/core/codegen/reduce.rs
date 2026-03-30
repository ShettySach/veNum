//! Reduce operation helpers for Cranelift IR.

use cranelift::prelude::{types, FunctionBuilder, InstBuilder, IntCC, Value};

use crate::core::dtype::DType;

/// Kinds of reduction operations.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Sum,
    Prod,
    Max,
    Min,
}

/// Emit the identity value for a reduction operation.
pub fn emit_reduce_identity(
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
pub fn emit_reduce_combine(
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
