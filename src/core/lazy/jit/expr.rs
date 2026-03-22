use std::collections::HashMap;

use anyhow::Result;
use cranelift::prelude::{types, FunctionBuilder, InstBuilder, MemFlags, Value};

use crate::core::lazy::dtype::{DType, Scalar};
use crate::core::lazy::graph::{Graph, NodeId, Op};
use crate::core::lazy::jit::math::{math_func_ref_for_op, MathFuncRefs};

pub(super) struct ExprBuildContext<'a> {
    pub graph: &'a Graph,
    pub input_ptrs: &'a [Value],
    pub input_index: &'a HashMap<NodeId, usize>,
    pub math: &'a MathFuncRefs,
    pub tracked_byte_offsets: &'a HashMap<NodeId, Value>,
    pub shape_source_map: &'a HashMap<NodeId, NodeId>,
    pub dtype: DType,
    pub cl_type: types::Type,
}

/// Recursively build the Cranelift IR for the expression tree rooted at `id`.
pub(super) fn build_expression(
    expr_ctx: &ExprBuildContext<'_>,
    id: NodeId,
    builder: &mut FunctionBuilder,
    byte_offset: Value,
) -> Result<Value> {
    let node = expr_ctx.graph.node(id);

    match &node.op {
        op if !op.is_elementwise() && !matches!(op, Op::Const(_)) => {
            let resolved_id = expr_ctx.shape_source_map.get(&id).copied().unwrap_or(id);
            if let Some(&idx) = expr_ctx.input_index.get(&resolved_id) {
                // Source is a buffer input — load via tracked or flat offset.
                let ptr = expr_ctx.input_ptrs[idx];
                let offset = expr_ctx
                    .tracked_byte_offsets
                    .get(&resolved_id)
                    .copied()
                    .unwrap_or(byte_offset);
                let addr = builder.ins().iadd(ptr, offset);
                Ok(builder
                    .ins()
                    .load(expr_ctx.cl_type, MemFlags::new(), addr, 0))
            } else {
                // Source was inlined through a contiguous shape op chain —
                // recursively evaluate its expression tree.
                build_expression(expr_ctx, resolved_id, builder, byte_offset)
            }
        }
        Op::Load => {
            let &idx = expr_ctx
                .input_index
                .get(&id)
                .ok_or_else(|| anyhow::anyhow!("Load node {:?} not found in input_index", id))?;
            let ptr = expr_ctx.input_ptrs[idx];
            let offset = expr_ctx
                .tracked_byte_offsets
                .get(&id)
                .copied()
                .unwrap_or(byte_offset);
            let addr = builder.ins().iadd(ptr, offset);
            Ok(builder
                .ins()
                .load(expr_ctx.cl_type, MemFlags::new(), addr, 0))
        }
        Op::Const(scalar) => emit_const(builder, *scalar, expr_ctx.dtype),
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            let lhs = build_expression(expr_ctx, node.inputs[0], builder, byte_offset)?;
            let rhs = build_expression(expr_ctx, node.inputs[1], builder, byte_offset)?;
            Ok(match (node.op.clone(), expr_ctx.dtype.is_float()) {
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
            let val = build_expression(expr_ctx, node.inputs[0], builder, byte_offset)?;
            if expr_ctx.dtype.is_float() {
                Ok(builder.ins().fneg(val))
            } else {
                let zero = builder.ins().iconst(expr_ctx.cl_type, 0);
                Ok(builder.ins().isub(zero, val))
            }
        }
        Op::Exp => {
            let val = build_expression(expr_ctx, node.inputs[0], builder, byte_offset)?;
            let func_ref = math_func_ref_for_op(expr_ctx.dtype, expr_ctx.math, "exp")?;
            let call = builder.ins().call(func_ref, &[val]);
            Ok(builder.inst_results(call)[0])
        }
        Op::Ln => {
            let val = build_expression(expr_ctx, node.inputs[0], builder, byte_offset)?;
            let func_ref = math_func_ref_for_op(expr_ctx.dtype, expr_ctx.math, "ln")?;
            let call = builder.ins().call(func_ref, &[val]);
            Ok(builder.inst_results(call)[0])
        }
        Op::Sqrt => {
            let val = build_expression(expr_ctx, node.inputs[0], builder, byte_offset)?;
            if !expr_ctx.dtype.is_float() {
                return Err(anyhow::anyhow!(
                    "sqrt not supported for {:?}",
                    expr_ctx.dtype
                ));
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
        // Cross-dtype const (e.g. Scalar::F32 used in I32 kernel) - convert
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
