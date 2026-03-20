use anyhow::Result;
use cranelift::prelude::{types, FunctionBuilder, InstBuilder, MemFlags, Value};

use super::super::dtype::{DType, Scalar};
use super::super::graph::{Graph, NodeId, Op};
use super::index_map::id_to_index;
use super::math::{math_func_ref_for_op, MathFuncRefs};

/// Recursively build the Cranelift IR for the expression tree rooted at `id`.
pub(super) fn build_expression(
    graph: &Graph,
    id: NodeId,
    builder: &mut FunctionBuilder,
    input_ptrs: &[Value],
    input_index: &[Option<usize>],
    byte_offset: Value,
    math: &MathFuncRefs,
    tracked_byte_offsets: &[Option<Value>],
    shape_source_map: &[Option<NodeId>],
    dtype: DType,
    cl_type: types::Type,
) -> Result<Value> {
    let node = graph.node(id);

    match &node.op {
        op if !op.is_elementwise() && !matches!(op, Op::Const(_)) => {
            let resolved_id = shape_source_map[id_to_index(id)].unwrap_or(id);
            let idx = input_index[id_to_index(resolved_id)]
                .ok_or_else(|| anyhow::anyhow!("Barrier node {:?} not found in input_index", id))?;
            let ptr = input_ptrs[idx];
            let offset = tracked_byte_offsets[id_to_index(resolved_id)].unwrap_or(byte_offset);
            let addr = builder.ins().iadd(ptr, offset);
            Ok(builder.ins().load(cl_type, MemFlags::new(), addr, 0))
        }
        Op::Load => {
            let idx = input_index[id_to_index(id)]
                .ok_or_else(|| anyhow::anyhow!("Load node {:?} not found in input_index", id))?;
            let ptr = input_ptrs[idx];
            let offset = tracked_byte_offsets[id_to_index(id)].unwrap_or(byte_offset);
            let addr = builder.ins().iadd(ptr, offset);
            Ok(builder.ins().load(cl_type, MemFlags::new(), addr, 0))
        }
        Op::Const(scalar) => emit_const(builder, *scalar, dtype),
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            let lhs = build_expression(
                graph,
                node.inputs[0],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
                tracked_byte_offsets,
                shape_source_map,
                dtype,
                cl_type,
            )?;
            let rhs = build_expression(
                graph,
                node.inputs[1],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
                tracked_byte_offsets,
                shape_source_map,
                dtype,
                cl_type,
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
                graph,
                node.inputs[0],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
                tracked_byte_offsets,
                shape_source_map,
                dtype,
                cl_type,
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
                graph,
                node.inputs[0],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
                tracked_byte_offsets,
                shape_source_map,
                dtype,
                cl_type,
            )?;
            let func_ref = math_func_ref_for_op(dtype, math, "exp")?;
            let call = builder.ins().call(func_ref, &[val]);
            Ok(builder.inst_results(call)[0])
        }
        Op::Ln => {
            let val = build_expression(
                graph,
                node.inputs[0],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
                tracked_byte_offsets,
                shape_source_map,
                dtype,
                cl_type,
            )?;
            let func_ref = math_func_ref_for_op(dtype, math, "ln")?;
            let call = builder.ins().call(func_ref, &[val]);
            Ok(builder.inst_results(call)[0])
        }
        Op::Sqrt => {
            let val = build_expression(
                graph,
                node.inputs[0],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
                tracked_byte_offsets,
                shape_source_map,
                dtype,
                cl_type,
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
