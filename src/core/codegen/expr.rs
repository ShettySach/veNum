//! Expression tree compilation to Cranelift IR.
//!
//! # Scheduler-Codegen Contract
//!
//! This module implements the codegen side of the contract established by the scheduler
//! (`collect_kernel_inputs` in `shared/schedule/fused_kernel.rs`).
//!
//! ## Contract Overview
//!
//! The scheduler decides which nodes to **inline** (compute on-the-fly) vs **materialize**
//! (load from buffer). The codegen must respect these decisions when building expression trees.
//!
//! ### Input Data Structures
//!
//! From `FusedKernel` (provided by scheduler):
//! - `input_index: HashMap<NodeId, usize>` - Maps materialized buffer NodeIds to input array position
//! - `shape_source_map: HashMap<NodeId, NodeId>` - Maps absorbed shape-op NodeIds to source buffer
//! - `tracked_byte_offsets: HashMap<NodeId, Value>` - Pre-computed offsets for non-contiguous access
//!
//! ### Codegen Behavior
//!
//! `build_expression(id)` handles nodes based on scheduler decisions:
//!
//! 1. **Materialized nodes** (in `input_index`):
//!    - Generate buffer load instruction: `load(input_ptrs[idx] + offset)`
//!    - Early return - don't recurse into their inputs
//!
//! 2. **Inlined constants** (`Op::Const`):
//!    - Generate const instruction: `f32const(v)` / `iconst(v)`
//!    - Scheduler always marks these as inlined
//!
//! 3. **Inlined elementwise** (`Op::Add`, `Op::Mul`, etc.):
//!    - Recursively build expression trees for inputs
//!    - Generate operation instruction: `fadd(lhs, rhs)`
//!    - CSE memoization prevents duplicate computation
//!
//! 4. **Absorbed shape-ops** (in `shape_source_map`):
//!    - Resolve to source buffer and recurse
//!    - Source may be inlined or materialized based on policy
//!
//! ### CSE Memoization
//!
//! `build_expression_impl` includes Common Subexpression Elimination:
//! - Maintains `memo: HashMap<NodeId, Value>` cache
//! - Before computing, checks if node already evaluated
//! - After computing, caches result for future lookups
//! - Critical for the default fusion policy which inlines multi-consumer nodes
//!
//! ### Error Cases (Contract Violations)
//!
//! - `Op::Load` not in `input_index` → Scheduler failed to materialize a load
//! - Unsupported op type → Graph contains op not handled by codegen
//! - These are internal bugs, not user errors
//!
//! ## Policy-Specific Behavior
//!
//! ### Aggressive Fusion Policy
//! - Multi-consumer nodes are inlined → Expression trees duplicated across consumers
//! - CSE memoization is critical to prevent redundant IR building for the same node
//! - Backend compiler (Cranelift) may further optimize duplicate computation

use std::collections::HashMap;

use anyhow::Result;
use cranelift::prelude::{types, FunctionBuilder, InstBuilder, MemFlags, Value};

use crate::core::codegen::math::{math_func_ref_for_op, MathFuncRefs};
use crate::core::dtype::{DType, Scalar};
use crate::core::graph::{Graph, NodeId, Op};

/// Context for building expression trees in Cranelift IR.
pub struct ExprBuildContext<'a> {
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
///
/// This function includes Common Subexpression Elimination (CSE) through memoization.
/// When a node is encountered that has already been computed (present in `memo`),
/// the cached Cranelift value is returned instead of recomputing. This is especially
/// important when the fusion policy inlines multi-consumer nodes, preventing redundant
/// computation.
pub fn build_expression(
    expr_ctx: &ExprBuildContext<'_>,
    id: NodeId,
    builder: &mut FunctionBuilder,
    byte_offset: Value,
) -> Result<Value> {
    build_expression_impl(expr_ctx, id, builder, byte_offset, &mut HashMap::new())
}

/// Implementation of build_expression with CSE memoization.
fn build_expression_impl(
    expr_ctx: &ExprBuildContext<'_>,
    id: NodeId,
    builder: &mut FunctionBuilder,
    byte_offset: Value,
    memo: &mut HashMap<NodeId, Value>,
) -> Result<Value> {
    // Check CSE cache first
    if let Some(&cached_value) = memo.get(&id) {
        return Ok(cached_value);
    }

    // Any node that appears in `input_index` is a materialized kernel input,
    // regardless of its original op. Treat it as a leaf load.
    let resolved_id = expr_ctx.shape_source_map.get(&id).copied().unwrap_or(id);
    if let Some(&idx) = expr_ctx.input_index.get(&resolved_id) {
        let ptr = expr_ctx.input_ptrs[idx];
        let offset = expr_ctx
            .tracked_byte_offsets
            .get(&resolved_id)
            .copied()
            .unwrap_or(byte_offset);
        let addr = builder.ins().iadd(ptr, offset);
        let value = builder
            .ins()
            .load(expr_ctx.cl_type, MemFlags::new(), addr, 0);
        memo.insert(id, value);
        return Ok(value);
    }

    let node = expr_ctx.graph.node(id);

    let value = match &node.op {
        op if !op.is_elementwise() && !matches!(op, Op::Const(_)) => {
            // Source was inlined through a contiguous shape op chain —
            // recursively evaluate its expression tree.
            return build_expression_impl(expr_ctx, resolved_id, builder, byte_offset, memo);
        }
        Op::Load => {
            return Err(anyhow::anyhow!(
                "Load node {:?} was not materialized as a kernel input",
                id
            ))
        }
        Op::Const(scalar) => emit_const(builder, *scalar, expr_ctx.dtype)?,
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            let lhs = build_expression_impl(expr_ctx, node.inputs[0], builder, byte_offset, memo)?;
            let rhs = build_expression_impl(expr_ctx, node.inputs[1], builder, byte_offset, memo)?;
            match (node.op.clone(), expr_ctx.dtype.is_float()) {
                (Op::Add, true) => builder.ins().fadd(lhs, rhs),
                (Op::Sub, true) => builder.ins().fsub(lhs, rhs),
                (Op::Mul, true) => builder.ins().fmul(lhs, rhs),
                (Op::Div, true) => builder.ins().fdiv(lhs, rhs),
                (Op::Add, false) => builder.ins().iadd(lhs, rhs),
                (Op::Sub, false) => builder.ins().isub(lhs, rhs),
                (Op::Mul, false) => builder.ins().imul(lhs, rhs),
                (Op::Div, false) => builder.ins().sdiv(lhs, rhs),
                _ => unreachable!(),
            }
        }
        Op::Neg => {
            let val = build_expression_impl(expr_ctx, node.inputs[0], builder, byte_offset, memo)?;
            if expr_ctx.dtype.is_float() {
                builder.ins().fneg(val)
            } else {
                let zero = builder.ins().iconst(expr_ctx.cl_type, 0);
                builder.ins().isub(zero, val)
            }
        }
        Op::Exp => {
            let val = build_expression_impl(expr_ctx, node.inputs[0], builder, byte_offset, memo)?;
            let func_ref = math_func_ref_for_op(expr_ctx.dtype, expr_ctx.math, "exp")?;
            let call = builder.ins().call(func_ref, &[val]);
            builder.inst_results(call)[0]
        }
        Op::Ln => {
            let val = build_expression_impl(expr_ctx, node.inputs[0], builder, byte_offset, memo)?;
            let func_ref = math_func_ref_for_op(expr_ctx.dtype, expr_ctx.math, "ln")?;
            let call = builder.ins().call(func_ref, &[val]);
            builder.inst_results(call)[0]
        }
        Op::Sqrt => {
            let val = build_expression_impl(expr_ctx, node.inputs[0], builder, byte_offset, memo)?;
            if !expr_ctx.dtype.is_float() {
                return Err(anyhow::anyhow!(
                    "sqrt not supported for {:?}",
                    expr_ctx.dtype
                ));
            }
            builder.ins().sqrt(val)
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported op in JIT expression builder for node {:?}: {:?}",
                id,
                node.op
            ))
        }
    };

    memo.insert(id, value);
    Ok(value)
}

/// Emit a constant value instruction for the given Scalar and DType.
pub fn emit_const(builder: &mut FunctionBuilder, scalar: Scalar, dtype: DType) -> Result<Value> {
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
