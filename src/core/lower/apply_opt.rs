use anyhow::{Result, anyhow, bail};

use crate::core::hlir::BufferId;
use crate::core::llir::affine::AffineExpr;
use crate::core::llir::loop_nest::{Loop, LoopKind, LoopNest, ReductionAccumulator};
use crate::core::llir::memory::{AccessKind, BufferAlloc, MemoryAccess, MemorySpace};
use crate::core::llir::stmt::{AbstractVectorOp, BinaryOp, Expr, Stmt};
use crate::core::schedule::{Opt, OptOp};

use super::axis::validate_axis;

pub struct OptApplyResult {
    pub nest: LoopNest,
    pub allocs: Vec<BufferAlloc>,
}

pub fn apply_opt(mut nest: LoopNest, opt: &Opt) -> Result<OptApplyResult> {
    match opt.op {
        OptOp::Tile => apply_tile(&mut nest, opt)?,
        OptOp::Vectorize => apply_vectorize(&mut nest, opt)?,
        OptOp::Unroll => apply_unroll(&mut nest, opt)?,
        OptOp::Parallelize => apply_parallelize(&mut nest, opt)?,
        OptOp::PadTo => apply_pad_to(&mut nest, opt)?,
        OptOp::GroupReduce => {
            let alloc = apply_group_reduce(&mut nest, opt)?;
            return Ok(OptApplyResult {
                nest,
                allocs: vec![alloc],
            });
        }
    }

    Ok(OptApplyResult {
        nest,
        allocs: Vec::new(),
    })
}

fn apply_tile(nest: &mut LoopNest, opt: &Opt) -> Result<()> {
    validate_axis(&nest.loops, opt.axis)?;
    let amt = positive_amt(opt)?;
    let axis = opt.axis;
    let original = nest.loops[axis].clone();
    let original_upper = original
        .upper
        .as_const_value()
        .ok_or_else(|| anyhow!("tile requires constant loop bound"))?;

    nest.loops[axis].var = format!("{}_outer", original.var);
    nest.loops[axis].upper = AffineExpr::constant(div_ceil_i64(original_upper, amt));

    let mut inner = original;
    inner.var = format!("{}_inner", inner.var);
    inner.upper = AffineExpr::constant(amt);
    nest.loops.insert(axis + 1, inner);

    if original_upper % amt != 0 {
        nest.body.push(Stmt::Epilogue {
            main_loop_var: nest.loops[axis].var.clone(),
            remainder_body: Vec::new(),
        });
    }
    Ok(())
}

fn apply_parallelize(nest: &mut LoopNest, opt: &Opt) -> Result<()> {
    apply_tile(nest, opt)?;
    nest.loops[opt.axis].kind = LoopKind::Parallel;
    Ok(())
}

fn apply_vectorize(nest: &mut LoopNest, opt: &Opt) -> Result<()> {
    validate_axis(&nest.loops, opt.axis)?;
    let width = usize::try_from(positive_amt(opt)?).map_err(|_| anyhow!("invalid vector width"))?;

    if matches!(nest.loops[opt.axis].kind, LoopKind::Reduce { .. }) {
        bail!("Vectorize is illegal on reduce axis");
    }

    nest.loops[opt.axis].kind = LoopKind::Vectorized { width };
    lift_body_to_vector_ops(&mut nest.body, width);
    Ok(())
}

fn apply_unroll(nest: &mut LoopNest, opt: &Opt) -> Result<()> {
    validate_axis(&nest.loops, opt.axis)?;
    let factor =
        usize::try_from(positive_amt(opt)?).map_err(|_| anyhow!("invalid unroll factor"))?;
    nest.loops[opt.axis].kind = LoopKind::Unrolled { factor };
    Ok(())
}

fn apply_pad_to(nest: &mut LoopNest, opt: &Opt) -> Result<()> {
    validate_axis(&nest.loops, opt.axis)?;
    let amt = positive_amt(opt)?;
    let lp = &mut nest.loops[opt.axis];
    let old_upper = lp
        .upper
        .as_const_value()
        .ok_or_else(|| anyhow!("pad_to requires constant loop bound"))?;
    let padded_upper = div_ceil_i64(old_upper, amt) * amt;
    lp.upper = AffineExpr::constant(padded_upper);

    if padded_upper != old_upper {
        let old_body = std::mem::take(&mut nest.body);
        nest.body = vec![Stmt::If {
            cond: Expr::Literal(crate::core::hlir::Scalar::Bool(true)),
            then_body: old_body,
            else_body: Vec::new(),
        }];
    }
    Ok(())
}

fn apply_group_reduce(nest: &mut LoopNest, opt: &Opt) -> Result<BufferAlloc> {
    validate_axis(&nest.loops, opt.axis)?;
    let amt = positive_amt(opt)?;
    let axis = opt.axis;

    if !matches!(nest.loops[axis].kind, LoopKind::Reduce { .. }) {
        bail!("GroupReduce is only legal on Reduce-kinded loops");
    }

    let reduce_loop = nest.loops[axis].clone();
    let accumulators = match reduce_loop.kind {
        LoopKind::Reduce { accumulators } => accumulators,
        _ => Vec::new(),
    };
    let acc = accumulators
        .first()
        .cloned()
        .unwrap_or(ReductionAccumulator {
            var: "acc".to_owned(),
            op: crate::core::hlir::ReduceOp::Sum,
            init: crate::core::hlir::Scalar::F32(0.0),
            dtype: crate::core::hlir::DType::F32,
        });

    let parallel = Loop {
        var: "t".to_owned(),
        lower: AffineExpr::constant(0),
        upper: AffineExpr::constant(amt),
        step: 1,
        kind: LoopKind::Parallel,
        annotations: reduce_loop.annotations.clone(),
    };

    nest.loops[axis] = parallel;
    nest.loops.insert(
        axis + 1,
        Loop {
            var: "k".to_owned(),
            lower: AffineExpr::constant(0),
            upper: AffineExpr::constant(div_ceil_i64(
            reduce_loop.upper.as_const_value().unwrap_or(1),
            amt,
        )),
            step: 1,
            kind: LoopKind::Sequential,
            annotations: reduce_loop.annotations,
        },
    );

    let staging = BufferId(usize::MAX - axis);
    let read_staging = Expr::Load(MemoryAccess {
        buffer: staging,
        indices: vec![AffineExpr::constant(0)],
        access_kind: AccessKind::Read,
    });

    let reduce_stmt = Stmt::Loop(
        Loop {
            var: "r".to_owned(),
            lower: AffineExpr::constant(0),
            upper: AffineExpr::constant(amt),
            step: 1,
            kind: LoopKind::Reduce {
                accumulators: vec![acc.clone()],
            },
            annotations: Default::default(),
        },
        vec![Stmt::Accumulate {
            dst: MemoryAccess {
                buffer: BufferId(0),
                indices: vec![AffineExpr::constant(0)],
                access_kind: AccessKind::ReadWrite,
            },
            op: acc.op,
            src: read_staging,
        }],
    );

    nest.body.push(Stmt::Barrier);
    nest.body.push(reduce_stmt);

    Ok(BufferAlloc {
        id: staging,
        shape: vec![AffineExpr::constant(amt)],
        dtype: acc.dtype,
        memory_space: MemorySpace::Shared,
    })
}

fn lift_body_to_vector_ops(stmts: &mut [Stmt], width: usize) {
    for stmt in stmts {
        match stmt {
            Stmt::Assign { src, .. } | Stmt::Accumulate { src, .. } => {
                let replacement = match src {
                    Expr::Binary {
                        op: BinaryOp::Mul,
                        lhs,
                        rhs,
                    } => Some(Expr::AbstractVector(AbstractVectorOp::Fma {
                        acc: Box::new(Expr::Literal(crate::core::hlir::Scalar::F32(0.0))),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        width,
                    })),
                    Expr::Binary { op, lhs, rhs } => {
                        Some(Expr::AbstractVector(AbstractVectorOp::VecBinary {
                            op: *op,
                            lhs: lhs.clone(),
                            rhs: rhs.clone(),
                            width,
                        }))
                    }
                    _ => None,
                };
                if let Some(new_src) = replacement {
                    *src = new_src;
                }
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                lift_body_to_vector_ops(then_body, width);
                lift_body_to_vector_ops(else_body, width);
            }
            Stmt::Loop(_, body) => lift_body_to_vector_ops(body, width),
            Stmt::Epilogue { remainder_body, .. } => lift_body_to_vector_ops(remainder_body, width),
            Stmt::Barrier => {}
        }
    }
}

fn positive_amt(opt: &Opt) -> Result<i64> {
    if opt.amt <= 0 {
        bail!("opt amount must be positive, got {}", opt.amt);
    }
    Ok(opt.amt)
}

fn div_ceil_i64(n: i64, d: i64) -> i64 {
    (n + d - 1) / d
}
