use anyhow::Result;

use crate::core::hlir::{Dim, HLIRGraph, Op};
use crate::core::llir::affine::{AffineExpr, Var};
use crate::core::llir::loop_nest::{
    Loop, LoopAnnotations, LoopKind, LoopNest, ReductionAccumulator,
};
use crate::core::llir::memory::{AccessKind, MemoryAccess};
use crate::core::llir::program::{Kernel, KernelId, LLIRProgram};
use crate::core::llir::stmt::{BinaryOp, Expr, Stmt};
use crate::core::schedule::ScheduleDecision;
use crate::core::traits::DependenceAnalyzer;

mod apply_opt;
mod axis;
mod fusion_merge;
mod legality;

pub fn lower(
    hlir: &HLIRGraph,
    decision: &ScheduleDecision,
    dep: &impl DependenceAnalyzer,
) -> Result<LLIRProgram> {
    let mut kernels = Vec::new();

    for (kidx, fg) in decision.fusion_groups.iter().enumerate() {
        if fg.nodes.is_empty() {
            continue;
        }

        let mut per_group_kernels = Vec::new();
        for (nidx, node) in fg.nodes.iter().copied().enumerate() {
            per_group_kernels.push(build_base_kernel(hlir, node, kidx * 1024 + nidx)?);
        }

        let group_opts = decision.opts.get(&fg.id).cloned().unwrap_or_default();
        for opt in &group_opts {
            for kernel in &mut per_group_kernels {
                // Check legality BEFORE applying the opt so we never
                // corrupt the kernel state with an illegal transform.
                if !legality::check_opt_legality(dep, kernel, opt)? {
                    return Err(anyhow::anyhow!(
                        "legality violation: {:?} is illegal on {}",
                        opt,
                        kernel.name
                    ));
                }

                let apply = apply_opt::apply_opt(kernel.loop_nest.clone(), opt)?;
                kernel.loop_nest = apply.nest;
                kernel.allocs.extend(apply.allocs);
            }
        }

        let merged =
            fusion_merge::merge_fusion_topology(hlir, &fg.topology, &fg.nodes, &per_group_kernels)?;
        kernels.extend(merged);
    }

    Ok(LLIRProgram { kernels })
}

fn build_base_kernel(
    hlir: &HLIRGraph,
    root: crate::core::hlir::NodeId,
    kidx: usize,
) -> Result<Kernel> {
    let root_node = hlir.node(root);
    let mut loops = Vec::new();
    for (i, dim) in root_node.ty.shape.iter().enumerate() {
        loops.push(Loop {
            var: format!("i{i}"),
            lower: AffineExpr::constant(0),
            upper: dim_to_affine_expr(dim),
            step: 1,
            kind: LoopKind::Sequential,
            annotations: LoopAnnotations::default(),
        });
    }

    if let Op::Reduce {
        axes,
        op,
        input,
        ..
    } = &root_node.op
    {
        let input_shape = &hlir.node(*input).ty.shape;
        for (j, &axis) in axes.iter().enumerate() {
            let reduce_dim = input_shape
                .get(axis)
                .cloned()
                .unwrap_or(Dim::Const(1));
            loops.push(Loop {
                var: format!("r{j}"),
                lower: AffineExpr::constant(0),
                upper: dim_to_affine_expr(&reduce_dim),
                step: 1,
                kind: LoopKind::Reduce {
                    accumulators: vec![ReductionAccumulator {
                        var: "acc".to_owned(),
                        op: *op,
                        init: crate::core::hlir::Scalar::F32(0.0),
                        dtype: root_node.ty.dtype,
                    }],
                },
                annotations: LoopAnnotations::default(),
            });
        }
    }

    let body = build_root_stmt(hlir, root)?;
    Ok(Kernel {
        id: KernelId(kidx),
        name: format!("kernel_{kidx}"),
        root,
        op: root_node.op.clone(),
        ty: root_node.ty.clone(),
        loop_nest: LoopNest {
            loops,
            body: vec![body],
        },
        allocs: Vec::new(),
    })
}

fn build_root_stmt(hlir: &HLIRGraph, root: crate::core::hlir::NodeId) -> Result<Stmt> {
    let node = hlir.node(root);
    let dst = MemoryAccess {
        buffer: crate::core::hlir::BufferId(0),
        indices: vec![AffineExpr::constant(0)],
        access_kind: AccessKind::Write,
    };

    let src = match &node.op {
        Op::Add(a, b) => Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(expr_from_node(*a)),
            rhs: Box::new(expr_from_node(*b)),
        },
        Op::Mul(a, b) => Expr::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(expr_from_node(*a)),
            rhs: Box::new(expr_from_node(*b)),
        },
        Op::Reduce { input, .. } => expr_from_node(*input),
        _ => expr_from_node(root),
    };

    Ok(Stmt::Assign { dst, src })
}

fn expr_from_node(id: crate::core::hlir::NodeId) -> Expr {
    Expr::Load(MemoryAccess {
        buffer: crate::core::hlir::BufferId(id.0),
        indices: vec![AffineExpr::constant(0)],
        access_kind: AccessKind::Read,
    })
}

/// Convert an HLIR `Dim` to an LLIR `AffineExpr`.
///
/// Affine cases (`Const`, `Sym`, `Add`, `Mul` by const) produce proper
/// symbolic expressions.  Non-affine cases (`Div`, `Mod`, `Sym*Sym`)
/// fall back to constant `1` so lowering never panics.
fn dim_to_affine_expr(dim: &Dim) -> AffineExpr {
    match dim {
        Dim::Const(v) => AffineExpr::constant(*v),
        Dim::Sym(sym) => AffineExpr::constant(0).with_term(1, Var::Param(*sym)),
        Dim::Add(a, b) => dim_to_affine_expr(a).add(&dim_to_affine_expr(b)),
        Dim::Mul(a, b) => {
            let ae = dim_to_affine_expr(a);
            let be = dim_to_affine_expr(b);
            if let Some(c) = ae.as_const_value() {
                be.scale(c)
            } else if let Some(c) = be.as_const_value() {
                ae.scale(c)
            } else {
                // Non-affine: both sides symbolic. Conservative fallback.
                AffineExpr::constant(1)
            }
        }
        Dim::Div(..) | Dim::Mod(..) => AffineExpr::constant(1),
    }
}
