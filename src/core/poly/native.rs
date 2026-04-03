use anyhow::Result;

use super::access_map::memory_access_to_access_map;

use crate::core::llir::{
    AffineConstraint, ConstraintKind, DepKind, Dependence, DependenceRelation, Kernel, Loop,
    MemoryAccess,
};
use crate::core::traits::{DependenceAnalyzer, ScheduleTransform};

#[derive(Default)]
pub struct NativeDependenceAnalyzer;

impl DependenceAnalyzer for NativeDependenceAnalyzer {
    fn analyze_kernel(&self, kernel: &Kernel) -> Result<Vec<Dependence>> {
        let mut deps = Vec::new();
        let mut writes: Vec<MemoryAccess> = Vec::new();
        let mut reads: Vec<MemoryAccess> = Vec::new();
        collect_accesses(&kernel.loop_nest.body, &mut reads, &mut writes);

        for w in &writes {
            for r in &reads {
                if let Some(rel) = self.access_dependence(w, r, &kernel.loop_nest.loops)? {
                    deps.push(Dependence {
                        from: 0,
                        to: 0,
                        kind: DepKind::Raw,
                        distance: Some(vec![0; kernel.loop_nest.loops.len()]),
                        relation: rel,
                    });
                }
            }
        }

        Ok(deps)
    }

    fn check_legality(&self, deps: &[Dependence], transform: &ScheduleTransform) -> Result<bool> {
        let legal = match transform {
            ScheduleTransform::Interchange { outer, inner } => !deps.iter().any(|d| {
                d.distance
                    .as_ref()
                    .map(|dist| has_positive_distance_for_pair(dist, outer, inner, &d.relation))
                    .unwrap_or(false)
            }),
            ScheduleTransform::Vectorize { .. } => !deps.iter().any(|d| d.kind == DepKind::Waw),
            ScheduleTransform::Parallelize { .. } => !deps.iter().any(|d| {
                d.distance
                    .as_ref()
                    .is_some_and(|dist| dist.iter().any(|x| *x > 0))
            }),
            _ => true,
        };
        Ok(legal)
    }

    fn access_dependence(
        &self,
        write: &MemoryAccess,
        read: &MemoryAccess,
        loops: &[Loop],
    ) -> Result<Option<DependenceRelation>> {
        if write.buffer != read.buffer {
            return Ok(None);
        }
        if write.indices.len() != read.indices.len() {
            return Ok(None);
        }

        let source_vars: Vec<String> = loops.iter().map(|l| l.var.clone()).collect();
        let sink_vars = source_vars.clone();
        let write_map = memory_access_to_access_map(write, &source_vars);
        let read_map = memory_access_to_access_map(read, &sink_vars);

        if write_map.mapping.len() != read_map.mapping.len() {
            return Ok(None);
        }

        let mut constraints = Vec::new();
        for (w_idx, r_idx) in write.indices.iter().zip(read.indices.iter()) {
            let eq_expr = w_idx.sub(r_idx);
            constraints.push(AffineConstraint {
                expr: eq_expr,
                kind: ConstraintKind::Eq,
            });
        }

        Ok(Some(DependenceRelation {
            source_vars,
            sink_vars,
            constraints,
        }))
    }
}

fn has_positive_distance_for_pair(
    dist: &[i64],
    outer: &str,
    inner: &str,
    relation: &DependenceRelation,
) -> bool {
    let oi = relation.source_vars.iter().position(|v| v == outer);
    let ii = relation.source_vars.iter().position(|v| v == inner);
    match (oi, ii) {
        (Some(o), Some(i)) if o < dist.len() && i < dist.len() => dist[o] > 0 || dist[i] > 0,
        _ => false,
    }
}

fn collect_accesses(
    body: &[crate::core::llir::Stmt],
    reads: &mut Vec<MemoryAccess>,
    writes: &mut Vec<MemoryAccess>,
) {
    for stmt in body {
        match stmt {
            crate::core::llir::Stmt::Assign { dst, src } => {
                writes.push(dst.clone());
                collect_expr_reads(src, reads);
            }
            crate::core::llir::Stmt::Accumulate { dst, src, .. } => {
                writes.push(dst.clone());
                reads.push(dst.clone());
                collect_expr_reads(src, reads);
            }
            crate::core::llir::Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                collect_expr_reads(cond, reads);
                collect_accesses(then_body, reads, writes);
                collect_accesses(else_body, reads, writes);
            }
            crate::core::llir::Stmt::Loop(_, sub) => collect_accesses(sub, reads, writes),
            crate::core::llir::Stmt::Epilogue { remainder_body, .. } => {
                collect_accesses(remainder_body, reads, writes)
            }
            crate::core::llir::Stmt::Barrier => {}
        }
    }
}

fn collect_expr_reads(expr: &crate::core::llir::stmt::Expr, reads: &mut Vec<MemoryAccess>) {
    match expr {
        crate::core::llir::stmt::Expr::Load(ma) => reads.push(ma.clone()),
        crate::core::llir::stmt::Expr::Unary { arg, .. } => collect_expr_reads(arg, reads),
        crate::core::llir::stmt::Expr::Binary { lhs, rhs, .. } => {
            collect_expr_reads(lhs, reads);
            collect_expr_reads(rhs, reads);
        }
        crate::core::llir::stmt::Expr::Ternary {
            cond,
            then_val,
            else_val,
        } => {
            collect_expr_reads(cond, reads);
            collect_expr_reads(then_val, reads);
            collect_expr_reads(else_val, reads);
        }
        crate::core::llir::stmt::Expr::Cast { arg, .. } => collect_expr_reads(arg, reads),
        crate::core::llir::stmt::Expr::AbstractVector(op) => {
            collect_vector_reads(op, reads);
        }
        crate::core::llir::stmt::Expr::Literal(_) => {}
    }
}

fn collect_vector_reads(
    op: &crate::core::llir::stmt::AbstractVectorOp,
    reads: &mut Vec<MemoryAccess>,
) {
    match op {
        crate::core::llir::stmt::AbstractVectorOp::Fma { acc, lhs, rhs, .. } => {
            collect_expr_reads(acc, reads);
            collect_expr_reads(lhs, reads);
            collect_expr_reads(rhs, reads);
        }
        crate::core::llir::stmt::AbstractVectorOp::HorizontalReduce { arg, .. }
        | crate::core::llir::stmt::AbstractVectorOp::Broadcast { scalar: arg, .. }
        | crate::core::llir::stmt::AbstractVectorOp::VecCast { arg, .. } => {
            collect_expr_reads(arg, reads)
        }
        crate::core::llir::stmt::AbstractVectorOp::Gather { .. } => {}
        crate::core::llir::stmt::AbstractVectorOp::Scatter { value, .. } => {
            collect_expr_reads(value, reads);
        }
        crate::core::llir::stmt::AbstractVectorOp::VecBinary { lhs, rhs, .. } => {
            collect_expr_reads(lhs, reads);
            collect_expr_reads(rhs, reads);
        }
    }
}
