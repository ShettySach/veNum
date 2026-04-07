pub mod distance;
pub mod feasibility;
pub mod fm;

use anyhow::Result;

use super::access_map::memory_access_to_access_map;
use super::analysis::dependence::analyze_kernel_poly;
use super::analysis::legality;
use super::domain::{Aff, Constraint, Domain, IterName};
use super::sets::Relation;

use crate::core::hlir::Symbol;
use crate::core::llir::dependence::{AffineConstraint, ConstraintKind};
use crate::core::llir::{Dependence, DependenceRelation, Kernel, Loop, MemoryAccess};
use crate::core::traits::{DependenceAnalyzer, ScheduleTransform};

#[derive(Default)]
pub struct NativeDependenceAnalyzer;

impl DependenceAnalyzer for NativeDependenceAnalyzer {
    fn analyze_kernel(&self, kernel: &Kernel) -> Result<Vec<Dependence>> {
        Ok(analyze_kernel_poly(kernel))
    }

    fn check_legality(
        &self,
        deps: &[Dependence],
        transform: &ScheduleTransform,
        kernel: &Kernel,
    ) -> Result<bool> {
        let legal = match transform {
            ScheduleTransform::Parallelize { loop_var } => {
                legality::can_parallelize(deps, loop_var)
            }
            ScheduleTransform::Interchange { outer, inner } => {
                legality::can_interchange(deps, outer, inner)
            }
            ScheduleTransform::Vectorize { loop_var, .. } => {
                legality::can_vectorize(deps, loop_var, kernel)
            }
            ScheduleTransform::Tile { loop_var, .. } => legality::can_tile(deps, loop_var),
            ScheduleTransform::Unroll { loop_var, .. } => legality::can_unroll(deps, loop_var),
            ScheduleTransform::GroupReduce { loop_var } => {
                legality::can_group_reduce(deps, loop_var, kernel)
            }
            ScheduleTransform::PadTo { loop_var } => legality::can_pad_to(deps, loop_var),
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

        let domain = loops_to_domain(loops);
        let loop_vars = domain.iters.clone();
        let write_map = memory_access_to_access_map(write, &loop_vars);
        let read_map = memory_access_to_access_map(read, &loop_vars);

        let mut feasible_relation = None;
        for order_dim in 0..loop_vars.len() {
            let rel = Relation::build_dependence_with_order_dim(
                &domain, &domain, &write_map, &read_map, order_dim,
            );
            if !rel.system.is_empty() {
                feasible_relation = Some(rel);
                break;
            }
        }

        let rel = match feasible_relation {
            Some(rel) => rel,
            None => return Ok(None),
        };

        Ok(Some(DependenceRelation {
            source_vars: rel
                .source_iters
                .iter()
                .map(|n| n.as_str().to_owned())
                .collect(),
            sink_vars: rel
                .sink_iters
                .iter()
                .map(|n| n.as_str().to_owned())
                .collect(),
            constraints: relation_constraints_to_llir(&rel),
        }))
    }
}

fn loops_to_domain(loops: &[Loop]) -> Domain {
    let mut iters = Vec::with_capacity(loops.len());
    let mut params = Vec::new();
    let mut constraints = Vec::with_capacity(loops.len() * 4);

    for lp in loops {
        iters.push(IterName::from(lp.var.clone()));
        collect_params(&lp.lower, &mut params);
        collect_params(&lp.upper, &mut params);

        constraints.push(Constraint::Ineq(
            Aff::iter_var(lp.var.as_str()).sub(&Aff::from(&lp.lower)),
        ));
        constraints.push(Constraint::Ineq(
            Aff::from(&lp.upper)
                .sub(&Aff::iter_var(lp.var.as_str()))
                .add(&Aff::constant(-1)),
        ));
    }

    Domain {
        iters,
        params,
        constraints,
    }
}

fn collect_params(expr: &crate::core::llir::affine::AffineExpr, params: &mut Vec<Symbol>) {
    for (_, var) in &expr.terms {
        if let crate::core::llir::affine::Var::Param(sym) = var
            && !params.contains(sym)
        {
            params.push(*sym);
        }
    }
}

fn relation_constraints_to_llir(rel: &Relation) -> Vec<AffineConstraint> {
    let mut constraints = Vec::new();

    for eq in &rel.system.equalities {
        constraints.push(AffineConstraint {
            expr: eq.into(),
            kind: ConstraintKind::Eq,
        });
    }
    for ineq in &rel.system.inequalities {
        constraints.push(AffineConstraint {
            expr: ineq.into(),
            kind: ConstraintKind::Ge,
        });
    }

    constraints
}
