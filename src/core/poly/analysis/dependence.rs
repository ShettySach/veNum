use crate::core::llir::dependence::{
    AffineConstraint, ConstraintKind, DepKind, Dependence, DependenceRelation,
};
use crate::core::llir::program::Kernel;

use crate::core::llir::affine::AffineExpr;
use crate::core::poly::access_map::AccessMap;
use crate::core::poly::analysis::extract::{StatementInstance, extract_instances};
use crate::core::poly::native::distance::compute_distance;
use crate::core::poly::sets::relation::{Relation, shared_iter_names};

/// Analyze a kernel for data dependences using the polyhedral model.
///
/// Pipeline:
/// 1. Extract normalized statement instances (domains + access maps).
/// 2. Pair candidate write/read and write/write accesses by buffer.
/// 3. Build dependence relations (domain + memory-equality + order).
/// 4. Check feasibility — only emit a dependence if the relation is
///    non-empty.
/// 5. Derive conservative distance/direction summaries.
pub fn analyze_kernel_poly(kernel: &Kernel) -> Vec<Dependence> {
    let instances = extract_instances(kernel);
    let mut deps = Vec::new();

    // For each pair of statements, check write→read (RAW) and
    // write→write (WAW) dependences.
    for (si_idx, si) in instances.iter().enumerate() {
        for (sj_idx, sj) in instances.iter().enumerate() {
            // RAW: si writes, sj reads.
            for w in &si.writes {
                for r in &sj.reads {
                    if w.buffer != r.buffer {
                        continue;
                    }
                    if w.mapping.len() != r.mapping.len() {
                        continue;
                    }
                    if let Some(dep) = build_dependence(si, sj, w, r, si_idx, sj_idx, DepKind::Raw)
                    {
                        deps.push(dep);
                    }
                }
            }

            // WAW: both write to the same buffer.
            if si_idx != sj_idx {
                for wi in &si.writes {
                    for wj in &sj.writes {
                        if wi.buffer != wj.buffer {
                            continue;
                        }
                        if wi.mapping.len() != wj.mapping.len() {
                            continue;
                        }
                        if let Some(dep) =
                            build_dependence(si, sj, wi, wj, si_idx, sj_idx, DepKind::Waw)
                        {
                            deps.push(dep);
                        }
                    }
                }
            }
        }
    }

    deps
}

/// Build a single dependence if the relation is feasible.
fn build_dependence(
    source: &StatementInstance,
    sink: &StatementInstance,
    write_access: &AccessMap,
    read_access: &AccessMap,
    from_id: usize,
    to_id: usize,
    kind: DepKind,
) -> Option<Dependence> {
    let shared = shared_iter_names(&source.domain.iters, &sink.domain.iters);
    if shared.is_empty() {
        return None;
    }

    let mut feasible_rel: Option<Relation> = None;
    for order_dim in 0..shared.len() {
        let rel = Relation::build_dependence_with_order_dim(
            &source.domain,
            &sink.domain,
            write_access,
            read_access,
            order_dim,
        );
        if !rel.system.is_empty() {
            feasible_rel = Some(rel);
            break;
        }
    }

    let rel = feasible_rel?;

    // Compute distance/direction.
    let (distance, _directions) = compute_distance(
        &rel.system,
        &rel.source_iters,
        &rel.sink_iters,
        Relation::SOURCE_PREFIX,
        Relation::SINK_PREFIX,
    );

    // Build the LLIR-level DependenceRelation for backward compatibility.
    let llir_relation = relation_to_llir(&rel);

    Some(Dependence {
        from: from_id,
        to: to_id,
        kind,
        distance,
        relation: llir_relation,
    })
}

/// Convert a poly `Relation` to an LLIR `DependenceRelation`.
fn relation_to_llir(rel: &Relation) -> DependenceRelation {
    let mut constraints = Vec::new();

    for eq in &rel.system.equalities {
        let expr: AffineExpr = eq.into();
        constraints.push(AffineConstraint {
            expr,
            kind: ConstraintKind::Eq,
        });
    }
    for ineq in &rel.system.inequalities {
        let expr: AffineExpr = ineq.into();
        constraints.push(AffineConstraint {
            expr,
            kind: ConstraintKind::Ge,
        });
    }

    DependenceRelation {
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
        constraints,
    }
}
