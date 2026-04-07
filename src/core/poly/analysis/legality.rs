use crate::core::llir::dependence::{DepKind, Dependence};
use crate::core::llir::loop_nest::LoopKind;
use crate::core::llir::program::Kernel;

/// Check whether a loop can be parallelized.
///
/// A loop is parallelizable when no dependence carries a positive distance
/// on that loop dimension.  Unknown distances are treated conservatively
/// as carried (i.e. parallelization is rejected).
pub fn can_parallelize(deps: &[Dependence], loop_var: &str) -> bool {
    !deps.iter().any(|d| is_carried_on(d, loop_var))
}

/// Check whether two loops can be interchanged.
///
/// Interchange of `outer` and `inner` is legal when no dependence has a
/// distance vector that would become lexicographically negative after the
/// swap.
///
/// Concretely, for each dependence:
///   - If the distance on `outer` is positive, swapping would put the
///     inner distance first.  If that inner distance is negative, the
///     reordering is illegal.
///   - If the outer distance is zero, the inner dimension becomes the new
///     outer, and its sign must stay non-negative.
///
/// Unknown distances conservatively reject the interchange.
pub fn can_interchange(deps: &[Dependence], outer: &str, inner: &str) -> bool {
    for d in deps {
        let (d_outer, d_inner) = match &d.distance {
            Some(dist) => {
                let oi = d.relation.source_vars.iter().position(|v| v == outer);
                let ii = d.relation.source_vars.iter().position(|v| v == inner);
                match (oi, ii) {
                    (Some(o), Some(i)) if o < dist.len() && i < dist.len() => (dist[o], dist[i]),
                    _ => return false, // can't find vars → conservative reject
                }
            }
            None => return false, // unknown → conservative reject
        };

        // After interchange the distance vector at these positions becomes
        // (d_inner, d_outer).  This must remain lexicographically non-negative.
        if d_inner < 0 {
            return false;
        }
        if d_inner == 0 && d_outer < 0 {
            return false;
        }
    }
    true
}

/// Check whether a loop can be vectorized.
///
/// Vectorization is legal when:
/// - No WAW dependence exists (would require masking).
/// - No loop-carried RAW dependence exists on the target loop.
/// - The target loop is not a reduction loop.
pub fn can_vectorize(deps: &[Dependence], loop_var: &str, kernel: &Kernel) -> bool {
    // Reject vectorization on reduce loops.
    let is_reduce = kernel
        .loop_nest
        .loops
        .iter()
        .any(|l| l.var == loop_var && matches!(l.kind, LoopKind::Reduce { .. }));
    if is_reduce {
        return false;
    }

    // Reject if any WAW exists.
    if deps.iter().any(|d| d.kind == DepKind::Waw) {
        return false;
    }

    // Reject if a carried RAW dependence exists on this loop.
    !deps
        .iter()
        .filter(|d| d.kind == DepKind::Raw)
        .any(|d| is_carried_on(d, loop_var))
}

/// Check whether a GroupReduce transform is legal on a given loop.
///
/// GroupReduce is only legal on loops with `LoopKind::Reduce`.  The
/// dependence must be a reduction-carried dependence (accumulator
/// read-write on the same buffer), not an arbitrary loop-carried
/// dependence.
pub fn can_group_reduce(deps: &[Dependence], loop_var: &str, kernel: &Kernel) -> bool {
    let is_reduce = kernel
        .loop_nest
        .loops
        .iter()
        .any(|l| l.var == loop_var && matches!(l.kind, LoopKind::Reduce { .. }));
    if !is_reduce {
        return false;
    }

    // For a reduce loop, the only carried dependences should be the
    // accumulator's self-dependence.  If there are non-reduction carried
    // dependences, reject.
    //
    // Currently we allow GroupReduce as long as the loop is a reduce loop
    // and no WAW dependences exist (which would indicate conflicting writes
    // that can't be safely split across groups).
    !deps.iter().any(|d| d.kind == DepKind::Waw)
}

/// Check whether a PadTo transform is legal.
///
/// PadTo only extends the loop bound and wraps the body in a guard.
/// It does not reorder iterations, so it is always legal.
pub fn can_pad_to(_deps: &[Dependence], _loop_var: &str) -> bool {
    true
}

/// Check whether a Tile transform is legal.
///
/// Tiling does not reorder iterations within a tile when the tile inner
/// loop preserves the original order.  It is always legal for sequential
/// loops.
pub fn can_tile(_deps: &[Dependence], _loop_var: &str) -> bool {
    true
}

/// Check whether an Unroll transform is legal.
///
/// Unrolling does not change iteration order, so it is always legal.
pub fn can_unroll(_deps: &[Dependence], _loop_var: &str) -> bool {
    true
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether a dependence is carried on a specific loop dimension.
///
/// A dependence is carried on dimension `loop_var` when:
/// - Its distance on that dimension is positive, or
/// - Its distance is unknown (conservative).
fn is_carried_on(dep: &Dependence, loop_var: &str) -> bool {
    match &dep.distance {
        Some(dist) => {
            let idx = dep.relation.source_vars.iter().position(|v| v == loop_var);
            match idx {
                Some(i) if i < dist.len() => dist[i] > 0,
                _ => true, // can't find → conservative
            }
        }
        None => true, // unknown → conservative
    }
}
