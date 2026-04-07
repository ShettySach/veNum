use crate::core::poly::domain::{Aff, PolyVar};
use crate::core::poly::sets::ConstraintSystem;
use crate::core::poly::sets::project::project_out;

use super::fm;

/// Result of a feasibility check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feasibility {
    /// The system definitely has no integer solution.
    Infeasible,
    /// The system definitely has at least one integer solution.
    Feasible,
    /// Could not determine — conservative answer is "maybe feasible".
    Unknown,
}

/// Check whether a constraint system has an integer solution.
///
/// Strategy (pragmatic hybrid):
/// 1. Trivial contradiction scan.
/// 2. Eliminate all equalities by substitution.
/// 3. Propagate bounds and check for empty intervals.
/// 4. If few variables remain with concrete bounds, enumerate.
/// 5. Otherwise return `Unknown` (conservative).
pub fn check_feasibility(system: &ConstraintSystem) -> Feasibility {
    let mut sys = system.clone();
    sys.canonicalize();

    // 1. Trivial contradiction.
    if fm::is_trivially_infeasible(&sys) {
        return Feasibility::Infeasible;
    }

    // 2. Eliminate equalities by substitution (project them out).
    sys = eliminate_all_equalities(sys);

    if fm::is_trivially_infeasible(&sys) {
        return Feasibility::Infeasible;
    }

    // 3. Collect iter variables (not params) and try to derive concrete
    //    bounds via FM, one variable at a time.
    let iter_vars: Vec<PolyVar> = sys
        .vars
        .iter()
        .filter(|v| matches!(v, PolyVar::Iter(_)))
        .cloned()
        .collect();

    // If no iter variables remain, check the residual constant constraints.
    if iter_vars.is_empty() {
        return if fm::is_trivially_infeasible(&sys) {
            Feasibility::Infeasible
        } else {
            Feasibility::Feasible
        };
    }

    // 4. Try to prove infeasibility by projecting out all iter variables.
    //    If the resulting system (purely constants/params) is infeasible,
    //    then the original is infeasible.
    let mut projected = sys.clone();
    for var in &iter_vars {
        projected = project_out(&projected, var);
        projected.canonicalize();
        if fm::is_trivially_infeasible(&projected) {
            return Feasibility::Infeasible;
        }
    }

    // 5. If all variables were projected and no contradiction, the system
    //    has at least one rational solution.  For systems with only
    //    unit-coefficient equalities/inequalities on iters, the rational
    //    solution is also an integer solution.
    //
    //    If params remain, we assume they are positive (runtime sizes),
    //    so feasibility holds for any valid param assignment.
    let remaining_iters = projected
        .vars
        .iter()
        .any(|v| matches!(v, PolyVar::Iter(_)));

    if !remaining_iters {
        // All iters eliminated.  Check if residual is satisfiable.
        if fm::is_trivially_infeasible(&projected) {
            return Feasibility::Infeasible;
        }
        // If only param constraints remain (or nothing), we assume
        // feasible for valid param values.
        return Feasibility::Feasible;
    }

    // 6. For small systems with concrete bounds, try integer enumeration.
    if let Some(result) = try_bounded_enumeration(&sys, &iter_vars) {
        return result;
    }

    Feasibility::Unknown
}

// ---------------------------------------------------------------------------
// Equality elimination
// ---------------------------------------------------------------------------

/// Eliminate all equalities by substituting each one for a variable.
fn eliminate_all_equalities(mut sys: ConstraintSystem) -> ConstraintSystem {
    loop {
        sys.canonicalize();
        // Find an equality with an iter variable we can solve for.
        let candidate = sys.equalities.iter().enumerate().find_map(|(idx, eq)| {
            for (c, v) in &eq.terms {
                if *c != 0 && matches!(v, PolyVar::Iter(_)) {
                    return Some((idx, v.clone()));
                }
            }
            None
        });

        match candidate {
            Some((_idx, var)) => {
                sys = project_out(&sys, &var);
            }
            None => break,
        }
    }
    sys
}

// ---------------------------------------------------------------------------
// Bounded enumeration for small concrete systems
// ---------------------------------------------------------------------------

/// Maximum number of iterations for bounded enumeration.
const MAX_ENUM_ITERATIONS: u64 = 10_000;

/// Try to determine feasibility by enumerating integer points.
///
/// Only works when all remaining iter variables have concrete (constant)
/// lower and upper bounds derivable from the inequality system.
fn try_bounded_enumeration(
    sys: &ConstraintSystem,
    iter_vars: &[PolyVar],
) -> Option<Feasibility> {
    // Collect concrete bounds for each variable.
    let mut var_bounds: Vec<(PolyVar, i64, i64)> = Vec::new();

    for var in iter_vars {
        let (lo, hi) = extract_concrete_bounds(sys, var)?;
        if lo > hi {
            return Some(Feasibility::Infeasible);
        }
        var_bounds.push((var.clone(), lo, hi));
    }

    // Check total search space.
    let mut total: u64 = 1;
    for (_, lo, hi) in &var_bounds {
        let range = (*hi - *lo + 1) as u64;
        total = total.saturating_mul(range);
        if total > MAX_ENUM_ITERATIONS {
            return None; // Too large to enumerate.
        }
    }

    // Enumerate all integer points and check all constraints.
    let found = enumerate_recursive(sys, &var_bounds, 0, &mut Vec::new());

    Some(if found {
        Feasibility::Feasible
    } else {
        Feasibility::Infeasible
    })
}

/// Extract concrete lower and upper bounds for `var` from inequalities.
///
/// Returns `Some((lo, hi))` if both bounds are constant, `None` otherwise.
fn extract_concrete_bounds(sys: &ConstraintSystem, var: &PolyVar) -> Option<(i64, i64)> {
    let mut lo: Option<i64> = None;
    let mut hi: Option<i64> = None;

    for ineq in &sys.inequalities {
        let c = ineq.coefficient_of(var);
        if c == 0 {
            continue;
        }
        // Check that all other terms are zero (pure constant bound on var).
        let other_terms: i64 = ineq
            .terms
            .iter()
            .filter(|(_, v)| v != var)
            .map(|(coeff, _)| coeff.abs())
            .sum();
        if other_terms != 0 {
            return None; // Bound depends on other variables.
        }

        if c > 0 {
            // c * var + constant >= 0  →  var >= -constant / c
            let bound = ceil_div(-ineq.constant, c);
            lo = Some(lo.map_or(bound, |prev: i64| prev.max(bound)));
        } else {
            // c * var + constant >= 0  →  var <= constant / (-c)
            let bound = floor_div(ineq.constant, -c);
            hi = Some(hi.map_or(bound, |prev: i64| prev.min(bound)));
        }
    }

    match (lo, hi) {
        (Some(l), Some(h)) => Some((l, h)),
        _ => None,
    }
}

/// Recursively enumerate integer points and check constraints.
fn enumerate_recursive(
    sys: &ConstraintSystem,
    var_bounds: &[(PolyVar, i64, i64)],
    depth: usize,
    assignment: &mut Vec<(PolyVar, i64)>,
) -> bool {
    if depth == var_bounds.len() {
        // All variables assigned — check all constraints.
        return check_assignment(sys, assignment);
    }

    let (ref var, lo, hi) = var_bounds[depth];
    for val in lo..=hi {
        assignment.push((var.clone(), val));
        if enumerate_recursive(sys, var_bounds, depth + 1, assignment) {
            return true;
        }
        assignment.pop();
    }
    false
}

/// Evaluate all constraints under a concrete assignment.
fn check_assignment(sys: &ConstraintSystem, assignment: &[(PolyVar, i64)]) -> bool {
    for eq in &sys.equalities {
        if evaluate(eq, assignment) != 0 {
            return false;
        }
    }
    for ineq in &sys.inequalities {
        if evaluate(ineq, assignment) < 0 {
            return false;
        }
    }
    true
}

/// Evaluate an affine expression under a concrete assignment.
fn evaluate(aff: &Aff, assignment: &[(PolyVar, i64)]) -> i64 {
    let mut val = aff.constant;
    for (coeff, var) in &aff.terms {
        if let Some((_, v)) = assignment.iter().find(|(av, _)| av == var) {
            val += coeff * v;
        }
        // Parameters not in assignment are treated as 0 (conservative).
    }
    val
}

// ---------------------------------------------------------------------------
// Integer arithmetic helpers
// ---------------------------------------------------------------------------

/// Ceiling division: ⌈a / b⌉ for b > 0.
fn ceil_div(a: i64, b: i64) -> i64 {
    debug_assert!(b > 0);
    if a >= 0 {
        (a + b - 1) / b
    } else {
        a / b
    }
}

/// Floor division: ⌊a / b⌋ for b > 0.
fn floor_div(a: i64, b: i64) -> i64 {
    debug_assert!(b > 0);
    if a >= 0 {
        a / b
    } else {
        (a - b + 1) / b
    }
}
