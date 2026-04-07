use crate::core::poly::domain::{IterName, PolyVar};
use crate::core::poly::sets::relation::ConstraintSystem;

/// Per-loop-dimension direction summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Dependence distance is exactly 0 on this dimension.
    Zero,
    /// Dependence flows forward (distance > 0).
    Positive,
    /// Dependence flows backward (distance < 0).
    Negative,
    /// Could not determine — treat as unknown.
    Unknown,
}

/// Compute a conservative distance vector for a dependence relation.
///
/// For each shared iterator, tries to derive the constant distance
/// `t_i - s_i` from the constraint system.  If the distance is not a
/// unique constant, falls back to a direction summary.
///
/// Returns `(distance, directions)`:
/// - `distance`: `Some(vec)` when all dimensions have exact constant
///    distances, `None` otherwise.
/// - `directions`: per-dimension `Direction` summary.
pub fn compute_distance(
    system: &ConstraintSystem,
    source_iters: &[IterName],
    sink_iters: &[IterName],
    source_prefix: &str,
    sink_prefix: &str,
) -> (Option<Vec<i64>>, Vec<Direction>) {
    let shared_iters: Vec<&IterName> = source_iters
        .iter()
        .filter(|i| sink_iters.contains(i))
        .collect();

    let mut distances = Vec::with_capacity(shared_iters.len());
    let mut directions = Vec::with_capacity(shared_iters.len());
    let mut all_exact = true;

    for iter in &shared_iters {
        let s_var = PolyVar::Iter(format!("{source_prefix}{iter}").into());
        let t_var = PolyVar::Iter(format!("{sink_prefix}{iter}").into());

        match try_exact_distance(system, &s_var, &t_var) {
            Some(d) => {
                distances.push(d);
                directions.push(if d == 0 {
                    Direction::Zero
                } else if d > 0 {
                    Direction::Positive
                } else {
                    Direction::Negative
                });
            }
            None => {
                all_exact = false;
                distances.push(0); // placeholder
                directions.push(derive_direction(system, &s_var, &t_var));
            }
        }
    }

    let dist = if all_exact { Some(distances) } else { None };

    (dist, directions)
}

/// Try to derive an exact constant distance `t - s` from the constraint
/// system.
///
/// Looks for an equality of the form `t - s + c = 0` (i.e. `t - s = -c`).
fn try_exact_distance(system: &ConstraintSystem, s_var: &PolyVar, t_var: &PolyVar) -> Option<i64> {
    let sys = system.clone();

    for eq in &sys.equalities {
        let s_coeff = eq.coefficient_of(s_var);
        let t_coeff = eq.coefficient_of(t_var);

        // Looking for: t_coeff * t + s_coeff * s + constant = 0
        // where t_coeff = 1, s_coeff = -1 (or scaled versions)
        if s_coeff == 0 || t_coeff == 0 {
            continue;
        }
        if s_coeff != -t_coeff {
            continue;
        }

        // Check that no other variables are involved.
        let other_terms: i64 = eq
            .terms
            .iter()
            .filter(|(_, v)| v != s_var && v != t_var)
            .map(|(c, _)| c.abs())
            .sum();
        if other_terms != 0 {
            continue;
        }

        // t_coeff * t - t_coeff * s + constant = 0
        // => t - s = -constant / t_coeff
        if eq.constant % t_coeff == 0 {
            return Some(-eq.constant / t_coeff);
        }
    }

    None
}

/// Derive a conservative direction for `t - s` from inequalities.
fn derive_direction(system: &ConstraintSystem, s_var: &PolyVar, t_var: &PolyVar) -> Direction {
    // Introduce a delta variable: delta = t - s.
    // We build a system with delta replacing t, then project out s and t
    // to get bounds on delta.
    //
    // Simpler approach: check if the system implies delta >= 0 or
    // delta <= 0 by looking at the execution-order constraint.

    // Check if there's an inequality t - s >= 0 (positive direction).
    let has_forward = system.inequalities.iter().any(|ineq| {
        let s_c = ineq.coefficient_of(s_var);
        let t_c = ineq.coefficient_of(t_var);
        let other: i64 = ineq
            .terms
            .iter()
            .filter(|(_, v)| v != s_var && v != t_var)
            .map(|(c, _)| c.abs())
            .sum();
        // t - s + const >= 0 with const >= 0 and no other vars
        t_c > 0 && s_c == -t_c && other == 0 && ineq.constant >= 0
    });

    if has_forward {
        // The system constrains t >= s, so direction is non-negative.
        // But we can't distinguish Zero from Positive without more work.
        // Conservative: report Unknown (could be 0 or positive).
        return Direction::Unknown;
    }

    Direction::Unknown
}
