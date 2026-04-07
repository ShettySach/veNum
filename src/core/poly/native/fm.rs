use crate::core::poly::domain::{Aff, PolyVar};
use crate::core::poly::sets::ConstraintSystem;

/// Maximum number of new inequalities FM may produce before bailing out.
/// This prevents blow-up on large systems.
const FM_MAX_GENERATED: usize = 256;

/// Fourier-Motzkin elimination of `var` from the inequality part of a
/// constraint system.
///
/// Precondition: all equalities involving `var` have already been
/// eliminated (via equality substitution in `project_out`).
///
/// The algorithm:
/// 1. Partition inequalities into lower bounds on `var`, upper bounds,
///    and constraints that do not mention `var`.
/// 2. For each (lower, upper) pair, combine them to produce a new
///    inequality that does not mention `var`.
/// 3. Return the unrelated constraints plus the new combined ones.
///
/// Returns `None` if the number of generated constraints would exceed
/// the safety limit (caller should fall back to conservative handling).
pub fn fourier_motzkin_eliminate(
    system: &ConstraintSystem,
    var: &PolyVar,
) -> Option<Vec<Aff>> {
    let mut lower_bounds: Vec<Aff> = Vec::new(); // coeff > 0: var has a lower bound
    let mut upper_bounds: Vec<Aff> = Vec::new(); // coeff < 0: var has an upper bound
    let mut unrelated: Vec<Aff> = Vec::new();

    for ineq in &system.inequalities {
        let c = ineq.coefficient_of(var);
        if c == 0 {
            unrelated.push(ineq.clone());
        } else if c > 0 {
            lower_bounds.push(ineq.clone());
        } else {
            upper_bounds.push(ineq.clone());
        }
    }

    // Check blow-up: FM produces |lower| * |upper| new constraints.
    let product = lower_bounds.len() * upper_bounds.len();
    if product > FM_MAX_GENERATED {
        return None;
    }

    let mut result = unrelated;

    for lb in &lower_bounds {
        for ub in &upper_bounds {
            let combined = combine_bounds(lb, ub, var);
            result.push(combined);
        }
    }

    // Light redundancy cleanup.
    cleanup(&mut result);

    Some(result)
}

/// Combine a lower-bound inequality and an upper-bound inequality to
/// eliminate `var`.
///
/// Given:
///   lb:  a * var + rest_lb >= 0   (a > 0)
///   ub:  b * var + rest_ub >= 0   (b < 0)
///
/// Multiply lb by |b| and ub by a, then add:
///   |b| * rest_lb + a * rest_ub >= 0
///
/// This eliminates `var` from the combined constraint.
fn combine_bounds(lb: &Aff, ub: &Aff, var: &PolyVar) -> Aff {
    let a = lb.coefficient_of(var); // > 0
    let b = ub.coefficient_of(var); // < 0

    debug_assert!(a > 0);
    debug_assert!(b < 0);

    let abs_b = -b;

    // Scale lb by |b|, ub by a, then add.
    let scaled_lb = lb.scale(abs_b);
    let scaled_ub = ub.scale(a);
    let mut combined = scaled_lb.add(&scaled_ub);
    // The var terms cancel: a*|b| + |b|*b = a*|b| - |b|*a = 0.
    combined.canonicalize();
    combined
}

/// Light redundancy cleanup on a list of inequalities.
fn cleanup(ineqs: &mut Vec<Aff>) {
    // Canonicalize all.
    for a in ineqs.iter_mut() {
        a.canonicalize();
    }

    // Deduplicate.
    ineqs.sort_by(|a, b| {
        a.constant
            .cmp(&b.constant)
            .then_with(|| a.terms.cmp(&b.terms))
    });
    ineqs.dedup();

    // Drop trivially true: constant >= 0 with no variables.
    ineqs.retain(|a| !(a.terms.is_empty() && a.constant >= 0));
}

/// Check whether a constraint system is trivially infeasible.
///
/// Returns `true` if any constraint is a contradiction:
/// - an equality `c = 0` where c != 0 and no variables
/// - an inequality `c >= 0` where c < 0 and no variables
pub fn is_trivially_infeasible(system: &ConstraintSystem) -> bool {
    for eq in &system.equalities {
        if eq.terms.is_empty() && eq.constant != 0 {
            return true;
        }
    }
    for ineq in &system.inequalities {
        if ineq.terms.is_empty() && ineq.constant < 0 {
            return true;
        }
    }
    false
}
