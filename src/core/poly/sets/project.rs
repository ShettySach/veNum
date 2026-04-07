use super::relation::ConstraintSystem;
use crate::core::poly::domain::PolyVar;
use crate::core::poly::native::fm;

/// Project (eliminate) a variable from a constraint system.
///
/// Strategy:
///
/// 1. If an equality contains the target variable, solve for it and
///    substitute into all other constraints, then remove the equality.
/// 2. Otherwise apply Fourier-Motzkin elimination over the inequalities.
/// 3. If exact elimination is unavailable, keep the original system.
///
/// Returns a new `ConstraintSystem` without the eliminated variable when
/// exact elimination is available, otherwise returns the original system.
pub fn project_out(system: &ConstraintSystem, var: &PolyVar) -> ConstraintSystem {
    match try_project_out_exact(system, var) {
        Some(sys) => sys,
        None => system.clone(),
    }
}

/// Try to project (eliminate) a variable exactly.
///
/// Returns `None` when this module cannot eliminate `var` without relaxing
/// constraints.
pub fn try_project_out_exact(system: &ConstraintSystem, var: &PolyVar) -> Option<ConstraintSystem> {
    let mut sys = system.clone();
    sys.canonicalize();

    // Try equality-based elimination first.
    if let Some(idx) = find_equality_with_var(&sys, var) {
        eliminate_via_equality(&mut sys, idx, var);
    } else if equality_mentions_var(&sys, var) {
        // A non-unit equality contains `var`; exact elimination would require
        // division in Presburger space, so bail out.
        return None;
    } else if let Some(new_ineqs) = fm::fourier_motzkin_eliminate(&sys, var) {
        // FM succeeded — replace the inequality set.
        sys.inequalities = new_ineqs;
    } else {
        // Unable to eliminate exactly.
        return None;
    }

    // Remove the variable from the var list.
    sys.vars.retain(|v| v != var);
    sys.simplify();
    Some(sys)
}

fn equality_mentions_var(sys: &ConstraintSystem, var: &PolyVar) -> bool {
    sys.equalities.iter().any(|a| a.coefficient_of(var) != 0)
}

// ---------------------------------------------------------------------------
// Equality-based elimination
// ---------------------------------------------------------------------------

/// Find the first equality that mentions `var`.
fn find_equality_with_var(sys: &ConstraintSystem, var: &PolyVar) -> Option<usize> {
    sys.equalities.iter().position(|a| {
        let coeff = a.coefficient_of(var);
        coeff == 1 || coeff == -1
    })
}

/// Solve `equalities[idx]` for `var` and substitute into all other
/// constraints.
fn eliminate_via_equality(sys: &mut ConstraintSystem, idx: usize, var: &PolyVar) {
    let eq = sys.equalities.remove(idx);
    let coeff = eq.coefficient_of(var);
    debug_assert!(coeff == 1 || coeff == -1);

    // eq: coeff * var + rest = 0
    // => var = -rest / coeff
    let rest = eliminate_var_from_aff(&eq, var);
    // replacement: -rest / coeff, exact for coeff in {+1, -1}.
    let replacement = if coeff == 1 { rest.scale(-1) } else { rest };

    let substitute = |a: &mut Vec<crate::core::poly::domain::Aff>| {
        for expr in a.iter_mut() {
            let c = expr.coefficient_of(var);
            if c == 0 {
                continue;
            }
            *expr = expr.substitute(var, &replacement);
            expr.canonicalize();
        }
    };

    substitute(&mut sys.equalities);
    substitute(&mut sys.inequalities);
}

/// Return `aff` with the term for `var` removed.
fn eliminate_var_from_aff(
    aff: &crate::core::poly::domain::Aff,
    var: &PolyVar,
) -> crate::core::poly::domain::Aff {
    crate::core::poly::domain::Aff {
        constant: aff.constant,
        terms: aff
            .terms
            .iter()
            .filter(|(_, v)| v != var)
            .cloned()
            .collect(),
    }
}
