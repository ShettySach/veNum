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
/// 3. If FM would blow up, fall back to conservatively dropping all
///    inequalities mentioning the variable (a relaxation).
///
/// Returns a new `ConstraintSystem` without the eliminated variable.
pub fn project_out(system: &ConstraintSystem, var: &PolyVar) -> ConstraintSystem {
    let mut sys = system.clone();
    sys.canonicalize();

    // Try equality-based elimination first.
    if let Some(idx) = find_equality_with_var(&sys, var) {
        eliminate_via_equality(&mut sys, idx, var);
    } else if let Some(new_ineqs) = fm::fourier_motzkin_eliminate(&sys, var) {
        // FM succeeded — replace the inequality set.
        sys.inequalities = new_ineqs;
    } else {
        // FM would blow up — conservative fallback: drop constraints
        // that mention the variable (relaxation).
        sys.inequalities.retain(|a| a.coefficient_of(var) == 0);
    }

    // Remove the variable from the var list.
    sys.vars.retain(|v| v != var);
    sys.simplify();
    sys
}

// ---------------------------------------------------------------------------
// Equality-based elimination
// ---------------------------------------------------------------------------

/// Find the first equality that mentions `var`.
fn find_equality_with_var(sys: &ConstraintSystem, var: &PolyVar) -> Option<usize> {
    sys.equalities
        .iter()
        .position(|a| a.coefficient_of(var) != 0)
}

/// Solve `equalities[idx]` for `var` and substitute into all other
/// constraints.
fn eliminate_via_equality(sys: &mut ConstraintSystem, idx: usize, var: &PolyVar) {
    let eq = sys.equalities.remove(idx);
    let coeff = eq.coefficient_of(var);
    debug_assert!(coeff != 0);

    // eq: coeff * var + rest = 0
    // => var = -rest / coeff
    // For integer exactness we need coeff to divide all other
    // coefficients cleanly.  When it doesn't, we keep the
    // substitution anyway (this is still sound for feasibility
    // checking, though not exact for counting).
    let rest = eliminate_var_from_aff(&eq, var);
    // replacement: -rest / coeff  (we scale instead of dividing)
    // We substitute var -> (-rest) and then everything is multiplied
    // by |coeff| to stay in integers.  This is equivalent to
    // scaling all constraints by |coeff| first.
    //
    // Simple path: when |coeff| == 1 the substitution is exact.
    let replacement = rest.scale(-1);

    let substitute = |a: &mut Vec<crate::core::poly::domain::Aff>| {
        for expr in a.iter_mut() {
            let c = expr.coefficient_of(var);
            if c == 0 {
                continue;
            }
            if coeff == 1 || coeff == -1 {
                // Exact: var = ±replacement
                *expr = expr.substitute(var, &replacement.scale(1_i64 / coeff));
            } else {
                // Scale the whole expression by |coeff|, then substitute.
                let scaled_expr = expr.scale(coeff.abs());
                let sign = if coeff > 0 { -1 } else { 1 };
                *expr = scaled_expr.substitute(var, &replacement.scale(sign));
            }
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
