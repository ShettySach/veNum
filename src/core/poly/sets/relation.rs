use crate::core::hlir::Symbol;
use crate::core::poly::access_map::AccessMap;
use crate::core::poly::domain::{Aff, Constraint, Domain, PolyVar};

/// A constraint system: separated equalities and inequalities over affine
/// expressions.  This is the internal representation used by the relation
/// engine and projection algorithms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintSystem {
    /// All named variables (iterators + parameters) in the system.
    pub vars: Vec<PolyVar>,
    /// Equality constraints: `expr = 0`.
    pub equalities: Vec<Aff>,
    /// Inequality constraints: `expr >= 0`.
    pub inequalities: Vec<Aff>,
}

impl ConstraintSystem {
    pub fn new() -> Self {
        Self {
            vars: Vec::new(),
            equalities: Vec::new(),
            inequalities: Vec::new(),
        }
    }

    /// Add a variable if not already present.
    pub fn add_var(&mut self, var: PolyVar) {
        if !self.vars.contains(&var) {
            self.vars.push(var);
        }
    }

    /// Add an equality constraint `expr = 0`.
    pub fn add_equality(&mut self, aff: Aff) {
        self.equalities.push(aff);
    }

    /// Add an inequality constraint `expr >= 0`.
    pub fn add_inequality(&mut self, aff: Aff) {
        self.inequalities.push(aff);
    }

    /// Import all constraints from a `Domain`, registering its iterators and
    /// parameters as variables.
    pub fn add_domain(&mut self, domain: &Domain) {
        for iter in &domain.iters {
            self.add_var(PolyVar::Iter(iter.clone()));
        }
        for param in &domain.params {
            self.add_var(PolyVar::Param(*param));
        }
        for c in &domain.constraints {
            match c {
                Constraint::Eq(a) => self.equalities.push(a.clone()),
                Constraint::Ineq(a) => self.inequalities.push(a.clone()),
            }
        }
    }

    /// Import all constraints from a `Domain`, renaming its iterators with the
    /// given `prefix` to distinguish source from sink variables.
    pub fn add_domain_prefixed(&mut self, domain: &Domain, prefix: &str) {
        for iter in &domain.iters {
            let renamed = format!("{prefix}{iter}");
            self.add_var(PolyVar::Iter(renamed.clone()));
        }
        for param in &domain.params {
            self.add_var(PolyVar::Param(*param));
        }
        for c in &domain.constraints {
            let renamed = match c {
                Constraint::Eq(a) => rename_iters(a, &domain.iters, prefix),
                Constraint::Ineq(a) => rename_iters(a, &domain.iters, prefix),
            };
            match c {
                Constraint::Eq(_) => self.equalities.push(renamed),
                Constraint::Ineq(_) => self.inequalities.push(renamed),
            }
        }
    }

    /// Total number of constraints (equalities + inequalities).
    pub fn num_constraints(&self) -> usize {
        self.equalities.len() + self.inequalities.len()
    }

    /// Returns true if the system has no constraints at all.
    pub fn is_unconstrained(&self) -> bool {
        self.equalities.is_empty() && self.inequalities.is_empty()
    }

    /// Canonicalize all constraints in-place.
    pub fn canonicalize(&mut self) {
        for a in &mut self.equalities {
            a.canonicalize();
        }
        for a in &mut self.inequalities {
            a.canonicalize();
        }
    }

    /// Check whether this constraint system is feasible (has an integer
    /// solution).
    pub fn is_feasible(&self) -> crate::core::poly::native::feasibility::Feasibility {
        crate::core::poly::native::feasibility::check_feasibility(self)
    }

    /// Check whether this constraint system has no integer solutions.
    ///
    /// Returns `true` when provably empty, `false` when feasible or
    /// unknown (conservative: may return `false` for actually-empty
    /// systems that the solver cannot prove empty).
    pub fn is_empty(&self) -> bool {
        self.is_feasible() == crate::core::poly::native::feasibility::Feasibility::Infeasible
    }

    /// Remove duplicate and trivially-true constraints.
    pub fn simplify(&mut self) {
        self.canonicalize();
        self.equalities.dedup();
        self.inequalities.dedup();
        // Drop trivially true inequalities: constant >= 0 with no vars.
        self.inequalities
            .retain(|a| !(a.terms.is_empty() && a.constant >= 0));
    }
}

// ---------------------------------------------------------------------------
// Relation: a pair of (source, sink) iteration spaces connected by
// memory-equality and execution-order constraints.
// ---------------------------------------------------------------------------

/// A dependence relation between a source statement instance and a sink
/// statement instance.
///
/// Internally, source iterators are prefixed with `s_` and sink iterators
/// with `t_`, so they live in the same constraint system without collision.
#[derive(Clone, Debug)]
pub struct Relation {
    /// Source iterator names (un-prefixed).
    pub source_iters: Vec<String>,
    /// Sink iterator names (un-prefixed).
    pub sink_iters: Vec<String>,
    /// Shared parameters.
    pub params: Vec<Symbol>,
    /// The combined constraint system (source domain + sink domain +
    /// memory-equality + execution-order).
    pub system: ConstraintSystem,
}

impl Relation {
    pub const SOURCE_PREFIX: &'static str = "s_";
    pub const SINK_PREFIX: &'static str = "t_";

    /// Build a dependence relation from:
    ///
    /// * `source_domain` — iteration domain of the writing statement
    /// * `sink_domain` — iteration domain of the reading statement
    /// * `write_access` — access map of the write
    /// * `read_access` — access map of the read
    ///
    /// The relation encodes:
    /// 1. source domain constraints (with `s_` prefix)
    /// 2. sink domain constraints (with `t_` prefix)
    /// 3. memory-equality: `write_map[k] - read_map[k] = 0` for each dim k
    /// 4. execution-order: source lexicographically before-or-equal sink
    ///    (encoded conservatively as `t_i - s_i >= 0` for the first shared
    ///    iterator, which is correct for sequential loops in original order)
    pub fn build_dependence(
        source_domain: &Domain,
        sink_domain: &Domain,
        write_access: &AccessMap,
        read_access: &AccessMap,
    ) -> Self {
        let mut system = ConstraintSystem::new();

        // 1. Source domain constraints with s_ prefix.
        system.add_domain_prefixed(source_domain, Self::SOURCE_PREFIX);

        // 2. Sink domain constraints with t_ prefix.
        system.add_domain_prefixed(sink_domain, Self::SINK_PREFIX);

        // 3. Memory-equality constraints.
        let n_dims = write_access.mapping.len().min(read_access.mapping.len());
        for k in 0..n_dims {
            let w = rename_iters(
                &write_access.mapping[k],
                &source_domain.iters,
                Self::SOURCE_PREFIX,
            );
            let r = rename_iters(
                &read_access.mapping[k],
                &sink_domain.iters,
                Self::SINK_PREFIX,
            );
            // w - r = 0
            system.add_equality(w.sub(&r));
        }

        // 4. Execution-order: for the original sequential schedule the
        //    source instance executes before the sink if the source
        //    iteration vector is lexicographically <= the sink vector.
        //    A simple conservative encoding: for the outermost shared
        //    iterator, require t_i - s_i >= 0.
        if let Some(first_iter) = source_domain.iters.first()
            && sink_domain.iters.contains(first_iter)
        {
            let s_var = Aff::iter_var(format!("{}{first_iter}", Self::SOURCE_PREFIX));
            let t_var = Aff::iter_var(format!("{}{first_iter}", Self::SINK_PREFIX));
            system.add_inequality(t_var.sub(&s_var));
        }

        // Collect params.
        let mut params = source_domain.params.clone();
        for p in &sink_domain.params {
            if !params.contains(p) {
                params.push(*p);
            }
        }

        Relation {
            source_iters: source_domain.iters.clone(),
            sink_iters: sink_domain.iters.clone(),
            params,
            system,
        }
    }

    /// Prefixed source iterator name.
    pub fn source_var(&self, iter: &str) -> PolyVar {
        PolyVar::Iter(format!("{}{iter}", Self::SOURCE_PREFIX))
    }

    /// Prefixed sink iterator name.
    pub fn sink_var(&self, iter: &str) -> PolyVar {
        PolyVar::Iter(format!("{}{iter}", Self::SINK_PREFIX))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rename iterator variables in an `Aff` by adding a prefix.
fn rename_iters(aff: &Aff, domain_iters: &[String], prefix: &str) -> Aff {
    Aff {
        constant: aff.constant,
        terms: aff
            .terms
            .iter()
            .map(|(c, v)| {
                let renamed = match v {
                    PolyVar::Iter(name) if domain_iters.contains(name) => {
                        PolyVar::Iter(format!("{prefix}{name}"))
                    }
                    other => other.clone(),
                };
                (*c, renamed)
            })
            .collect(),
    }
}
