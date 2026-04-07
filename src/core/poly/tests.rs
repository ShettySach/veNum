#[cfg(test)]
mod poly_tests {
    use anyhow::Result;

    use crate::core::hlir::{BufferId, DType, Dim, HLIRGraph, Op, Symbol, TensorType};
    use crate::core::llir::affine::{AffineExpr, Var};
    use crate::core::lower::lower;
    use crate::core::poly::analysis::dependence::analyze_kernel_poly;
    use crate::core::poly::native::feasibility::{check_feasibility, Feasibility};
    use crate::core::poly::native::fm;
    use crate::core::poly::{
        dim_to_aff, extract_instances, project_out, shape_to_domain, shape_to_domain_checked,
        strides_to_access, Aff, ConstraintSystem, NativeDependenceAnalyzer, PolyVar, Relation,
    };
    use crate::core::schedule::search::{ScheduleSearcher, TrivialHardware};
    use crate::core::traits::{DependenceAnalyzer, ScheduleTransform};

    // -----------------------------------------------------------------------
    // Domain construction
    // -----------------------------------------------------------------------

    #[test]
    fn shape_to_domain_emits_two_constraints_per_dim() {
        let domain = shape_to_domain(&[Dim::Const(8), Dim::Const(16)]);
        assert_eq!(domain.iters, vec!["i0", "i1"]);
        assert_eq!(domain.constraints.len(), 4);
    }

    #[test]
    fn shape_to_domain_handles_symbolic_dim() {
        let sym = Symbol(0);
        let domain = shape_to_domain(&[Dim::Sym(sym)]);
        assert_eq!(domain.params, vec![sym]);
        assert_eq!(domain.constraints.len(), 2);
    }

    #[test]
    fn shape_to_domain_handles_affine_add_dim() {
        // Dim::Add(Const(4), Sym(1))  →  bound = 4 + N
        let dim = Dim::Add(Box::new(Dim::Const(4)), Box::new(Dim::Sym(Symbol(1))));
        let domain = shape_to_domain(&[dim]);
        assert_eq!(domain.params, vec![Symbol(1)]);
        assert_eq!(domain.constraints.len(), 2);
    }

    #[test]
    fn shape_to_domain_rejects_non_affine_dim() {
        let dim = Dim::Div(Box::new(Dim::Const(8)), Box::new(Dim::Const(2)));
        let err = shape_to_domain_checked(&[dim]).unwrap_err();
        assert!(err.contains("non-affine dimension"));
    }

    // -----------------------------------------------------------------------
    // Aff canonicalization
    // -----------------------------------------------------------------------

    #[test]
    fn aff_canonicalize_combines_like_terms() {
        // 2*i + 3*i → 5*i
        let mut aff = Aff {
            constant: 0,
            terms: vec![
                (2, PolyVar::Iter("i".into())),
                (3, PolyVar::Iter("i".into())),
            ],
        };
        aff.canonicalize();
        assert_eq!(aff.terms.len(), 1);
        assert_eq!(aff.terms[0].0, 5);
    }

    #[test]
    fn aff_canonicalize_drops_zeros() {
        // x - x + 3 → 3
        let a = Aff::iter_var("x");
        let minus_a = a.scale(-1);
        let mut expr = a.add(&minus_a).add(&Aff::constant(3));
        expr.canonicalize();
        assert!(expr.terms.is_empty());
        assert_eq!(expr.constant, 3);
    }

    #[test]
    fn aff_canonicalize_sorts_deterministically() {
        let mut aff = Aff {
            constant: 0,
            terms: vec![
                (1, PolyVar::Iter("j".into())),
                (1, PolyVar::Iter("i".into())),
                (1, PolyVar::Param(Symbol(0))),
            ],
        };
        aff.canonicalize();
        // Ord: Iter < Param; within Iter, lexicographic.
        assert_eq!(aff.terms[0].1, PolyVar::Iter("i".into()));
        assert_eq!(aff.terms[1].1, PolyVar::Iter("j".into()));
        assert_eq!(aff.terms[2].1, PolyVar::Param(Symbol(0)));
    }

    // -----------------------------------------------------------------------
    // Aff substitution
    // -----------------------------------------------------------------------

    #[test]
    fn aff_substitute_replaces_variable() {
        // 2*i + 3  →  substitute i = j + 1  →  2*j + 5
        let expr = Aff {
            constant: 3,
            terms: vec![(2, PolyVar::Iter("i".into()))],
        };
        let replacement = Aff {
            constant: 1,
            terms: vec![(1, PolyVar::Iter("j".into()))],
        };
        let result = expr
            .substitute(&PolyVar::Iter("i".into()), &replacement)
            .canonicalized();
        assert_eq!(result.constant, 5);
        assert_eq!(result.terms, vec![(2, PolyVar::Iter("j".into()))]);
    }

    #[test]
    fn aff_substitute_noop_for_absent_var() {
        let expr = Aff::constant(7);
        let replacement = Aff::iter_var("x");
        let result = expr.substitute(&PolyVar::Iter("y".into()), &replacement);
        assert_eq!(result, expr);
    }

    // -----------------------------------------------------------------------
    // Aff coefficient_of
    // -----------------------------------------------------------------------

    #[test]
    fn aff_coefficient_of_returns_correct_value() {
        let aff = Aff {
            constant: 0,
            terms: vec![
                (3, PolyVar::Iter("i".into())),
                (5, PolyVar::Iter("j".into())),
            ],
        };
        assert_eq!(aff.coefficient_of(&PolyVar::Iter("i".into())), 3);
        assert_eq!(aff.coefficient_of(&PolyVar::Iter("j".into())), 5);
        assert_eq!(aff.coefficient_of(&PolyVar::Iter("k".into())), 0);
    }

    // -----------------------------------------------------------------------
    // Dim → Aff conversion
    // -----------------------------------------------------------------------

    #[test]
    fn dim_to_aff_const() {
        let aff = dim_to_aff(&Dim::Const(42)).unwrap();
        assert_eq!(aff.constant, 42);
        assert!(aff.terms.is_empty());
    }

    #[test]
    fn dim_to_aff_symbolic() {
        let aff = dim_to_aff(&Dim::Sym(Symbol(3))).unwrap();
        assert_eq!(aff.constant, 0);
        assert_eq!(aff.terms, vec![(1, PolyVar::Param(Symbol(3)))]);
    }

    #[test]
    fn dim_to_aff_add() {
        let dim = Dim::Add(Box::new(Dim::Const(5)), Box::new(Dim::Sym(Symbol(0))));
        let aff = dim_to_aff(&dim).unwrap();
        assert_eq!(aff.constant, 5);
        assert_eq!(aff.coefficient_of(&PolyVar::Param(Symbol(0))), 1);
    }

    #[test]
    fn dim_to_aff_const_mul() {
        // 3 * N
        let dim = Dim::Mul(Box::new(Dim::Const(3)), Box::new(Dim::Sym(Symbol(0))));
        let aff = dim_to_aff(&dim).unwrap();
        assert_eq!(aff.coefficient_of(&PolyVar::Param(Symbol(0))), 3);
    }

    #[test]
    fn dim_to_aff_rejects_div() {
        let dim = Dim::Div(Box::new(Dim::Const(4)), Box::new(Dim::Const(2)));
        assert!(dim_to_aff(&dim).is_none());
    }

    #[test]
    fn dim_to_aff_rejects_sym_times_sym() {
        let dim = Dim::Mul(Box::new(Dim::Sym(Symbol(0))), Box::new(Dim::Sym(Symbol(1))));
        assert!(dim_to_aff(&dim).is_none());
    }

    // -----------------------------------------------------------------------
    // AffineExpr ↔ Aff conversions
    // -----------------------------------------------------------------------

    #[test]
    fn affine_expr_to_aff_roundtrip() {
        let expr = AffineExpr::constant(5)
            .with_term(2, Var::Loop("i".into()))
            .with_term(1, Var::Param(Symbol(0)));
        let aff: Aff = (&expr).into();
        assert_eq!(aff.constant, 5);
        assert_eq!(aff.coefficient_of(&PolyVar::Iter("i".into())), 2);
        assert_eq!(aff.coefficient_of(&PolyVar::Param(Symbol(0))), 1);

        let back: AffineExpr = (&aff).into();
        assert_eq!(back.constant, 5);
        assert_eq!(back.coefficient_of(&Var::Loop("i".into())), 2);
        assert_eq!(back.coefficient_of(&Var::Param(Symbol(0))), 1);
    }

    // -----------------------------------------------------------------------
    // AffineExpr operations
    // -----------------------------------------------------------------------

    #[test]
    fn affine_expr_canonicalize_combines_like_terms() {
        let mut expr = AffineExpr::constant(0)
            .with_term(2, Var::Loop("i".into()))
            .with_term(3, Var::Loop("i".into()));
        expr.canonicalize();
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(expr.terms[0].0, 5);
    }

    #[test]
    fn affine_expr_substitute() {
        // 2*i + 3 → subst i = j+1 → 2*j + 5
        let expr = AffineExpr::constant(3).with_term(2, Var::Loop("i".into()));
        let repl = AffineExpr::constant(1).with_term(1, Var::Loop("j".into()));
        let result = expr
            .substitute(&Var::Loop("i".into()), &repl)
            .canonicalized();
        assert_eq!(result.constant, 5);
        assert_eq!(result.terms, vec![(2, Var::Loop("j".into()))]);
    }

    // -----------------------------------------------------------------------
    // Access map
    // -----------------------------------------------------------------------

    #[test]
    fn strides_to_access_rejects_symbolic_stride() {
        let err = strides_to_access(&[Dim::Sym(0_u32.into())], &["i0".to_owned()]).unwrap_err();
        assert!(err.to_string().contains("symbolic stride"));
    }

    // -----------------------------------------------------------------------
    // Native analyzer (existing)
    // -----------------------------------------------------------------------

    #[test]
    fn access_dependence_emits_real_constraints_for_same_cell() -> Result<()> {
        use crate::core::llir::loop_nest::{Loop, LoopAnnotations, LoopKind};
        use crate::core::llir::memory::AccessKind;

        let loops = vec![Loop {
            var: "i0".into(),
            lower: AffineExpr::constant(0),
            upper: AffineExpr::constant(8),
            step: 1,
            kind: LoopKind::Sequential,
            annotations: LoopAnnotations::default(),
        }];

        let write = crate::core::llir::MemoryAccess {
            buffer: BufferId(0),
            indices: vec![AffineExpr::constant(0).with_term(1, Var::Loop("i0".into()))],
            access_kind: AccessKind::Write,
        };
        let read = crate::core::llir::MemoryAccess {
            buffer: BufferId(0),
            indices: vec![AffineExpr::constant(0).with_term(1, Var::Loop("i0".into()))],
            access_kind: AccessKind::Read,
        };

        let analyzer = NativeDependenceAnalyzer;
        let rel = analyzer
            .access_dependence(&write, &read, &loops)?
            .expect("expected dependence relation");

        assert!(
            !rel.constraints.is_empty(),
            "relation should not be placeholder-empty"
        );
        assert!(
            rel.constraints
                .iter()
                .any(|c| c.kind == crate::core::llir::ConstraintKind::Eq),
            "relation should include memory equality"
        );
        assert!(
            rel.constraints
                .iter()
                .any(|c| c.kind == crate::core::llir::ConstraintKind::Ge),
            "relation should include ordering/domain inequalities"
        );
        Ok(())
    }

    #[test]
    fn access_dependence_returns_none_for_reverse_time_shift() -> Result<()> {
        use crate::core::llir::loop_nest::{Loop, LoopAnnotations, LoopKind};
        use crate::core::llir::memory::AccessKind;

        let loops = vec![Loop {
            var: "i0".into(),
            lower: AffineExpr::constant(0),
            upper: AffineExpr::constant(8),
            step: 1,
            kind: LoopKind::Sequential,
            annotations: LoopAnnotations::default(),
        }];

        // write A[i0], read A[i0 + 1] => requires sink before source for same cell.
        let write = crate::core::llir::MemoryAccess {
            buffer: BufferId(0),
            indices: vec![AffineExpr::constant(0).with_term(1, Var::Loop("i0".into()))],
            access_kind: AccessKind::Write,
        };
        let read = crate::core::llir::MemoryAccess {
            buffer: BufferId(0),
            indices: vec![AffineExpr::constant(1).with_term(1, Var::Loop("i0".into()))],
            access_kind: AccessKind::Read,
        };

        let analyzer = NativeDependenceAnalyzer;
        let rel = analyzer.access_dependence(&write, &read, &loops)?;
        assert!(rel.is_none());
        Ok(())
    }

    #[test]
    fn native_analyzer_finds_simple_dependence() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let deps = analyzer.analyze_kernel(llir.kernels.last().expect("kernel must exist"))?;
        assert!(!deps.is_empty());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Step 2: LLIR preserves affine bounds
    // -----------------------------------------------------------------------

    #[test]
    fn lowered_const_dim_produces_correct_loop_bound() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(64)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let kernel = llir.kernels.last().unwrap();
        assert_eq!(kernel.loop_nest.loops[0].upper.as_const_value(), Some(64));
        Ok(())
    }

    #[test]
    fn lowered_reduce_gets_input_dim_as_bound() -> Result<()> {
        use crate::core::hlir::ReduceOp;
        let mut g = HLIRGraph::new();
        // Input: [4, 8], reduce axis 1 → output [4]
        let ty = TensorType::contiguous(vec![Dim::Const(4), Dim::Const(8)], DType::F32);
        let a = g.load(BufferId(0), ty);
        let _r = g.reduce(a, vec![1], ReduceOp::Sum, false);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let kernel = llir.kernels.last().unwrap();

        // Output loop: i0 bound = 4
        assert_eq!(kernel.loop_nest.loops[0].upper.as_const_value(), Some(4));
        // Reduction loop: r0 bound = 8 (from input axis 1)
        let reduce_loop = kernel
            .loop_nest
            .loops
            .iter()
            .find(|l| l.var.starts_with('r'))
            .expect("should have a reduce loop");
        assert_eq!(reduce_loop.upper.as_const_value(), Some(8));
        Ok(())
    }

    #[test]
    fn lowered_symbolic_dim_produces_param_bound() -> Result<()> {
        let mut g = HLIRGraph::new();
        let sym = Symbol(42);
        let ty = TensorType::contiguous(vec![Dim::Sym(sym)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let kernel = llir.kernels.last().unwrap();

        // Bound should be symbolic (not constant 1).
        let ub = &kernel.loop_nest.loops[0].upper;
        assert!(
            ub.as_const_value().is_none(),
            "symbolic dim should not collapse to constant"
        );
        assert_eq!(ub.coefficient_of(&Var::Param(sym)), 1);
        Ok(())
    }

    #[test]
    fn native_legality_rejects_parallel_positive_distance() -> Result<()> {
        let dep = crate::core::llir::Dependence {
            from: 0,
            to: 1,
            kind: crate::core::llir::DepKind::Raw,
            distance: Some(vec![1]),
            relation: crate::core::llir::DependenceRelation {
                source_vars: vec!["i0".to_owned()],
                sink_vars: vec!["i0".to_owned()],
                constraints: vec![],
            },
        };

        // Build a minimal kernel for the legality check.
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);
        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let kernel = llir.kernels.last().unwrap();

        let legal = analyzer.check_legality(
            &[dep],
            &ScheduleTransform::Parallelize {
                loop_var: "i0".to_owned(),
            },
            kernel,
        )?;
        assert!(!legal);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Step 3: Polyhedral extraction from LLIR
    // -----------------------------------------------------------------------

    #[test]
    fn extract_elementwise_kernel_produces_one_statement() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4), Dim::Const(8)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let kernel = llir.kernels.last().unwrap();

        let instances = extract_instances(kernel);
        assert_eq!(
            instances.len(),
            1,
            "elementwise add should produce one statement"
        );

        let si = &instances[0];
        assert_eq!(si.stmt_id, 0);
        // Domain should have two iterators (i0, i1).
        assert_eq!(si.domain.iters.len(), 2);
        // Two loads (a, b) → two reads; one store → one write.
        assert_eq!(si.reads.len(), 2);
        assert_eq!(si.writes.len(), 1);
        Ok(())
    }

    #[test]
    fn extract_reduction_kernel_has_accumulator_read_write() -> Result<()> {
        use crate::core::hlir::ReduceOp;
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4), Dim::Const(8)], DType::F32);
        let a = g.load(BufferId(0), ty);
        let _r = g.reduce(a, vec![1], ReduceOp::Sum, false);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let kernel = llir.kernels.last().unwrap();

        let instances = extract_instances(kernel);
        assert!(
            !instances.is_empty(),
            "reduction kernel should produce statements"
        );

        // At least one statement should have both reads and writes.
        let has_rw = instances
            .iter()
            .any(|si| !si.reads.is_empty() && !si.writes.is_empty());
        assert!(has_rw, "reduction should have read+write accesses");
        Ok(())
    }

    #[test]
    fn extract_preserves_domain_iterators_from_loop_nest() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(16)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let kernel = llir.kernels.last().unwrap();

        let instances = extract_instances(kernel);
        let si = &instances[0];

        // Domain iterators should match the loop variables.
        let loop_vars: Vec<&str> = kernel
            .loop_nest
            .loops
            .iter()
            .map(|l| l.var.as_str())
            .collect();
        let domain_iters: Vec<&str> = si.domain.iters.iter().map(|s| s.as_str()).collect();
        assert_eq!(domain_iters, loop_vars);

        // Each iterator should have lower and upper bound constraints.
        assert!(si.domain.constraints.len() >= 2);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Step 4: Relation and set operations
    // -----------------------------------------------------------------------

    #[test]
    fn constraint_system_add_domain_registers_vars_and_constraints() {
        let domain = shape_to_domain(&[Dim::Const(8), Dim::Const(16)]);
        let mut sys = ConstraintSystem::new();
        sys.add_domain(&domain);

        // Should have two iter vars.
        assert_eq!(sys.vars.len(), 2);
        // 4 constraints: 2 lower + 2 upper bounds.
        assert_eq!(sys.num_constraints(), 4);
        assert!(sys.equalities.is_empty());
        assert_eq!(sys.inequalities.len(), 4);
    }

    #[test]
    fn constraint_system_prefixed_domain_renames_iters() {
        let domain = shape_to_domain(&[Dim::Const(4)]);
        let mut sys = ConstraintSystem::new();
        sys.add_domain_prefixed(&domain, "s_");

        assert!(sys.vars.contains(&PolyVar::Iter("s_i0".into())));
        assert!(!sys.vars.contains(&PolyVar::Iter("i0".into())));
    }

    #[test]
    fn constraint_system_simplify_removes_trivially_true() {
        let mut sys = ConstraintSystem::new();
        // 5 >= 0 is trivially true.
        sys.add_inequality(Aff::constant(5));
        // -1 >= 0 is false but not trivially true — kept.
        sys.add_inequality(Aff::constant(-1));
        sys.simplify();

        assert_eq!(sys.inequalities.len(), 1);
        assert_eq!(sys.inequalities[0].constant, -1);
    }

    #[test]
    fn relation_build_dependence_same_buffer() {
        use crate::core::poly::AccessMap;

        // S[i] -> A[i]  (write)
        // T[i] -> A[i]  (read)
        let domain = shape_to_domain(&[Dim::Const(8)]);
        let access = AccessMap {
            buffer: BufferId(0),
            domain_iters: vec!["i0".into()],
            mapping: vec![Aff::iter_var("i0")],
        };

        let rel = Relation::build_dependence(&domain, &domain, &access, &access);

        // Source and sink iterators.
        assert_eq!(rel.source_iters, vec!["i0"]);
        assert_eq!(rel.sink_iters, vec!["i0"]);

        // Should have: 2 source bounds + 2 sink bounds = 4 inequalities,
        // 1 memory-equality, 1 execution-order inequality.
        assert!(
            !rel.system.equalities.is_empty(),
            "should have memory-equality"
        );
        assert!(
            rel.system.inequalities.len() >= 5,
            "should have domain bounds + execution order"
        );
    }

    #[test]
    fn relation_build_dependence_2d_access() {
        use crate::core::poly::AccessMap;

        // S[i, j] -> A[i, j]
        let domain = shape_to_domain(&[Dim::Const(4), Dim::Const(8)]);
        let access = AccessMap {
            buffer: BufferId(0),
            domain_iters: vec!["i0".into(), "i1".into()],
            mapping: vec![Aff::iter_var("i0"), Aff::iter_var("i1")],
        };

        let rel = Relation::build_dependence(&domain, &domain, &access, &access);

        // Two memory-equality constraints (one per dimension).
        assert_eq!(rel.system.equalities.len(), 2);
    }

    #[test]
    fn relation_order_slice_enforces_prefix_equalities() {
        use crate::core::poly::AccessMap;

        let domain = shape_to_domain(&[Dim::Const(4), Dim::Const(8)]);
        let access = AccessMap {
            buffer: BufferId(0),
            domain_iters: vec!["i0".into(), "i1".into()],
            mapping: vec![Aff::iter_var("i0"), Aff::iter_var("i1")],
        };

        let rel = Relation::build_dependence_with_order_dim(&domain, &domain, &access, &access, 1);

        // order_dim=1 adds equality on dim 0 and inequality on dim 1.
        assert!(rel.system.equalities.len() >= 3);
    }

    #[test]
    fn project_out_via_equality_removes_variable() {
        // System: x + y = 0, x >= 0, x <= 3
        // Project out x → y = 0 after substitution x = -y
        // remaining: -y >= 0, 3 - (-y) >= 0 → y <= 0, y >= -3
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_var(PolyVar::Iter("y".into()));
        sys.add_equality(Aff {
            constant: 0,
            terms: vec![
                (1, PolyVar::Iter("x".into())),
                (1, PolyVar::Iter("y".into())),
            ],
        });
        // x >= 0
        sys.add_inequality(Aff::iter_var("x"));
        // 3 - x >= 0
        sys.add_inequality(Aff {
            constant: 3,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });

        let result = project_out(&sys, &PolyVar::Iter("x".into()));

        // x should be gone.
        assert!(!result.vars.contains(&PolyVar::Iter("x".into())));
        // Equality is consumed by substitution.
        assert!(result.equalities.is_empty());
        // Two inequalities remain (in terms of y).
        assert_eq!(result.inequalities.len(), 2);
    }

    #[test]
    fn project_out_without_equality_drops_ineqs() {
        // System: x >= 0, x <= 5, y >= 0
        // Project out x without equality → x-constraints dropped.
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_var(PolyVar::Iter("y".into()));
        sys.add_inequality(Aff::iter_var("x"));
        sys.add_inequality(Aff {
            constant: 5,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });
        sys.add_inequality(Aff::iter_var("y"));

        let result = project_out(&sys, &PolyVar::Iter("x".into()));

        assert!(!result.vars.contains(&PolyVar::Iter("x".into())));
        // Only y >= 0 remains.
        assert_eq!(result.inequalities.len(), 1);
    }

    #[test]
    fn project_out_non_unit_equality_does_not_drop_var() {
        // 2*x + y = 0 cannot be eliminated exactly in integer space.
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_var(PolyVar::Iter("y".into()));
        sys.add_equality(Aff {
            constant: 0,
            terms: vec![
                (2, PolyVar::Iter("x".into())),
                (1, PolyVar::Iter("y".into())),
            ],
        });
        sys.add_inequality(Aff::iter_var("x"));

        let result = project_out(&sys, &PolyVar::Iter("x".into()));
        assert!(result.vars.contains(&PolyVar::Iter("x".into())));
        assert_eq!(result.equalities.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Step 5: Native Fourier-Motzkin projection
    // -----------------------------------------------------------------------

    #[test]
    fn fm_eliminate_one_var_from_box() {
        // 0 <= x <= 5  →  project out x  →  trivially true (empty result)
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        // x >= 0
        sys.add_inequality(Aff::iter_var("x"));
        // 5 - x >= 0
        sys.add_inequality(Aff {
            constant: 5,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });

        let result = fm::fourier_motzkin_eliminate(&sys, &PolyVar::Iter("x".into()));
        let ineqs = result.expect("should not blow up");
        // Only generated constraint is 5 >= 0, which cleanup drops as trivially true.
        assert!(
            ineqs.is_empty(),
            "box projection should produce no non-trivial constraints"
        );
    }

    #[test]
    fn fm_eliminate_coupled_system() {
        // x >= 0, y - x >= 0, 3 - y >= 0  →  project out x
        // Expect: y >= 0, 3 - y >= 0 (x constraints combined yield y >= 0)
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_var(PolyVar::Iter("y".into()));
        // x >= 0
        sys.add_inequality(Aff::iter_var("x"));
        // y - x >= 0
        sys.add_inequality(Aff {
            constant: 0,
            terms: vec![
                (1, PolyVar::Iter("y".into())),
                (-1, PolyVar::Iter("x".into())),
            ],
        });
        // 3 - y >= 0
        sys.add_inequality(Aff {
            constant: 3,
            terms: vec![(-1, PolyVar::Iter("y".into()))],
        });

        let result = fm::fourier_motzkin_eliminate(&sys, &PolyVar::Iter("x".into()));
        let ineqs = result.expect("should not blow up");
        // Unrelated: 3 - y >= 0.
        // Combined from (x >= 0) × (y - x >= 0): y >= 0.
        assert_eq!(ineqs.len(), 2);
    }

    #[test]
    fn fm_detects_infeasible_after_elimination() {
        // x >= 5, 3 - x >= 0  →  project out x  →  combined: 3 - 5 >= 0 → -2 >= 0
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        // x - 5 >= 0  (i.e. x >= 5)
        sys.add_inequality(Aff {
            constant: -5,
            terms: vec![(1, PolyVar::Iter("x".into()))],
        });
        // 3 - x >= 0  (i.e. x <= 3)
        sys.add_inequality(Aff {
            constant: 3,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });

        let result = fm::fourier_motzkin_eliminate(&sys, &PolyVar::Iter("x".into()));
        let ineqs = result.expect("should not blow up");
        // Combined: -2 >= 0, which is kept (it's a contradiction, not trivially true).
        assert_eq!(ineqs.len(), 1);
        assert!(ineqs[0].terms.is_empty());
        assert!(
            ineqs[0].constant < 0,
            "should be infeasible: {} < 0",
            ineqs[0].constant
        );
    }

    #[test]
    fn fm_trivially_infeasible_detected() {
        let mut sys = ConstraintSystem::new();
        // -1 >= 0  →  infeasible
        sys.add_inequality(Aff::constant(-1));
        assert!(fm::is_trivially_infeasible(&sys));
    }

    #[test]
    fn fm_trivially_infeasible_equality() {
        let mut sys = ConstraintSystem::new();
        // 5 = 0  →  infeasible
        sys.add_equality(Aff::constant(5));
        assert!(fm::is_trivially_infeasible(&sys));
    }

    #[test]
    fn fm_feasible_system_not_flagged() {
        let mut sys = ConstraintSystem::new();
        sys.add_inequality(Aff::iter_var("x"));
        sys.add_inequality(Aff {
            constant: 5,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });
        assert!(!fm::is_trivially_infeasible(&sys));
    }

    #[test]
    fn project_out_uses_fm_for_coupled_ineqs() {
        // x >= 0, y - x >= 0, 3 - y >= 0
        // Project out x → y >= 0, 3 - y >= 0
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_var(PolyVar::Iter("y".into()));
        sys.add_inequality(Aff::iter_var("x"));
        sys.add_inequality(Aff {
            constant: 0,
            terms: vec![
                (1, PolyVar::Iter("y".into())),
                (-1, PolyVar::Iter("x".into())),
            ],
        });
        sys.add_inequality(Aff {
            constant: 3,
            terms: vec![(-1, PolyVar::Iter("y".into()))],
        });

        let result = project_out(&sys, &PolyVar::Iter("x".into()));
        assert!(!result.vars.contains(&PolyVar::Iter("x".into())));
        // y >= 0 and 3 - y >= 0
        assert_eq!(result.inequalities.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Step 6: Native feasibility checking
    // -----------------------------------------------------------------------

    #[test]
    fn feasibility_simple_box_is_feasible() {
        // 0 <= x <= 5  →  feasible
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_inequality(Aff::iter_var("x"));
        sys.add_inequality(Aff {
            constant: 5,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });
        assert_eq!(check_feasibility(&sys), Feasibility::Feasible);
        assert!(!sys.is_empty());
    }

    #[test]
    fn feasibility_contradictory_bounds_is_infeasible() {
        // x >= 5, x <= 3  →  infeasible
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_inequality(Aff {
            constant: -5,
            terms: vec![(1, PolyVar::Iter("x".into()))],
        });
        sys.add_inequality(Aff {
            constant: 3,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });
        assert_eq!(check_feasibility(&sys), Feasibility::Infeasible);
        assert!(sys.is_empty());
    }

    #[test]
    fn feasibility_equality_with_no_solution() {
        // x = 5, x <= 3  →  infeasible
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_equality(Aff {
            constant: -5,
            terms: vec![(1, PolyVar::Iter("x".into()))],
        });
        sys.add_inequality(Aff {
            constant: 3,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });
        assert_eq!(check_feasibility(&sys), Feasibility::Infeasible);
    }

    #[test]
    fn feasibility_equality_with_solution() {
        // x = 3, 0 <= x <= 5  →  feasible (x=3)
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_equality(Aff {
            constant: -3,
            terms: vec![(1, PolyVar::Iter("x".into()))],
        });
        sys.add_inequality(Aff::iter_var("x"));
        sys.add_inequality(Aff {
            constant: 5,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });
        assert_eq!(check_feasibility(&sys), Feasibility::Feasible);
    }

    #[test]
    fn feasibility_2d_box() {
        // 0 <= x <= 3, 0 <= y <= 7  →  feasible
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_var(PolyVar::Iter("y".into()));
        sys.add_inequality(Aff::iter_var("x"));
        sys.add_inequality(Aff {
            constant: 3,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });
        sys.add_inequality(Aff::iter_var("y"));
        sys.add_inequality(Aff {
            constant: 7,
            terms: vec![(-1, PolyVar::Iter("y".into()))],
        });
        assert_eq!(check_feasibility(&sys), Feasibility::Feasible);
    }

    #[test]
    fn feasibility_coupled_infeasible() {
        // x >= 0, y >= 0, x + y <= -1  →  infeasible
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_var(PolyVar::Iter("y".into()));
        sys.add_inequality(Aff::iter_var("x"));
        sys.add_inequality(Aff::iter_var("y"));
        // -1 - x - y >= 0
        sys.add_inequality(Aff {
            constant: -1,
            terms: vec![
                (-1, PolyVar::Iter("x".into())),
                (-1, PolyVar::Iter("y".into())),
            ],
        });
        assert_eq!(check_feasibility(&sys), Feasibility::Infeasible);
    }

    #[test]
    fn feasibility_trivial_constant_contradiction() {
        // -1 >= 0  →  infeasible
        let mut sys = ConstraintSystem::new();
        sys.add_inequality(Aff::constant(-1));
        assert_eq!(check_feasibility(&sys), Feasibility::Infeasible);
    }

    #[test]
    fn feasibility_empty_system() {
        // No constraints  →  feasible (any point works)
        let sys = ConstraintSystem::new();
        assert_eq!(check_feasibility(&sys), Feasibility::Feasible);
    }

    #[test]
    fn feasibility_symbolic_parameter_is_unknown() {
        // N >= 0 with no iterator variables remains parameter-dependent.
        // Proof-oriented checker should not claim Feasible.
        let mut sys = ConstraintSystem::new();
        let n = Symbol(99);
        sys.add_var(PolyVar::Param(n));
        sys.add_inequality(Aff {
            constant: 0,
            terms: vec![(1, PolyVar::Param(n))],
        });

        assert_eq!(check_feasibility(&sys), Feasibility::Unknown);
    }

    #[test]
    fn feasibility_non_unit_equality_with_bounds_is_unknown() {
        // 2*x + y = 0, x >= 0, y >= 1 has no solution with x integer and y odd,
        // but exact elimination is unavailable here; checker must not claim Feasible.
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_var(PolyVar::Iter("y".into()));
        sys.add_equality(Aff {
            constant: 0,
            terms: vec![
                (2, PolyVar::Iter("x".into())),
                (1, PolyVar::Iter("y".into())),
            ],
        });
        sys.add_inequality(Aff::iter_var("x"));
        sys.add_inequality(Aff {
            constant: -1,
            terms: vec![(1, PolyVar::Iter("y".into()))],
        });

        assert_eq!(check_feasibility(&sys), Feasibility::Unknown);
    }

    #[test]
    fn feasibility_constraint_system_is_empty_method() {
        // Use the ConstraintSystem::is_empty convenience method.
        let mut sys = ConstraintSystem::new();
        sys.add_var(PolyVar::Iter("x".into()));
        sys.add_inequality(Aff {
            constant: -5,
            terms: vec![(1, PolyVar::Iter("x".into()))],
        });
        sys.add_inequality(Aff {
            constant: 3,
            terms: vec![(-1, PolyVar::Iter("x".into()))],
        });
        assert!(sys.is_empty());
    }

    #[test]
    fn feasibility_relation_dependence_exists() {
        use crate::core::poly::AccessMap;

        // S[i] -> A[i]  same buffer, same access
        // Dependence relation should be feasible (i = i is always true
        // when the domain is non-empty).
        let domain = shape_to_domain(&[Dim::Const(8)]);
        let access = AccessMap {
            buffer: BufferId(0),
            domain_iters: vec!["i0".into()],
            mapping: vec![Aff::iter_var("i0")],
        };

        let rel = Relation::build_dependence(&domain, &domain, &access, &access);
        assert!(
            !rel.system.is_empty(),
            "self-dependence on non-empty domain should be feasible"
        );
    }

    // -----------------------------------------------------------------------
    // Step 7: Dependence analyzer rewrite
    // -----------------------------------------------------------------------

    #[test]
    fn poly_analyzer_finds_raw_dependence_elementwise() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let kernel = llir.kernels.last().unwrap();

        let deps = analyze_kernel_poly(kernel);
        assert!(
            !deps.is_empty(),
            "elementwise kernel should have dependences"
        );

        // All dependences should be RAW.
        assert!(deps
            .iter()
            .all(|d| d.kind == crate::core::llir::DepKind::Raw));
        Ok(())
    }

    #[test]
    fn poly_analyzer_reduction_kernel() -> Result<()> {
        use crate::core::hlir::ReduceOp;
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4), Dim::Const(8)], DType::F32);
        let a = g.load(BufferId(0), ty);
        let _r = g.reduce(a, vec![1], ReduceOp::Sum, false);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let kernel = llir.kernels.last().unwrap();

        let deps = analyze_kernel_poly(kernel);
        assert!(
            !deps.is_empty(),
            "reduction kernel should produce dependences"
        );
        Ok(())
    }

    #[test]
    fn poly_analyzer_through_trait_still_works() -> Result<()> {
        // Verify the NativeDependenceAnalyzer trait impl delegates correctly.
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(8)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let kernel = llir.kernels.last().unwrap();

        let deps = analyzer.analyze_kernel(kernel)?;
        assert!(!deps.is_empty());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Step 8: Real legality queries
    // -----------------------------------------------------------------------

    #[test]
    fn legality_positive_distance_blocks_parallelize() {
        use crate::core::poly::analysis::legality::can_parallelize;

        let dep = crate::core::llir::Dependence {
            from: 0,
            to: 0,
            kind: crate::core::llir::DepKind::Raw,
            distance: Some(vec![1]),
            relation: crate::core::llir::DependenceRelation {
                source_vars: vec!["i0".into()],
                sink_vars: vec!["i0".into()],
                constraints: vec![],
            },
        };
        assert!(!can_parallelize(&[dep], "i0"));
    }

    #[test]
    fn legality_zero_distance_allows_parallelize() {
        use crate::core::poly::analysis::legality::can_parallelize;

        let dep = crate::core::llir::Dependence {
            from: 0,
            to: 0,
            kind: crate::core::llir::DepKind::Raw,
            distance: Some(vec![0]),
            relation: crate::core::llir::DependenceRelation {
                source_vars: vec!["i0".into()],
                sink_vars: vec!["i0".into()],
                constraints: vec![],
            },
        };
        assert!(can_parallelize(&[dep], "i0"));
    }

    #[test]
    fn legality_pointwise_zero_distance_allows_parallelize() {
        use crate::core::poly::analysis::legality::can_parallelize;

        // A truly pointwise dependence: each iteration accesses its own
        // element (distance 0 on all dims).
        let dep = crate::core::llir::Dependence {
            from: 0,
            to: 0,
            kind: crate::core::llir::DepKind::Raw,
            distance: Some(vec![0, 0]),
            relation: crate::core::llir::DependenceRelation {
                source_vars: vec!["i0".into(), "i1".into()],
                sink_vars: vec!["i0".into(), "i1".into()],
                constraints: vec![],
            },
        };
        assert!(can_parallelize(&[dep.clone()], "i0"));
        assert!(can_parallelize(&[dep], "i1"));
    }

    #[test]
    fn legality_interchange_independent_loops() {
        use crate::core::poly::analysis::legality::can_interchange;

        let dep = crate::core::llir::Dependence {
            from: 0,
            to: 0,
            kind: crate::core::llir::DepKind::Raw,
            distance: Some(vec![0, 0]),
            relation: crate::core::llir::DependenceRelation {
                source_vars: vec!["i0".into(), "i1".into()],
                sink_vars: vec!["i0".into(), "i1".into()],
                constraints: vec![],
            },
        };
        assert!(can_interchange(&[dep], "i0", "i1"));
    }

    #[test]
    fn legality_interchange_carried_dep_rejected() {
        use crate::core::poly::analysis::legality::can_interchange;

        // Dependence with distance (1, -1): after interchange becomes (-1, 1)
        // which is lexicographically negative → illegal.
        let dep = crate::core::llir::Dependence {
            from: 0,
            to: 0,
            kind: crate::core::llir::DepKind::Raw,
            distance: Some(vec![1, -1]),
            relation: crate::core::llir::DependenceRelation {
                source_vars: vec!["i0".into(), "i1".into()],
                sink_vars: vec!["i0".into(), "i1".into()],
                constraints: vec![],
            },
        };
        assert!(!can_interchange(&[dep], "i0", "i1"));
    }

    #[test]
    fn legality_vectorize_rejected_on_waw() {
        use crate::core::poly::analysis::legality::can_vectorize;

        let dep = crate::core::llir::Dependence {
            from: 0,
            to: 0,
            kind: crate::core::llir::DepKind::Waw,
            distance: Some(vec![0]),
            relation: crate::core::llir::DependenceRelation {
                source_vars: vec!["i0".into()],
                sink_vars: vec!["i0".into()],
                constraints: vec![],
            },
        };

        // Need a minimal kernel to check LoopKind.
        let kernel = crate::core::llir::Kernel {
            id: crate::core::llir::program::KernelId(0),
            name: "test".into(),
            root: crate::core::hlir::NodeId(0),
            op: Op::Add(crate::core::hlir::NodeId(0), crate::core::hlir::NodeId(1)),
            ty: TensorType::contiguous(vec![Dim::Const(4)], DType::F32),
            loop_nest: crate::core::llir::LoopNest {
                loops: vec![crate::core::llir::Loop {
                    var: "i0".into(),
                    lower: AffineExpr::constant(0),
                    upper: AffineExpr::constant(4),
                    step: 1,
                    kind: crate::core::llir::LoopKind::Sequential,
                    annotations: Default::default(),
                }],
                body: vec![],
            },
            allocs: vec![],
        };

        assert!(!can_vectorize(&[dep], "i0", &kernel));
    }

    #[test]
    fn legality_pad_to_always_legal() {
        use crate::core::poly::analysis::legality::can_pad_to;

        let dep = crate::core::llir::Dependence {
            from: 0,
            to: 0,
            kind: crate::core::llir::DepKind::Raw,
            distance: Some(vec![1]),
            relation: crate::core::llir::DependenceRelation {
                source_vars: vec!["i0".into()],
                sink_vars: vec!["i0".into()],
                constraints: vec![],
            },
        };
        assert!(can_pad_to(&[dep], "i0"));
    }

    // -----------------------------------------------------------------------
    // Step 9: Legality wired into lowering
    // -----------------------------------------------------------------------

    #[test]
    fn lowering_legal_tiled_pointwise_survives() -> Result<()> {
        use crate::core::schedule::{Opt, OptOp};

        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(16)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        // Build a schedule with a tile opt.
        let mut decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let fg_id = decision.fusion_groups[0].id;
        decision.opts.insert(
            fg_id,
            vec![Opt {
                op: OptOp::Tile,
                axis: 0,
                amt: 4,
            }],
        );

        let analyzer = NativeDependenceAnalyzer;
        let result = lower(&g, &decision, &analyzer);
        assert!(result.is_ok(), "legal tile should survive lowering");
        Ok(())
    }

    #[test]
    fn lowering_group_reduce_on_non_reduce_loop_fails() -> Result<()> {
        use crate::core::schedule::{Opt, OptOp};

        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(8)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let mut decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let fg_id = decision.fusion_groups[0].id;
        decision.opts.insert(
            fg_id,
            vec![Opt {
                op: OptOp::GroupReduce,
                axis: 0,
                amt: 2,
            }],
        );

        let analyzer = NativeDependenceAnalyzer;
        let result = lower(&g, &decision, &analyzer);
        assert!(
            result.is_err(),
            "GroupReduce on non-reduce loop should fail"
        );
        Ok(())
    }

    #[test]
    fn lowering_vectorize_on_reduce_loop_fails() -> Result<()> {
        use crate::core::hlir::ReduceOp;
        use crate::core::schedule::{Opt, OptOp};

        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4), Dim::Const(8)], DType::F32);
        let a = g.load(BufferId(0), ty);
        let _r = g.reduce(a, vec![1], ReduceOp::Sum, false);

        let mut decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let fg_id = decision.fusion_groups[0].id;
        // Axis 1 is the reduce loop (r0).
        decision.opts.insert(
            fg_id,
            vec![Opt {
                op: OptOp::Vectorize,
                axis: 1,
                amt: 4,
            }],
        );

        let analyzer = NativeDependenceAnalyzer;
        let result = lower(&g, &decision, &analyzer);
        assert!(result.is_err(), "Vectorize on reduce loop should fail");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Step 10: LLIR polyhedral optimization
    // -----------------------------------------------------------------------

    #[test]
    fn optimize_llir_preserves_kernel_count() -> Result<()> {
        use crate::core::compile::optimize_llir;

        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4), Dim::Const(8)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let n_kernels = llir.kernels.len();

        let optimized = optimize_llir(llir, &analyzer)?;
        assert_eq!(
            optimized.kernels.len(),
            n_kernels,
            "optimization should not add or remove kernels"
        );
        Ok(())
    }

    #[test]
    fn optimize_llir_attempts_interchange_on_eligible_kernel() -> Result<()> {
        use crate::core::compile::optimize_llir;
        use crate::core::llir::{LoopKind, LoopNest};

        // Build a kernel where inner loop has a larger bound than outer.
        // Loops: i0[0..4], i1[0..16] → interchange should swap them.
        let kernel = crate::core::llir::Kernel {
            id: crate::core::llir::program::KernelId(0),
            name: "test".into(),
            root: crate::core::hlir::NodeId(0),
            op: Op::Add(crate::core::hlir::NodeId(0), crate::core::hlir::NodeId(1)),
            ty: TensorType::contiguous(vec![Dim::Const(4), Dim::Const(16)], DType::F32),
            loop_nest: LoopNest {
                loops: vec![
                    crate::core::llir::Loop {
                        var: "i0".into(),
                        lower: AffineExpr::constant(0),
                        upper: AffineExpr::constant(4),
                        step: 1,
                        kind: LoopKind::Sequential,
                        annotations: Default::default(),
                    },
                    crate::core::llir::Loop {
                        var: "i1".into(),
                        lower: AffineExpr::constant(0),
                        upper: AffineExpr::constant(16),
                        step: 1,
                        kind: LoopKind::Sequential,
                        annotations: Default::default(),
                    },
                ],
                body: vec![],
            },
            allocs: vec![],
        };

        let program = crate::core::llir::LLIRProgram {
            kernels: vec![kernel],
        };
        let analyzer = NativeDependenceAnalyzer;
        let optimized = optimize_llir(program, &analyzer)?;

        // With empty body (no dependences), interchange should be legal
        // and applied since inner (16) > outer (4).
        let loops = &optimized.kernels[0].loop_nest.loops;
        assert_eq!(loops[0].var, "i1", "larger loop should move to outer");
        assert_eq!(loops[1].var, "i0", "smaller loop should move to inner");
        Ok(())
    }

    #[test]
    fn optimize_llir_skips_when_outer_already_larger() -> Result<()> {
        use crate::core::compile::optimize_llir;
        use crate::core::llir::{LoopKind, LoopNest};

        // Outer already has larger bound → no interchange needed.
        let kernel = crate::core::llir::Kernel {
            id: crate::core::llir::program::KernelId(0),
            name: "test".into(),
            root: crate::core::hlir::NodeId(0),
            op: Op::Add(crate::core::hlir::NodeId(0), crate::core::hlir::NodeId(1)),
            ty: TensorType::contiguous(vec![Dim::Const(16), Dim::Const(4)], DType::F32),
            loop_nest: LoopNest {
                loops: vec![
                    crate::core::llir::Loop {
                        var: "i0".into(),
                        lower: AffineExpr::constant(0),
                        upper: AffineExpr::constant(16),
                        step: 1,
                        kind: LoopKind::Sequential,
                        annotations: Default::default(),
                    },
                    crate::core::llir::Loop {
                        var: "i1".into(),
                        lower: AffineExpr::constant(0),
                        upper: AffineExpr::constant(4),
                        step: 1,
                        kind: LoopKind::Sequential,
                        annotations: Default::default(),
                    },
                ],
                body: vec![],
            },
            allocs: vec![],
        };

        let program = crate::core::llir::LLIRProgram {
            kernels: vec![kernel],
        };
        let analyzer = NativeDependenceAnalyzer;
        let optimized = optimize_llir(program, &analyzer)?;

        // Order should be preserved.
        let loops = &optimized.kernels[0].loop_nest.loops;
        assert_eq!(loops[0].var, "i0");
        assert_eq!(loops[1].var, "i1");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Step 11: Search uses polyhedral facts
    // -----------------------------------------------------------------------

    #[test]
    fn candidates_exclude_parallelize_on_carried_dep_axis() {
        use crate::core::schedule::candidates::opt_candidates;
        use crate::core::schedule::{BackendClass, KernelContext, OptOp};

        let ctx = KernelContext {
            loop_bounds: vec![64, 64],
            reduce_axes: vec![],
            carried_dep_axes: vec![0], // axis 0 has a carried dependence
            dtype: DType::F32,
            shared_budget: 0,
            backend: BackendClass::Cpu,
        };
        let opts = opt_candidates(&ctx);

        // No Parallelize on axis 0.
        let par_on_0 = opts
            .iter()
            .any(|o| o.op == OptOp::Parallelize && o.axis == 0);
        assert!(
            !par_on_0,
            "Parallelize should be excluded on carried-dep axis"
        );

        // Parallelize on axis 1 should still be present.
        let par_on_1 = opts
            .iter()
            .any(|o| o.op == OptOp::Parallelize && o.axis == 1);
        assert!(par_on_1, "Parallelize should remain on non-carried axis");
    }

    #[test]
    fn candidates_exclude_vectorize_on_carried_dep_axis() {
        use crate::core::schedule::candidates::opt_candidates;
        use crate::core::schedule::{BackendClass, KernelContext, OptOp};

        let ctx = KernelContext {
            loop_bounds: vec![64],
            reduce_axes: vec![],
            carried_dep_axes: vec![0],
            dtype: DType::F32,
            shared_budget: 0,
            backend: BackendClass::Cpu,
        };
        let opts = opt_candidates(&ctx);

        let vec_on_0 = opts.iter().any(|o| o.op == OptOp::Vectorize && o.axis == 0);
        assert!(
            !vec_on_0,
            "Vectorize should be excluded on carried-dep axis"
        );
    }

    #[test]
    fn candidates_still_allow_tile_on_carried_dep_axis() {
        use crate::core::schedule::candidates::opt_candidates;
        use crate::core::schedule::{BackendClass, KernelContext, OptOp};

        let ctx = KernelContext {
            loop_bounds: vec![64],
            reduce_axes: vec![],
            carried_dep_axes: vec![0],
            dtype: DType::F32,
            shared_budget: 0,
            backend: BackendClass::Cpu,
        };
        let opts = opt_candidates(&ctx);

        let tile_on_0 = opts.iter().any(|o| o.op == OptOp::Tile && o.axis == 0);
        assert!(
            tile_on_0,
            "Tile should still be allowed on carried-dep axis"
        );
    }
}
