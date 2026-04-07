use venum::{
    check_feasibility, extract_instances, legality, run_context, AccessKind, Aff, AffineConstraint,
    AffineExpr, BinaryOp, Buffer, BufferId, ConstraintKind, Context, DType, DependenceAnalyzer,
    Dim, Expr, Feasibility, Kernel, KernelId, Loop, LoopKind, LoopNest, MemoryAccess,
    NativeDependenceAnalyzer, NodeId, Op, PolyVar, Relation, Scalar, Stmt, Symbol, Tensor,
    TensorType, Var,
};

fn main() -> anyhow::Result<()> {
    println!("=== Native Polyhedral Demo ===");

    example_runtime_and_extraction()?;
    example_access_dependence()?;
    example_feasibility_unknown();

    Ok(())
}

fn example_runtime_and_extraction() -> anyhow::Result<()> {
    println!("\n-- Example 1: Real kernel extraction + legality --");

    let cx = Context::new();
    let a = Tensor::placeholder(&cx, DType::F32, vec![8, 8]);
    let b = Tensor::placeholder(&cx, DType::F32, vec![8, 8]);
    let c = a.mul(&b)?.add(&a)?;

    let out = run_context(
        &cx,
        &[c.id()],
        &[
            Buffer::F32((0..64).map(|x| x as f32).collect()),
            Buffer::F32((0..64).map(|x| (x as f32) * 0.5).collect()),
        ],
    )?;

    match &out[0] {
        Buffer::F32(v) => println!("runtime output length: {}", v.len()),
        _ => println!("unexpected output dtype"),
    }

    // Build a tiny LLIR kernel directly so we can showcase extraction and
    // legality without touching private lowering APIs.
    let kernel = Kernel {
        id: KernelId(0),
        name: "demo_kernel".into(),
        root: NodeId(0),
        op: Op::Add(NodeId(0), NodeId(1)),
        ty: TensorType::contiguous(vec![Dim::Const(8)], DType::F32),
        loop_nest: LoopNest {
            loops: vec![Loop {
                var: "i0".into(),
                lower: AffineExpr::constant(0),
                upper: AffineExpr::constant(8),
                step: 1,
                kind: LoopKind::Sequential,
                annotations: Default::default(),
            }],
            body: vec![Stmt::If {
                cond: Expr::Binary {
                    op: BinaryOp::Lt,
                    lhs: Box::new(Expr::Literal(Scalar::I32(0))),
                    rhs: Box::new(Expr::Literal(Scalar::I32(4))),
                },
                then_body: vec![Stmt::Assign {
                    dst: MemoryAccess {
                        buffer: BufferId(0),
                        indices: vec![AffineExpr::constant(0).with_term(1, Var::Loop("i0".into()))],
                        access_kind: AccessKind::Write,
                    },
                    src: Expr::Load(MemoryAccess {
                        buffer: BufferId(1),
                        indices: vec![AffineExpr::constant(0).with_term(1, Var::Loop("i0".into()))],
                        access_kind: AccessKind::Read,
                    }),
                }],
                else_body: vec![],
            }],
        },
        allocs: vec![],
    };

    let instances = extract_instances(&kernel);
    println!("extracted statement instances: {}", instances.len());

    let dep = NativeDependenceAnalyzer;
    let deps = dep.analyze_kernel(&kernel)?;
    println!("analyzed dependences: {}", deps.len());
    println!(
        "parallelize i0 legal: {}",
        legality::can_parallelize(&deps, "i0")
    );

    Ok(())
}

fn example_access_dependence() -> anyhow::Result<()> {
    println!("\n-- Example 2: Direct access_dependence --");

    let analyzer = NativeDependenceAnalyzer;
    let loops = vec![Loop {
        var: "i0".into(),
        lower: AffineExpr::constant(0),
        upper: AffineExpr::constant(8),
        step: 1,
        kind: LoopKind::Sequential,
        annotations: Default::default(),
    }];

    let write = MemoryAccess {
        buffer: BufferId(0),
        indices: vec![AffineExpr::constant(0).with_term(1, Var::Loop("i0".into()))],
        access_kind: AccessKind::Write,
    };
    let read_same = MemoryAccess {
        buffer: BufferId(0),
        indices: vec![AffineExpr::constant(0).with_term(1, Var::Loop("i0".into()))],
        access_kind: AccessKind::Read,
    };
    let read_shifted = MemoryAccess {
        buffer: BufferId(0),
        indices: vec![AffineExpr::constant(1).with_term(1, Var::Loop("i0".into()))],
        access_kind: AccessKind::Read,
    };

    let rel_same = analyzer.access_dependence(&write, &read_same, &loops)?;
    println!("same-cell dependence exists: {}", rel_same.is_some());
    if let Some(rel) = rel_same {
        let eq = rel
            .constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Eq)
            .count();
        let ge = rel
            .constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Ge)
            .count();
        println!(
            "  constraints: total={}, eq={}, ge={}",
            rel.constraints.len(),
            eq,
            ge
        );
    }

    let rel_shifted = analyzer.access_dependence(&write, &read_shifted, &loops)?;
    println!(
        "shifted (reverse-time) dependence exists: {}",
        rel_shifted.is_some()
    );

    Ok(())
}

fn example_feasibility_unknown() {
    println!("\n-- Example 3: Proof-oriented feasibility --");

    let domain = venum::shape_to_domain(&[Dim::Const(8)]);
    let access = venum::AccessMap {
        buffer: BufferId(0),
        domain_iters: vec!["i0".into()],
        mapping: vec![Aff::iter_var("i0")],
    };
    let rel = Relation::build_dependence(&domain, &domain, &access, &access);
    println!("simple relation empty: {}", rel.system.is_empty());

    let mut sys = venum::ConstraintSystem::new();
    let n = Symbol(7);
    sys.add_var(PolyVar::Param(n));
    sys.add_inequality(Aff {
        constant: 0,
        terms: vec![(1, PolyVar::Param(n))],
    });

    let feas = check_feasibility(&sys);
    println!("symbolic feasibility result: {:?}", feas);
    assert!(matches!(feas, Feasibility::Unknown));

    let sample = AffineConstraint {
        expr: AffineExpr::constant(7).with_term(-1, Var::Loop("i0".into())),
        kind: ConstraintKind::Ge,
    };
    println!("sample affine constraint: {:?}", sample);
}
