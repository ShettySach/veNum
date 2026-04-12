use super::cost::CanonicalExprCostModel;
use super::decode::decode_with_extracted;
use super::encode::{encode_algebraic, EggExpr};
use super::extract::{extract_program_roots, ExtractionConfig};
use super::facts::{AxesId, DimFact, FactDatabase, LayoutFact, ScalarArena};
use super::region::{plan_regions, LeafRef};
use crate::core::hlir::types::{f32_to_bf16_bits, f32_to_f16_bits};
use crate::core::hlir::Op;
use crate::core::hlir::{
    BufferId, DType, Dim, HLIRGraph, Range, ReduceOp, Scalar, Symbol, TensorType,
};

const TYPED_SCHEMA: &str = include_str!("schema.egg");

fn f32_ty(dims: &[i64]) -> TensorType {
    TensorType::contiguous(dims.iter().map(|&d| Dim::Const(d)).collect(), DType::F32)
}

fn assert_typed_schema_program_ok(program: &str) {
    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(None, TYPED_SCHEMA).unwrap();
    egraph.parse_and_run_program(None, program).unwrap();
}

fn assert_typed_schema_program_err(program: &str) {
    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(None, TYPED_SCHEMA).unwrap();
    let result = egraph.parse_and_run_program(None, program);
    assert!(result.is_err(), "expected egglog program to fail");
}

fn extract_typed_schema_expr(program: &str, root_expr: &str) -> String {
    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(None, TYPED_SCHEMA).unwrap();
    egraph.parse_and_run_program(None, program).unwrap();

    let expr = egglog::ast::Parser::default()
        .get_expr_from_string(None, root_expr)
        .unwrap();
    let (sort, value) = egraph.eval_expr(&expr).unwrap();
    let extractor = egglog::extract::Extractor::compute_costs_from_rootsorts(
        Some(vec![sort.clone()]),
        &egraph,
        CanonicalExprCostModel,
    );
    let mut termdag = egglog::TermDag::default();
    let (_, term) = extractor
        .extract_best_with_sort(&egraph, &mut termdag, value, sort)
        .unwrap();
    termdag.to_string(term)
}

fn extract_typed_schema_expr_default(program: &str, root_expr: &str) -> String {
    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(None, TYPED_SCHEMA).unwrap();
    egraph.parse_and_run_program(None, program).unwrap();

    let expr = egglog::ast::Parser::default()
        .get_expr_from_string(None, root_expr)
        .unwrap();
    let (sort, value) = egraph.eval_expr(&expr).unwrap();
    egraph.extract_value_to_string(&sort, value).unwrap().0
}

#[test]
fn region_planner_isolates_reduce_between_algebraic_regions() {
    let mut g = HLIRGraph::new();
    let x = g.load(BufferId(0), f32_ty(&[4, 8]));
    let y = g.load(BufferId(1), f32_ty(&[4]));
    let zero = g.constant(
        Scalar::F32(0.0),
        vec![Dim::Const(4), Dim::Const(8)],
        DType::F32,
    );
    let add_in = g.binary(x, zero, crate::core::hlir::Op::Add);
    let reduce = g.reduce(add_in, vec![1], ReduceOp::Sum, false);
    let root = g.binary(reduce, y, crate::core::hlir::Op::Add);

    let plan = plan_regions(&g, &[root]);

    assert_eq!(plan.barriers.len(), 1);
    assert_eq!(plan.barriers[0].src_id, reduce);
    assert!(plan.region_roots.contains(&root));
    assert!(plan.region_roots.contains(&add_in));
    assert!(plan
        .symbolic_leaves
        .contains(&LeafRef::BarrierOutput(reduce)));
}

#[test]
fn shared_barrier_outputs_are_recorded_once_and_reused() {
    let mut g = HLIRGraph::new();
    let x = g.load(BufferId(0), f32_ty(&[4, 8]));
    let reduce = g.reduce(x, vec![1], ReduceOp::Sum, false);
    let root = g.binary(reduce, reduce, crate::core::hlir::Op::Add);

    let plan = plan_regions(&g, &[root]);
    let barrier_leaves = plan
        .symbolic_leaves
        .iter()
        .filter(|leaf| matches!(leaf, LeafRef::BarrierOutput(id) if *id == reduce))
        .count();

    assert_eq!(plan.barriers.len(), 1);
    assert_eq!(barrier_leaves, 1);
}

#[test]
fn barrier_order_is_topological_and_deterministic() {
    let mut g = HLIRGraph::new();
    let x = g.load(BufferId(0), f32_ty(&[4, 8]));
    let sliced = g.slice(
        x,
        vec![
            Range {
                start: Dim::Const(0),
                end: Dim::Const(4),
            },
            Range {
                start: Dim::Const(0),
                end: Dim::Const(8),
            },
        ],
    );
    let reduced = g.reduce(sliced, vec![1], ReduceOp::Sum, false);

    let plan = plan_regions(&g, &[reduced]);
    let barrier_ids: Vec<_> = plan.barriers.iter().map(|barrier| barrier.src_id).collect();

    assert_eq!(barrier_ids, vec![sliced, reduced]);
}

#[test]
fn scalar_arena_interns_exact_bit_patterns() {
    let mut arena = ScalarArena::new();
    let pos_zero = arena.intern(Scalar::F32(0.0));
    let neg_zero = arena.intern(Scalar::F32(f32::from_bits(0x8000_0000)));

    assert_ne!(pos_zero, neg_zero);
    assert!(arena.is_zero(pos_zero));
    assert!(arena.is_zero(neg_zero));
}

#[test]
fn scalar_arena_i64_folding_remains_exact() {
    let mut arena = ScalarArena::new();
    let max = arena.intern(Scalar::I64(i64::MAX));
    let one = arena.intern(Scalar::I64(1));
    let wrapped = arena.add(max, one);

    assert_eq!(arena.value(wrapped).bits, Scalar::I64(i64::MIN));
    assert_eq!(wrapped, arena.add(max, one));
    assert!(arena.is_one(one));
}

#[test]
fn scalar_arena_f16_folding_preserves_exact_bit_patterns() {
    let mut arena = ScalarArena::new();
    let one = arena.intern(Scalar::F16(0x3C00));
    let half = arena.intern(Scalar::F16(0x3800));
    let sum = arena.add(one, half);
    let zero = arena.intern(Scalar::F16(0x0000));
    let neg_zero = arena.neg(zero);

    assert_eq!(arena.value(sum).bits, Scalar::F16(f32_to_f16_bits(1.5)));
    assert_eq!(arena.value(neg_zero).bits, Scalar::F16(0x8000));
    assert!(arena.is_zero(neg_zero));
    assert!(arena.is_one(one));
}

#[test]
fn scalar_arena_bf16_folding_preserves_exact_bit_patterns() {
    let mut arena = ScalarArena::new();
    let one = arena.intern(Scalar::BF16(0x3F80));
    let half = arena.intern(Scalar::BF16(0x3F00));
    let sum = arena.add(one, half);

    assert_eq!(arena.value(sum).bits, Scalar::BF16(f32_to_bf16_bits(1.5)));
    assert_eq!(sum, arena.add(one, half));
    assert!(arena.is_one(one));
}

#[test]
fn typed_schema_parses_and_accepts_semantic_fact_commands() {
    let mut egraph = egglog::EGraph::default();
    egraph.parse_and_run_program(None, TYPED_SCHEMA).unwrap();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (let $shape0 (ShapeNil))
            (let $shape1 (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
            (let $axes01 (AxesCons 0 (AxesCons 1 (AxesNil))))
            (let $scalar (SConst 7))
            (let $leaf (Leaf 99))
            (let $const (Const $scalar $shape0 (F32)))

            (set (shape-of $leaf) $shape1)
            (set (dtype-of $leaf) (F32))
            (set (layout-of $leaf) (Contiguous))
            (set (shape-of $const) $shape0)
            (set (dtype-of $const) (F32))
            (set (layout-of $const) (Contiguous))
            (set (scalar-of $const) $scalar)
            (set (rank-of $shape0) 0)
            (set (rank-of $shape1) 2)
            (set (numel-id-of $shape1) 123)
            (set (base-of $leaf) $leaf)
            (set (coeff-of $leaf) $scalar)
            (set (scalar-dtype $scalar) (F32))
            (set (scalar-value-id $scalar) 7)

            (scalar-shape $shape0)
            (same-shape $shape1 $shape1)
            (same-numel $shape1 $shape1)
            (reshape-ok $shape1 $shape1)
            (broadcast-ok $shape0 $shape1)
            (expand-ok $shape1 $shape1)
            (permute-ok $shape1 $axes01 $shape1)
            (axes-compose $axes01 $axes01 $axes01)
            (axes-identity $axes01)
            (scalar-zero $scalar)
            (scalar-one $scalar)
            (scalar-neg $scalar $scalar)
            (scalar-add $scalar $scalar $scalar)
            (scalar-mul $scalar $scalar $scalar)
            (unary-pointwise (Neg $leaf))
            (binary-pointwise (Add $leaf $leaf))
            (scalar-expr $const)
            (tensor-expr $leaf)
            (legal-add-shapes $shape1 $shape1 $shape1)
            (legal-mul-shapes $shape1 $shape1 $shape1)
            (legal-max-shapes $shape1 $shape1 $shape1)
            (legal-min-shapes $shape1 $shape1 $shape1)

            (check (= (shape-of $leaf) $shape1))
            (check (= (dtype-of $const) (F32)))
            (check (= (layout-of $leaf) (Contiguous)))
            (check (= (rank-of $shape1) 2))
            (check (= (numel-id-of $shape1) 123))
            (check (= (scalar-dtype $scalar) (F32)))
            (check (= (scalar-value-id $scalar) 7))
            (check (scalar-shape $shape0))
            (check (broadcast-ok $shape0 $shape1))
            (check (permute-ok $shape1 $axes01 $shape1))
            (check (legal-add-shapes $shape1 $shape1 $shape1))
            "#,
        )
        .unwrap();
}

#[test]
fn typed_schema_rewrites_basic_identities_and_involutions() {
    assert_typed_schema_program_ok(
        r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $x (Leaf 0))
        (let $zero-s (SConst 0))
        (let $one-s (SConst 1))
        (let $zero (Const $zero-s $shape (F32)))
        (let $one (Const $one-s $shape (F32)))
        (let $add-right (Add $x $zero))
        (let $add-left (Add $zero $x))
        (let $mul-one-right (Mul $x $one))
        (let $mul-one-left (Mul $one $x))
        (let $mul-zero-right (Mul $x $zero))
        (let $mul-zero-left (Mul $zero $x))
        (let $double-neg (Neg (Neg $x)))
        (let $double-recip (Recip (Recip $x)))

        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (set (shape-of $zero) $shape)
        (set (dtype-of $zero) (F32))
        (set (scalar-of $zero) $zero-s)
        (set (shape-of $one) $shape)
        (set (dtype-of $one) (F32))
        (set (scalar-of $one) $one-s)

        (scalar-zero $zero-s)
        (scalar-one $one-s)

        (run-schedule (saturate (run)))

        (check (= $add-right $x))
        (check (= $add-left $x))
        (check (= $mul-one-right $x))
        (check (= $mul-one-left $x))
        (check (= $mul-zero-right $zero))
        (check (= $mul-zero-left $zero))
        (check (= $double-neg $x))
        (check (= $double-recip $x))
        "#,
    );
}

#[test]
fn typed_schema_gates_fast_math_inverse_rules() {
    assert_typed_schema_program_err(
        r#"
        (let $x (Leaf 0))
        (let $expr (Exp (Log $x)))
        (run-schedule (saturate (run)))
        (check (= $expr $x))
        "#,
    );

    assert_typed_schema_program_ok(
        r#"
        (let $x (Leaf 0))
        (let $exp-log (Exp (Log $x)))
        (let $log-exp (Log (Exp $x)))
        (fast-math-enabled)
        (run-schedule (saturate (run)))
        (check (= $exp-log $x))
        (check (= $log-exp $x))
        "#,
    );
}

#[test]
fn typed_schema_rewrites_reshape_view_rules_when_legal() {
    assert_typed_schema_program_ok(
        r#"
        (let $src (ShapeCons 2 (ShapeCons 2 (ShapeNil))))
        (let $flat (ShapeCons 4 (ShapeNil)))
        (let $x (Leaf 0))
        (let $y (Leaf 1))
        (let $reshape-collapse (Reshape (Reshape $x $src) $flat))
        (let $neg-reshape (Neg (Reshape $x $flat)))
        (let $cast-reshape (Cast (Reshape $x $flat) (F16)))
        (let $add-reshape (Add (Reshape $x $flat) (Reshape $y $flat)))

        (set (shape-of $x) $src)
        (set (shape-of $y) $src)

        (reshape-ok $src $src)
        (reshape-ok $src $flat)
        (reshape-ok $flat $flat)
        (legal-add-shapes $src $src $src)

        (run-schedule (saturate (run)))

        (check (= $reshape-collapse (Reshape $x $flat)))
        (check (= $neg-reshape (Reshape (Neg $x) $flat)))
        (check (= $cast-reshape (Reshape (Cast $x (F16)) $flat)))
        (check (= $add-reshape (Reshape (Add $x $y) $flat)))
        "#,
    );
}

#[test]
fn typed_schema_blocks_illegal_reshape_sinking_without_source_legality() {
    assert_typed_schema_program_err(
        r#"
        (let $src22 (ShapeCons 2 (ShapeCons 2 (ShapeNil))))
        (let $src4 (ShapeCons 4 (ShapeNil)))
        (let $flat (ShapeCons 4 (ShapeNil)))
        (let $x (Leaf 0))
        (let $y (Leaf 1))
        (let $add-reshape (Add (Reshape $x $flat) (Reshape $y $flat)))

        (set (shape-of $x) $src22)
        (set (shape-of $y) $src4)

        (reshape-ok $src22 $flat)
        (reshape-ok $src4 $flat)
        (reshape-ok $flat $flat)
        (legal-add-shapes $flat $flat $flat)

        (run-schedule (saturate (run)))

        (check (= $add-reshape (Reshape (Add $x $y) $flat)))
        "#,
    );
}

#[test]
fn typed_schema_rewrites_permute_view_rules_when_legal() {
    assert_typed_schema_program_ok(
        r#"
        (let $src (ShapeCons 7 (ShapeCons 11 (ShapeCons 13 (ShapeNil)))))
        (let $p102-shape (ShapeCons 11 (ShapeCons 7 (ShapeCons 13 (ShapeNil)))))
        (let $p021-shape (ShapeCons 7 (ShapeCons 13 (ShapeCons 11 (ShapeNil)))))
        (let $p102 (AxesCons 1 (AxesCons 0 (AxesCons 2 (AxesNil)))))
        (let $p120 (AxesCons 1 (AxesCons 2 (AxesCons 0 (AxesNil)))))
        (let $p021 (AxesCons 0 (AxesCons 2 (AxesCons 1 (AxesNil)))))
        (let $x (Leaf 0))
        (let $y (Leaf 1))
        (let $permute-collapse (Permute (Permute $x $p102) $p120))
        (let $neg-permute (Neg (Permute $x $p102)))
        (let $add-permute (Add (Permute $x $p102) (Permute $y $p102)))

        (set (shape-of $x) $src)
        (set (shape-of $y) $src)

        (permute-ok $src $p102 $p102-shape)
        (permute-ok $p102-shape $p120 $p021-shape)
        (permute-ok $src $p021 $p021-shape)
        (axes-compose $p102 $p120 $p021)
        (legal-add-shapes $src $src $src)
        (legal-add-shapes $p102-shape $p102-shape $p102-shape)

        (run-schedule (saturate (run)))

        (check (= $permute-collapse (Permute $x $p021)))
        (check (= $neg-permute (Permute (Neg $x) $p102)))
        (check (= $add-permute (Permute (Add $x $y) $p102)))
        "#,
    );
}

#[test]
fn typed_schema_rewrites_expand_and_broadcast_view_rules_when_legal() {
    assert_typed_schema_program_ok(
        r#"
        (let $scalar (ShapeNil))
        (let $s43 (ShapeCons 4 (ShapeCons 3 (ShapeNil))))
        (let $s143 (ShapeCons 1 (ShapeCons 4 (ShapeCons 3 (ShapeNil)))))
        (let $s243 (ShapeCons 2 (ShapeCons 4 (ShapeCons 3 (ShapeNil)))))
        (let $a (Leaf 0))
        (let $b (Leaf 1))
        (let $x (Leaf 2))
        (let $y (Leaf 3))
        (let $broadcast-collapse (Broadcast (Broadcast $a $s43) $s243))
        (let $neg-broadcast (Neg (Broadcast $a $s243)))
        (let $add-broadcast (Add (Broadcast $a $s243) (Broadcast $b $s243)))
        (let $expand-collapse (Expand (Expand $x $s143) $s243))
        (let $sqrt-expand (Sqrt (Expand $x $s243)))
        (let $add-expand (Add (Expand $x $s243) (Expand $y $s243)))

        (set (shape-of $a) $scalar)
        (set (shape-of $b) $scalar)
        (set (shape-of $x) $s143)
        (set (shape-of $y) $s143)

        (broadcast-ok $scalar $scalar)
        (broadcast-ok $scalar $s43)
        (broadcast-ok $s43 $s243)
        (broadcast-ok $scalar $s243)
        (expand-ok $s143 $s143)
        (expand-ok $s143 $s243)
        (expand-ok $s243 $s243)
        (legal-add-shapes $scalar $scalar $scalar)
        (legal-add-shapes $s143 $s143 $s143)

        (run-schedule (saturate (run)))

        (check (= $broadcast-collapse (Broadcast $a $s243)))
        (check (= $neg-broadcast (Broadcast (Neg $a) $s243)))
        (check (= $add-broadcast (Broadcast (Add $a $b) $s243)))

        (check (= $expand-collapse (Expand $x $s243)))
        (check (= $sqrt-expand (Expand (Sqrt $x) $s243)))
        (check (= $add-expand (Expand (Add $x $y) $s243)))
        "#,
    );
}

#[test]
fn typed_schema_extracts_identity_forms_to_the_same_canonical_term() {
    let add_program = r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $x (Leaf 0))
        (let $zero-s (SConst 0))
        (let $zero (Const $zero-s $shape (F32)))
        (let $add-right (Add $x $zero))
        (let $add-left (Add $zero $x))

        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (set (shape-of $zero) $shape)
        (set (dtype-of $zero) (F32))
        (set (scalar-of $zero) $zero-s)

        (scalar-zero $zero-s)

        (run-schedule (saturate (run)))
        "#;
    let mul_program = r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $x (Leaf 0))
        (let $one-s (SConst 1))
        (let $one (Const $one-s $shape (F32)))
        (let $mul-right (Mul $x $one))
        (let $mul-left (Mul $one $x))

        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (set (shape-of $one) $shape)
        (set (dtype-of $one) (F32))
        (set (scalar-of $one) $one-s)

        (scalar-one $one-s)

        (run-schedule (saturate (run)))
        "#;

    let add_right = extract_typed_schema_expr_default(add_program, "$add-right");
    let add_left = extract_typed_schema_expr_default(add_program, "$add-left");
    let mul_right = extract_typed_schema_expr_default(mul_program, "$mul-right");
    let mul_left = extract_typed_schema_expr_default(mul_program, "$mul-left");

    assert_eq!(add_right, "(Leaf 0)");
    assert_eq!(add_left, add_right);
    assert_eq!(mul_right, "(Leaf 0)");
    assert_eq!(mul_left, mul_right);
}

#[test]
fn typed_schema_extracts_add_operands_in_stable_canonical_order() {
    let program = r#"
        (let $a (Leaf 0))
        (let $b (Leaf 1))
        (let $c (Leaf 2))
        (let $lhs (Add (Add $a $b) $c))
        (let $rhs (Add $c (Add $b $a)))
        (run-schedule (saturate (run)))
        (run-schedule (saturate canonical-order))
        "#;

    let lhs = extract_typed_schema_expr(program, "$lhs");
    let rhs = extract_typed_schema_expr(program, "$rhs");

    assert_eq!(lhs, "(Add (Leaf 0) (Add (Leaf 1) (Leaf 2)))");
    assert_eq!(rhs, lhs);
}

#[test]
fn typed_schema_extracts_mul_operands_in_stable_canonical_order() {
    let program = r#"
        (let $a (Leaf 0))
        (let $b (Leaf 1))
        (let $c (Leaf 2))
        (let $lhs (Mul (Mul $a $b) $c))
        (let $rhs (Mul $c (Mul $b $a)))
        (run-schedule (saturate (run)))
        (run-schedule (saturate canonical-order))
        "#;

    let lhs = extract_typed_schema_expr(program, "$lhs");
    let rhs = extract_typed_schema_expr(program, "$rhs");

    assert_eq!(lhs, "(Mul (Leaf 0) (Mul (Leaf 1) (Leaf 2)))");
    assert_eq!(rhs, lhs);
}

#[test]
fn typed_schema_rewrites_scalar_plus_tensor_add_to_explicit_broadcast() {
    assert_typed_schema_program_ok(
        r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $x (Leaf 0))
        (let $two (SConst 2))
        (let $c (Const $two $scalar-shape (F32)))
        (let $expr (Add $x $c))

        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (set (shape-of $c) $scalar-shape)
        (set (dtype-of $c) (F32))
        (set (scalar-of $c) $two)

        (scalar-shape $scalar-shape)

        (run-schedule (saturate (run)))

        (check (= $expr (Add $x (Broadcast $c $shape))))
        "#,
    );
}

#[test]
fn typed_schema_rewrites_scalar_times_tensor_mul_to_explicit_broadcast() {
    assert_typed_schema_program_ok(
        r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $x (Leaf 0))
        (let $two (SConst 2))
        (let $c (Const $two $scalar-shape (F32)))
        (let $expr (Mul $x $c))

        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (set (shape-of $c) $scalar-shape)
        (set (dtype-of $c) (F32))
        (set (scalar-of $c) $two)

        (scalar-shape $scalar-shape)

        (run-schedule (saturate (run)))

        (check (= $expr (Mul (Broadcast $c $shape) $x)))
        "#,
    );
}

#[test]
fn typed_schema_casts_mixed_dtype_scalar_before_broadcast() {
    assert_typed_schema_program_ok(
        r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $x (Leaf 0))
        (let $two (SConst 2))
        (let $c (Const $two $scalar-shape (F32)))
        (let $expr (Add $x $c))

        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F16))
        (set (shape-of $c) $scalar-shape)
        (set (dtype-of $c) (F32))
        (set (scalar-of $c) $two)

        (scalar-shape $scalar-shape)
        (requires-cast (F32) (F16))

        (run-schedule (saturate (run)))

        (check (= $expr (Add $x (Broadcast (Cast $c (F16)) $shape))))
        "#,
    );
}

#[test]
fn typed_schema_collects_like_terms_x_plus_x_to_two_x() {
    assert_typed_schema_program_ok(
        r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $x (Leaf 0))
        (let $one (SConst 1))
        (let $two (SConst 2))
        (let $sum (Add $x $x))

        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))

        (scalar-shape $scalar-shape)
        (scalar-one $one)
        (set (scalar-dtype $one) (F32))
        (scalar-add $one $one $two)

        (run-schedule (saturate (run)))

        (check (= $sum (Mul (Broadcast (Const $two $scalar-shape (F32)) $shape) $x)))
        "#,
    );
}

#[test]
fn typed_schema_collects_like_terms_x_times_two_plus_x_to_three_x() {
    assert_typed_schema_program_ok(
        r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $x (Leaf 0))
        (let $one (SConst 1))
        (let $two (SConst 2))
        (let $three (SConst 3))
        (let $c2 (Const $two $scalar-shape (F32)))
        (let $expr (Add (Mul $x $c2) $x))

        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (set (shape-of $c2) $scalar-shape)
        (set (dtype-of $c2) (F32))

        (scalar-shape $scalar-shape)
        (scalar-one $one)
        (set (scalar-dtype $one) (F32))
        (scalar-add $two $one $three)

        (run-schedule (saturate (run)))

        (check (= $expr (Mul (Broadcast (Const $three $scalar-shape (F32)) $shape) $x)))
        "#,
    );
}

#[test]
fn typed_schema_collects_nested_like_terms_to_six_x() {
    assert_typed_schema_program_ok(
        r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $x (Leaf 0))
        (let $one (SConst 1))
        (let $two (SConst 2))
        (let $three (SConst 3))
        (let $six (SConst 6))
        (let $c2a (Const $two $scalar-shape (F32)))
        (let $c2b (Const $two $scalar-shape (F32)))
        (let $lhs (Add (Mul $x $c2a) $x))
        (let $rhs (Add (Mul $x $c2b) $x))
        (let $expr (Add $lhs $rhs))

        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (set (shape-of $c2a) $scalar-shape)
        (set (dtype-of $c2a) (F32))
        (set (shape-of $c2b) $scalar-shape)
        (set (dtype-of $c2b) (F32))

        (scalar-shape $scalar-shape)
        (scalar-one $one)
        (set (scalar-dtype $one) (F32))
        (scalar-add $two $one $three)
        (scalar-add $three $three $six)

        (run-schedule (saturate (run)))

        (check (= $expr (Mul (Broadcast (Const $six $scalar-shape (F32)) $shape) $x)))
        "#,
    );
}

#[test]
fn typed_schema_folds_scalar_bias_chain_x_plus_one_two_three_to_x_plus_six() {
    assert_typed_schema_program_ok(
        r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $x (Leaf 0))
        (let $one (SConst 1))
        (let $two (SConst 2))
        (let $three (SConst 3))
        (let $five (SConst 5))
        (let $six (SConst 6))
        (let $c1 (Const $one $scalar-shape (F32)))
        (let $c2 (Const $two $scalar-shape (F32)))
        (let $c3 (Const $three $scalar-shape (F32)))
        (let $add01 (Add $x $c1))
        (let $add012 (Add $add01 $c2))
        (let $expr (Add $add012 $c3))

        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (set (shape-of $c1) $scalar-shape)
        (set (dtype-of $c1) (F32))
        (set (shape-of $c2) $scalar-shape)
        (set (dtype-of $c2) (F32))
        (set (shape-of $c3) $scalar-shape)
        (set (dtype-of $c3) (F32))
        (set (shape-of $add01) $shape)
        (set (dtype-of $add01) (F32))
        (set (shape-of $add012) $shape)
        (set (dtype-of $add012) (F32))

        (scalar-shape $scalar-shape)
        (scalar-add $one $two $three)
        (scalar-add $three $two $five)
        (scalar-add $five $one $six)
        (scalar-add $three $three $six)

        (run-schedule (saturate (run)))

        (check (= $expr (Add $x (Broadcast (Const $six $scalar-shape (F32)) $shape))))
        "#,
    );
}

#[test]
fn extract_uses_shared_session_for_multiple_roots() {
    let program = r#"
        (let $shape (ShapeCons 4 (ShapeNil)))
        (let $x (Leaf 0))
        (let $y (Leaf 1))
        (let $z (Leaf 2))
        (let $lhs (Add (Add $x $y) $z))
        (let $rhs (Add $z (Add $y $x)))
        (run-schedule (saturate (run)))
        (run-schedule (saturate canonical-order))
    "#;

    let output =
        extract_program_roots(program, &["$lhs", "$rhs"], &ExtractionConfig::default()).unwrap();

    assert_eq!(output.extracted_terms.len(), 2);
    assert_eq!(output.extracted_terms[0], output.extracted_terms[1]);
}

#[test]
fn extract_reports_when_bounded_schedule_hits_limit() {
    let program = r#"
        (let $x (Leaf 0))
        (let $expr (Neg (Neg $x)))
    "#;

    let output = extract_program_roots(
        program,
        &["$expr"],
        &ExtractionConfig {
            max_iterations: Some(0),
        },
    )
    .unwrap();

    assert!(output.report.hit_iteration_limit);
    assert_eq!(output.report.iterations, 0);
}

#[test]
fn extract_cost_prefers_scalar_broadcast_over_materialized_tensor_const() {
    let program = r#"
        (let $shape (ShapeCons 2 (ShapeCons 3 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $s2 (SConst 2))
        (let $scalar (Const $s2 $scalar-shape (F32)))
        (let $tensor (Const $s2 $shape (F32)))
        (let $via-broadcast (Broadcast $scalar $shape))

        (set (shape-of $scalar) $scalar-shape)
        (set (dtype-of $scalar) (F32))
        (set (shape-of $tensor) $shape)
        (set (dtype-of $tensor) (F32))
        (set (shape-of $via-broadcast) $shape)
        (set (dtype-of $via-broadcast) (F32))

        (scalar-shape $scalar-shape)
        (broadcast-ok $scalar-shape $shape)

        (union $tensor $via-broadcast)
    "#;

    let output =
        extract_program_roots(program, &["$tensor"], &ExtractionConfig::default()).unwrap();

    assert_eq!(
        output.extracted_terms[0],
        "(Broadcast (Const (SConst 2) (ShapeNil) (F32)) (ShapeCons 2 (ShapeCons 3 (ShapeNil))))"
    );
}

#[test]
fn manual_extract_inspection_cases_match_expected_forms() {
    let add_program = r#"
        (let $shape (ShapeCons 2 (ShapeCons 3 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $x (Leaf 0))
        (let $c (Const (SConst 2) $scalar-shape (F32)))
        (let $expr (Add $x $c))
        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (scalar-shape $scalar-shape)
        (run-schedule (saturate (run)))
    "#;
    let add_extracted = extract_typed_schema_expr(add_program, "$expr");
    assert_eq!(
        add_extracted,
        "(Add (Leaf 0) (Const (SConst 2) (ShapeNil) (F32)))"
    );

    let mul_program = r#"
        (let $shape (ShapeCons 2 (ShapeCons 3 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $x (Leaf 0))
        (let $c (Const (SConst 2) $scalar-shape (F32)))
        (let $expr (Mul $x $c))
        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (scalar-shape $scalar-shape)
        (run-schedule (saturate (run)))
    "#;
    let mul_extracted = extract_typed_schema_expr(mul_program, "$expr");
    assert_eq!(
        mul_extracted,
        "(Mul (Leaf 0) (Const (SConst 2) (ShapeNil) (F32)))"
    );

    let six_x_program = r#"
        (let $shape (ShapeCons 4 (ShapeCons 8 (ShapeNil))))
        (let $scalar-shape (ShapeNil))
        (let $x (Leaf 0))
        (let $one (SConst 1))
        (let $two (SConst 2))
        (let $three (SConst 3))
        (let $six (SConst 6))
        (let $c2a (Const $two $scalar-shape (F32)))
        (let $c2b (Const $two $scalar-shape (F32)))
        (let $lhs (Add (Mul $x $c2a) $x))
        (let $rhs (Add (Mul $x $c2b) $x))
        (let $expr (Add $lhs $rhs))
        (set (shape-of $x) $shape)
        (set (dtype-of $x) (F32))
        (set (shape-of $c2a) $scalar-shape)
        (set (dtype-of $c2a) (F32))
        (set (shape-of $c2b) $scalar-shape)
        (set (dtype-of $c2b) (F32))
        (scalar-shape $scalar-shape)
        (scalar-one $one)
        (set (scalar-dtype $one) (F32))
        (scalar-add $two $one $three)
        (scalar-add $three $three $six)
        (run-schedule (saturate (run)))
    "#;
    let six_x_extracted = extract_typed_schema_expr(six_x_program, "$expr");
    assert_eq!(
        six_x_extracted,
        "(Mul (Broadcast (Const (SConst 6) (ShapeNil) (F32)) (ShapeCons 4 (ShapeCons 8 (ShapeNil)))) (Leaf 0))"
    );

    let legal_reshape_program = r#"
        (let $src (ShapeCons 2 (ShapeCons 2 (ShapeNil))))
        (let $dst (ShapeCons 4 (ShapeNil)))
        (let $x (Leaf 0))
        (let $y (Leaf 1))
        (let $xr (Reshape $x $dst))
        (let $yr (Reshape $y $dst))
        (let $sum (Add $xr $yr))
        (set (shape-of $x) $src)
        (set (dtype-of $x) (F32))
        (set (shape-of $y) $src)
        (set (dtype-of $y) (F32))
        (reshape-ok $src $dst)
        (legal-add-shapes $src $src $src)
        (run-schedule (saturate (run)))
    "#;
    let legal_reshape_extracted = extract_typed_schema_expr(legal_reshape_program, "$sum");
    assert_eq!(
        legal_reshape_extracted,
        "(Reshape (Add (Leaf 0) (Leaf 1)) (ShapeCons 4 (ShapeNil)))"
    );

    let illegal_reshape_program = r#"
        (let $src (ShapeCons 2 (ShapeCons 2 (ShapeNil))))
        (let $dst (ShapeCons 4 (ShapeNil)))
        (let $x (Leaf 0))
        (let $y (Leaf 1))
        (let $xr (Reshape $x $dst))
        (let $yr (Reshape $y $dst))
        (let $sum (Add $xr $yr))
        (set (shape-of $x) $src)
        (set (dtype-of $x) (F32))
        (set (shape-of $y) $src)
        (set (dtype-of $y) (F32))
        (run-schedule (saturate (run)))
    "#;
    let illegal_reshape_extracted = extract_typed_schema_expr(illegal_reshape_program, "$sum");
    assert_eq!(
        illegal_reshape_extracted,
        "(Add (Reshape (Leaf 0) (ShapeCons 4 (ShapeNil))) (Reshape (Leaf 1) (ShapeCons 4 (ShapeNil))))"
    );
}

#[test]
fn fact_emitter_interns_symbolic_shapes_and_axes() {
    let mut g = HLIRGraph::new();
    let shape = vec![
        Dim::Add(Box::new(Dim::Sym(Symbol(0))), Box::new(Dim::Const(1))),
        Dim::Mul(Box::new(Dim::Sym(Symbol(1))), Box::new(Dim::Const(4))),
    ];
    let x = g.load(
        BufferId(0),
        TensorType::contiguous(shape.clone(), DType::F32),
    );
    let y = g.load(BufferId(1), TensorType::contiguous(shape, DType::F32));
    let _ = g.permute(x, vec![1, 0]);

    let facts = FactDatabase::from_graph(&g);
    let x_shape = facts.expr(x).unwrap().shape;
    let y_shape = facts.expr(y).unwrap().shape;

    assert_eq!(x_shape, y_shape);
    assert_eq!(facts.shape(x_shape).len(), 2);
    assert!(facts.axes.iter().any(|axes| axes.as_slice() == [1, 0]));

    match facts.dim(facts.shape(x_shape)[0]) {
        DimFact::Add(lhs, rhs) => {
            assert!(matches!(facts.dim(*lhs), DimFact::Sym(Symbol(0))));
            assert!(matches!(facts.dim(*rhs), DimFact::Const(1)));
        }
        other => panic!("expected symbolic add dim, got {other:?}"),
    }

    match facts.dim(facts.shape(x_shape)[1]) {
        DimFact::Mul(lhs, rhs) => {
            assert!(matches!(facts.dim(*lhs), DimFact::Sym(Symbol(1))));
            assert!(matches!(facts.dim(*rhs), DimFact::Const(4)));
        }
        other => panic!("expected symbolic mul dim, got {other:?}"),
    }
}

#[test]
fn fact_emitter_records_expr_shape_dtype_layout_and_rank() {
    let mut g = HLIRGraph::new();
    let scalar = g.constant(Scalar::F32(2.0), vec![], DType::F32);
    let broadcast = g.broadcast(scalar, vec![Dim::Const(2), Dim::Const(3)]);

    let facts = FactDatabase::from_graph(&g);
    let scalar_facts = facts.expr(scalar).unwrap();
    let broadcast_facts = facts.expr(broadcast).unwrap();

    assert_eq!(scalar_facts.dtype, DType::F32);
    assert_eq!(scalar_facts.layout, LayoutFact::Contiguous);
    assert_eq!(facts.rank_of[&scalar_facts.shape], 0);
    assert!(facts.scalar_shapes.contains(&scalar_facts.shape));

    assert_eq!(broadcast_facts.layout, LayoutFact::Broadcasted);
    assert_eq!(facts.rank_of[&broadcast_facts.shape], 2);
    assert!(facts
        .broadcast_ok
        .contains(&(scalar_facts.shape, broadcast_facts.shape)));
}

#[test]
fn fact_emitter_blocks_illegal_reshape_sinking_by_source_shape_facts() {
    let mut g = HLIRGraph::new();
    let x = g.load(BufferId(0), f32_ty(&[2, 2]));
    let y = g.load(BufferId(1), f32_ty(&[4]));
    let reshaped_x = g.reshape(x, vec![Dim::Const(4)]);
    let reshaped_y = g.reshape(y, vec![Dim::Const(4)]);
    let _ = g.binary(reshaped_x, reshaped_y, crate::core::hlir::Op::Add);

    let facts = FactDatabase::from_graph(&g);
    let x_shape = facts.expr(x).unwrap().shape;
    let y_shape = facts.expr(y).unwrap().shape;
    let reshaped_x_shape = facts.expr(reshaped_x).unwrap().shape;
    let reshaped_y_shape = facts.expr(reshaped_y).unwrap().shape;

    assert_eq!(reshaped_x_shape, reshaped_y_shape);
    assert!(facts.reshape_ok.contains(&(x_shape, reshaped_x_shape)));
    assert!(facts.reshape_ok.contains(&(y_shape, reshaped_y_shape)));
    assert!(facts
        .same_shape
        .contains(&(reshaped_x_shape, reshaped_y_shape)));
    assert!(facts.legal_add_shapes.contains(&(
        reshaped_x_shape,
        reshaped_y_shape,
        reshaped_x_shape
    )));
    assert!(!facts
        .legal_add_shapes
        .iter()
        .any(|(lhs, rhs, _)| *lhs == x_shape && *rhs == y_shape));
}

#[test]
fn fact_emitter_derives_expand_broadcast_permute_and_axes_compose_facts() {
    let mut g = HLIRGraph::new();
    let x = g.load(BufferId(0), f32_ty(&[1, 4, 3]));
    let y = g.load(BufferId(1), f32_ty(&[4, 3]));
    let expanded = g.expand(x, vec![Dim::Const(2), Dim::Const(4), Dim::Const(3)]);
    let broadcasted = g.broadcast(y, vec![Dim::Const(2), Dim::Const(4), Dim::Const(3)]);
    let permuted = g.permute(expanded, vec![1, 0, 2]);
    let _ = g.permute(permuted, vec![1, 2, 0]);

    let facts = FactDatabase::from_graph(&g);
    let x_shape = facts.expr(x).unwrap().shape;
    let y_shape = facts.expr(y).unwrap().shape;
    let expanded_shape = facts.expr(expanded).unwrap().shape;
    let broadcasted_shape = facts.expr(broadcasted).unwrap().shape;
    let permuted_shape = facts.expr(permuted).unwrap().shape;
    let axes_102 = axes_id(&facts, &[1, 0, 2]);
    let axes_120 = axes_id(&facts, &[1, 2, 0]);
    let axes_021 = axes_id(&facts, &[0, 2, 1]);
    let identity = axes_id(&facts, &[0, 1, 2]);

    assert!(facts.expand_ok.contains(&(x_shape, expanded_shape)));
    assert!(facts.broadcast_ok.contains(&(y_shape, expanded_shape)));
    assert_eq!(broadcasted_shape, expanded_shape);
    assert!(facts
        .legal_add_shapes
        .contains(&(expanded_shape, expanded_shape, expanded_shape)));
    assert!(facts
        .permute_ok
        .contains(&(expanded_shape, axes_102, permuted_shape)));
    assert!(facts.axes_compose.contains(&(axes_102, axes_120, axes_021)));
    assert!(facts.axes_identity.contains(&identity));
}

#[test]
fn structural_encoder_preserves_identity_shape_ops() {
    let mut g = HLIRGraph::new();
    let x = g.load(BufferId(0), f32_ty(&[4, 8]));
    let reshaped = g.reshape(x, vec![Dim::Const(4), Dim::Const(8)]);
    let permuted = g.permute(reshaped, vec![0, 1]);
    let expanded = g.expand(permuted, vec![Dim::Const(4), Dim::Const(8)]);

    let encoding = encode_algebraic(&g, &[expanded]);
    let root = encoding.expr(expanded).unwrap();
    let expand_args = call_args(root, "Expand");
    let permute_args = call_args(&expand_args[0], "Permute");
    let reshape_args = call_args(&permute_args[0], "Reshape");
    let leaf_args = call_args(&reshape_args[0], "Leaf");
    let axes_01 = axes_id(&encoding.facts, &[0, 1]);
    let reshaped_shape = encoding.facts.expr(reshaped).unwrap().shape;
    let expanded_shape = encoding.facts.expr(expanded).unwrap().shape;

    assert_eq!(
        int_lit(&leaf_args[0]),
        encoding.leaf_id(&LeafRef::Load(x)).unwrap()
    );
    assert_eq!(
        render_expr(&reshape_args[1]),
        render_expr(encoding.shape_term(reshaped_shape).unwrap())
    );
    assert_eq!(
        render_expr(&permute_args[1]),
        render_expr(encoding.axes_term(axes_01).unwrap())
    );
    assert_eq!(
        render_expr(&expand_args[1]),
        render_expr(encoding.shape_term(expanded_shape).unwrap())
    );
}

#[test]
fn structural_encoder_keeps_scalar_tensor_add_as_scalar_const_term() {
    let mut g = HLIRGraph::new();
    let x = g.load(BufferId(0), f32_ty(&[4, 8]));
    let scalar = g.constant(Scalar::F32(2.0), vec![], DType::F32);
    let root = g.binary(x, scalar, crate::core::hlir::Op::Add);

    let encoding = encode_algebraic(&g, &[root]);
    let root_term = encoding.expr(root).unwrap();
    let add_args = call_args(root_term, "Add");
    let lhs_leaf = call_args(&add_args[0], "Leaf");
    let const_args = call_args(&add_args[1], "Const");
    let scalar_args = call_args(&const_args[0], "SConst");

    assert_eq!(
        int_lit(&lhs_leaf[0]),
        encoding.leaf_id(&LeafRef::Load(x)).unwrap()
    );
    assert_eq!(
        int_lit(&scalar_args[0]),
        i64::try_from(encoding.scalar_id_for_const(scalar).unwrap().0).unwrap()
    );
    assert_eq!(render_expr(&const_args[1]), "(ShapeNil)");
    let dtype_args = call_args(&const_args[2], "F32");
    assert!(dtype_args.is_empty());
    assert!(!contains_call(root_term, "Broadcast"));
    assert!(!contains_call(root_term, "Expand"));
}

#[test]
fn structural_encoder_represents_barrier_outputs_as_leaf_refs() {
    let mut g = HLIRGraph::new();
    let x = g.load(BufferId(0), f32_ty(&[4, 8]));
    let y = g.load(BufferId(1), f32_ty(&[4]));
    let reduced = g.reduce(x, vec![1], ReduceOp::Sum, false);
    let root = g.binary(reduced, y, crate::core::hlir::Op::Add);

    let encoding = encode_algebraic(&g, &[root]);
    let root_term = encoding.expr(root).unwrap();
    let add_args = call_args(root_term, "Add");
    let barrier_leaf = call_args(&add_args[0], "Leaf");
    let load_leaf = call_args(&add_args[1], "Leaf");

    assert!(encoding.expr(reduced).is_none());
    assert_eq!(
        int_lit(&barrier_leaf[0]),
        encoding.leaf_id(&LeafRef::BarrierOutput(reduced)).unwrap()
    );
    assert_eq!(
        int_lit(&load_leaf[0]),
        encoding.leaf_id(&LeafRef::Load(y)).unwrap()
    );
    assert!(!contains_call(root_term, "Reduce"));
}

#[test]
fn structural_decode_rebuilds_reduce_with_simplified_algebraic_input() {
    let mut g = HLIRGraph::new();
    let x = g.load(BufferId(0), f32_ty(&[4, 8]));
    let zero = g.constant(
        Scalar::F32(0.0),
        vec![Dim::Const(4), Dim::Const(8)],
        DType::F32,
    );
    let add = g.binary(x, zero, crate::core::hlir::Op::Add);
    let reduce = g.reduce(add, vec![1], ReduceOp::Sum, false);

    let encoding = encode_algebraic(&g, &[reduce]);
    let leaf_x = encoding.leaf_id(&LeafRef::Load(x)).unwrap();
    let extracted = vec![(add, format!("(Leaf {leaf_x})"))];

    let (decoded, remap) = decode_with_extracted(&g, &[reduce], &encoding, &extracted).unwrap();
    let out_root = remap[&reduce];

    match &decoded.node(out_root).op {
        Op::Reduce { input, .. } => {
            assert!(matches!(decoded.node(*input).op, Op::Load { .. }));
        }
        op => panic!("expected Reduce root, got {op:?}"),
    }
}

#[test]
fn structural_decode_remaps_barrier_outputs_for_algebraic_consumers() {
    let mut g = HLIRGraph::new();
    let x = g.load(BufferId(0), f32_ty(&[4, 8]));
    let y = g.load(BufferId(1), f32_ty(&[4]));
    let zero = g.constant(
        Scalar::F32(0.0),
        vec![Dim::Const(4), Dim::Const(8)],
        DType::F32,
    );
    let add_in = g.binary(x, zero, crate::core::hlir::Op::Add);
    let reduce = g.reduce(add_in, vec![1], ReduceOp::Sum, false);
    let root = g.binary(reduce, y, crate::core::hlir::Op::Add);

    let encoding = encode_algebraic(&g, &[root]);
    let root_term = render_expr(encoding.expr(root).unwrap());
    let leaf_x = encoding.leaf_id(&LeafRef::Load(x)).unwrap();
    let extracted = vec![(add_in, format!("(Leaf {leaf_x})")), (root, root_term)];

    let (decoded, remap) = decode_with_extracted(&g, &[root], &encoding, &extracted).unwrap();
    let out_root = remap[&root];

    let (lhs, rhs) = match &decoded.node(out_root).op {
        Op::Add(lhs, rhs) => (*lhs, *rhs),
        op => panic!("expected Add root, got {op:?}"),
    };
    assert!(matches!(decoded.node(rhs).op, Op::Load { .. }));
    match &decoded.node(lhs).op {
        Op::Reduce { input, .. } => assert!(matches!(decoded.node(*input).op, Op::Load { .. })),
        op => panic!("expected remapped Reduce on lhs, got {op:?}"),
    }
}

fn axes_id(facts: &FactDatabase, expected: &[usize]) -> AxesId {
    let index = facts
        .axes
        .iter()
        .position(|axes| axes.as_slice() == expected)
        .unwrap_or_else(|| panic!("missing axes fact for {expected:?}"));
    AxesId(index)
}

fn call_args<'a>(expr: &'a EggExpr, expected: &str) -> &'a [EggExpr] {
    match expr {
        EggExpr::Call(_, name, args) if name.to_string() == expected => args,
        other => panic!("expected {expected} call, got {}", render_expr(other)),
    }
}

fn int_lit(expr: &EggExpr) -> i64 {
    match expr {
        EggExpr::Lit(_, egglog::ast::Literal::Int(value)) => *value,
        other => panic!("expected integer literal, got {}", render_expr(other)),
    }
}

fn contains_call(expr: &EggExpr, expected: &str) -> bool {
    match expr {
        EggExpr::Call(_, name, args) => {
            name.to_string() == expected || args.iter().any(|arg| contains_call(arg, expected))
        }
        EggExpr::Lit(_, _) => false,
        other => panic!("unexpected egglog expression variant: {other:?}"),
    }
}

fn render_expr(expr: &EggExpr) -> String {
    match expr {
        EggExpr::Lit(_, egglog::ast::Literal::Int(value)) => value.to_string(),
        EggExpr::Call(_, name, args) if args.is_empty() => format!("({name})"),
        EggExpr::Call(_, name, args) => {
            let rendered_args = args.iter().map(render_expr).collect::<Vec<_>>().join(" ");
            format!("({name} {rendered_args})")
        }
        other => panic!("unexpected egglog expression variant: {other:?}"),
    }
}
