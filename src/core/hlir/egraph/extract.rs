//! Extraction and root coordination for the typed egglog pipeline.

use super::cost::{CanonicalExprCost, CanonicalExprCostModel};
use super::encode::{encode_algebraic, render_expr, StructuralEncoding};
use egglog::extract::{CostModel, Extractor};

use super::super::{DType, HLIRGraph, NodeId, Scalar};

const TYPED_SCHEMA: &str = include_str!("schema.egg");

#[derive(Clone, Debug)]
pub struct ExtractionConfig {
    pub max_iterations: Option<usize>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            max_iterations: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractionReport {
    pub iterations: usize,
    pub hit_iteration_limit: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractionOutput {
    pub extracted_terms: Vec<String>,
    pub report: ExtractionReport,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredExtractionOutput {
    pub extracted_terms: Vec<(NodeId, String)>,
    pub report: ExtractionReport,
}

#[allow(dead_code)]
pub fn extract_program_roots(
    program: &str,
    roots: &[&str],
    config: &ExtractionConfig,
) -> Result<ExtractionOutput, String> {
    let mut egraph = egglog::EGraph::default();
    egraph
        .parse_and_run_program(None, TYPED_SCHEMA)
        .map_err(|err| format!("failed to parse typed schema: {err}"))?;
    egraph
        .parse_and_run_program(None, program)
        .map_err(|err| format!("failed to run input program: {err}"))?;

    let report = match config.max_iterations {
        None => {
            egraph
                .parse_and_run_program(None, "(run-schedule (saturate (run)))")
                .map_err(|err| format!("failed to saturate e-graph: {err}"))?;
            ExtractionReport {
                iterations: 0,
                hit_iteration_limit: false,
            }
        }
        Some(limit) => run_bounded(&mut egraph, roots, limit)?,
    };

    let extracted_terms =
        extract_roots_with_shared_session(&mut egraph, roots, CanonicalExprCostModel)?;
    Ok(ExtractionOutput {
        extracted_terms,
        report,
    })
}

pub fn extract_structural_roots(
    graph: &HLIRGraph,
    roots: &[NodeId],
    config: &ExtractionConfig,
) -> Result<(StructuralEncoding, StructuredExtractionOutput), String> {
    let mut encoding = encode_algebraic(graph, roots);
    let mut egraph = egglog::EGraph::default();
    egraph
        .parse_and_run_program(None, TYPED_SCHEMA)
        .map_err(|err| format!("failed to parse typed schema: {err}"))?;

    inject_encoding_facts(&mut egraph, &mut encoding)?;

    let region_roots = encoding
        .plan
        .region_roots
        .iter()
        .filter(|id| encoding.expr(**id).is_some())
        .copied()
        .collect::<Vec<_>>();
    let root_exprs = region_roots
        .iter()
        .map(|id| render_expr(encoding.expr(*id).unwrap()))
        .collect::<Vec<_>>();
    let root_refs = root_exprs.iter().map(String::as_str).collect::<Vec<_>>();

    let report = match config.max_iterations {
        None => {
            egraph
                .parse_and_run_program(None, "(run-schedule (saturate (run)))")
                .map_err(|err| format!("failed to saturate e-graph: {err}"))?;
            ExtractionReport {
                iterations: 0,
                hit_iteration_limit: false,
            }
        }
        Some(limit) => run_bounded(&mut egraph, &root_refs, limit)?,
    };

    let extracted =
        extract_roots_with_shared_session(&mut egraph, &root_refs, CanonicalExprCostModel)?;
    let extracted_terms = region_roots.into_iter().zip(extracted).collect::<Vec<_>>();

    Ok((
        encoding,
        StructuredExtractionOutput {
            extracted_terms,
            report,
        },
    ))
}

fn run_bounded(
    egraph: &mut egglog::EGraph,
    roots: &[&str],
    limit: usize,
) -> Result<ExtractionReport, String> {
    let mut previous: Option<Vec<String>> = None;
    for iteration in 0..limit {
        egraph
            .parse_and_run_program(None, "(run)")
            .map_err(|err| format!("failed bounded run iteration {}: {err}", iteration + 1))?;
        let extracted = extract_roots_with_shared_session(egraph, roots, CanonicalExprCostModel)?;
        if previous.as_ref().is_some_and(|prev| prev == &extracted) {
            return Ok(ExtractionReport {
                iterations: iteration + 1,
                hit_iteration_limit: false,
            });
        }
        previous = Some(extracted);
    }

    Ok(ExtractionReport {
        iterations: limit,
        hit_iteration_limit: true,
    })
}

fn extract_roots_with_shared_session<M>(
    egraph: &mut egglog::EGraph,
    roots: &[&str],
    model: M,
) -> Result<Vec<String>, String>
where
    M: CostModel<CanonicalExprCost> + 'static,
{
    let mut parser = egglog::ast::Parser::default();
    let mut evals = Vec::with_capacity(roots.len());
    for root in roots {
        let expr = parser
            .get_expr_from_string(None, root)
            .map_err(|err| format!("failed to parse root `{root}`: {err}"))?;
        let (sort, value) = egraph
            .eval_expr(&expr)
            .map_err(|err| format!("failed to evaluate root `{root}`: {err}"))?;
        evals.push((sort, value));
    }

    let root_sorts = evals
        .iter()
        .map(|(sort, _)| sort.clone())
        .collect::<Vec<_>>();
    let extractor = Extractor::compute_costs_from_rootsorts(Some(root_sorts), egraph, model);
    let mut termdag = egglog::TermDag::default();
    let mut out = Vec::with_capacity(evals.len());
    for (sort, value) in evals {
        let (_, term_id) = extractor
            .extract_best_with_sort(egraph, &mut termdag, value, sort)
            .ok_or_else(|| String::from("failed extraction"))?;
        out.push(termdag.to_string(term_id));
    }
    Ok(out)
}

fn inject_encoding_facts(
    egraph: &mut egglog::EGraph,
    encoding: &mut StructuralEncoding,
) -> Result<(), String> {
    for (id, expr) in encoding.expr_terms_sorted() {
        let expr_text = render_expr(expr);
        let shape = encoding
            .facts
            .expr(id)
            .map(|facts| facts.shape)
            .ok_or_else(|| format!("missing shape fact for expr {expr_text}"))?;
        let dtype = encoding
            .facts
            .expr(id)
            .map(|facts| facts.dtype)
            .ok_or_else(|| format!("missing dtype fact for expr {expr_text}"))?;
        let layout = encoding
            .facts
            .expr(id)
            .map(|facts| facts.layout)
            .ok_or_else(|| format!("missing layout fact for expr {expr_text}"))?;

        let shape_expr = render_expr(
            encoding
                .shape_term(shape)
                .ok_or_else(|| format!("missing shape term for {shape:?}"))?,
        );
        let dtype_expr = dtype_ctor(dtype);
        let layout_expr = layout_ctor(layout);

        egraph
            .parse_and_run_program(None, &format!("(set (shape-of {expr_text}) {shape_expr})"))
            .map_err(|err| format!("failed to set shape-of: {err}"))?;
        egraph
            .parse_and_run_program(None, &format!("(set (dtype-of {expr_text}) {dtype_expr})"))
            .map_err(|err| format!("failed to set dtype-of: {err}"))?;
        egraph
            .parse_and_run_program(
                None,
                &format!("(set (layout-of {expr_text}) {layout_expr})"),
            )
            .map_err(|err| format!("failed to set layout-of: {err}"))?;
    }

    for scalar_id in encoding.scalar_arena.scalar_ids() {
        let scalar_term = render_expr(
            encoding
                .scalar_term(scalar_id)
                .ok_or_else(|| format!("missing scalar term for {scalar_id:?}"))?,
        );
        let scalar = encoding.scalar_arena.value(scalar_id);
        let dtype_expr = dtype_ctor(scalar.dtype);
        egraph
            .parse_and_run_program(
                None,
                &format!("(set (scalar-dtype {scalar_term}) {dtype_expr})"),
            )
            .map_err(|err| format!("failed to set scalar-dtype: {err}"))?;
        egraph
            .parse_and_run_program(
                None,
                &format!(
                    "(set (scalar-value-id {scalar_term}) {})",
                    i64::try_from(scalar_id.0).map_err(|_| "scalar id overflow")?
                ),
            )
            .map_err(|err| format!("failed to set scalar-value-id: {err}"))?;
        if encoding.scalar_arena.is_zero(scalar_id) {
            egraph
                .parse_and_run_program(None, &format!("(scalar-zero {scalar_term})"))
                .map_err(|err| format!("failed to assert scalar-zero: {err}"))?;
        }
        if encoding.scalar_arena.is_one(scalar_id) {
            egraph
                .parse_and_run_program(None, &format!("(scalar-one {scalar_term})"))
                .map_err(|err| format!("failed to assert scalar-one: {err}"))?;
        }
    }

    emit_pair_shape_relations(egraph, "same-shape", &encoding.facts.same_shape, encoding)?;
    emit_pair_shape_relations(egraph, "same-numel", &encoding.facts.same_numel, encoding)?;
    emit_pair_shape_relations(egraph, "reshape-ok", &encoding.facts.reshape_ok, encoding)?;
    emit_pair_shape_relations(
        egraph,
        "broadcast-ok",
        &encoding.facts.broadcast_ok,
        encoding,
    )?;
    emit_pair_shape_relations(egraph, "expand-ok", &encoding.facts.expand_ok, encoding)?;
    emit_binary_shape_relations(
        egraph,
        "legal-add-shapes",
        &encoding.facts.legal_add_shapes,
        encoding,
    )?;
    emit_binary_shape_relations(
        egraph,
        "legal-mul-shapes",
        &encoding.facts.legal_mul_shapes,
        encoding,
    )?;
    emit_binary_shape_relations(
        egraph,
        "legal-max-shapes",
        &encoding.facts.legal_max_shapes,
        encoding,
    )?;
    emit_binary_shape_relations(
        egraph,
        "legal-min-shapes",
        &encoding.facts.legal_min_shapes,
        encoding,
    )?;
    emit_permute_relations(egraph, "permute-ok", &encoding.facts.permute_ok, encoding)?;

    for (lhs, rhs, out) in &encoding.facts.axes_compose {
        let lhs_term = render_expr(
            encoding
                .axes_term(*lhs)
                .ok_or_else(|| format!("missing axes term for {lhs:?}"))?,
        );
        let rhs_term = render_expr(
            encoding
                .axes_term(*rhs)
                .ok_or_else(|| format!("missing axes term for {rhs:?}"))?,
        );
        let out_term = render_expr(
            encoding
                .axes_term(*out)
                .ok_or_else(|| format!("missing axes term for {out:?}"))?,
        );
        egraph
            .parse_and_run_program(
                None,
                &format!("(axes-compose {lhs_term} {rhs_term} {out_term})"),
            )
            .map_err(|err| format!("failed to assert axes-compose: {err}"))?;
    }

    for axes in &encoding.facts.axes_identity {
        let axes_term = render_expr(
            encoding
                .axes_term(*axes)
                .ok_or_else(|| format!("missing axes term for {axes:?}"))?,
        );
        egraph
            .parse_and_run_program(None, &format!("(axes-identity {axes_term})"))
            .map_err(|err| format!("failed to assert axes-identity: {err}"))?;
    }

    for shape_id in &encoding.facts.scalar_shapes {
        let shape = render_expr(
            encoding
                .shape_term(*shape_id)
                .ok_or_else(|| format!("missing scalar shape term for {shape_id:?}"))?,
        );
        egraph
            .parse_and_run_program(None, &format!("(scalar-shape {shape})"))
            .map_err(|err| format!("failed to assert scalar-shape: {err}"))?;
    }

    for (shape_id, rank) in &encoding.facts.rank_of {
        let shape = render_expr(
            encoding
                .shape_term(*shape_id)
                .ok_or_else(|| format!("missing rank shape term for {shape_id:?}"))?,
        );
        egraph
            .parse_and_run_program(None, &format!("(set (rank-of {shape}) {rank})"))
            .map_err(|err| format!("failed to set rank-of: {err}"))?;
    }

    let mut seen_scalar_meta = encoding.scalar_arena.scalar_ids();

    let mut dtypes = encoding
        .facts
        .expr_facts
        .values()
        .map(|facts| facts.dtype)
        .collect::<Vec<_>>();
    dtypes.sort_by_key(|dtype| *dtype as u8);
    dtypes.dedup();

    for dtype in dtypes {
        let zero = encoding.scalar_arena.intern(zero_scalar(dtype));
        let one = encoding.scalar_arena.intern(one_scalar(dtype));
        ensure_scalar_meta(egraph, zero, &encoding.scalar_arena, &mut seen_scalar_meta)?;
        ensure_scalar_meta(egraph, one, &encoding.scalar_arena, &mut seen_scalar_meta)?;
    }

    for _ in 0..2 {
        let scalar_ids = encoding.scalar_arena.scalar_ids();
        for &lhs in &scalar_ids {
            for &rhs in &scalar_ids {
                let lhs_term = scalar_term_for_id(lhs, encoding);
                let rhs_term = scalar_term_for_id(rhs, encoding);
                if encoding.scalar_arena.value(lhs).dtype != encoding.scalar_arena.value(rhs).dtype
                {
                    egraph
                        .parse_and_run_program(
                            None,
                            &format!(
                                "(requires-cast {} {})",
                                dtype_ctor(encoding.scalar_arena.value(lhs).dtype),
                                dtype_ctor(encoding.scalar_arena.value(rhs).dtype)
                            ),
                        )
                        .map_err(|err| format!("failed to assert requires-cast: {err}"))?;
                    let rhs_dtype = encoding.scalar_arena.value(rhs).dtype;
                    let cast_scalar = encoding.scalar_arena.cast(lhs, rhs_dtype);
                    ensure_scalar_meta(
                        egraph,
                        cast_scalar,
                        &encoding.scalar_arena,
                        &mut seen_scalar_meta,
                    )?;
                    continue;
                }

                if encoding.scalar_arena.value(lhs).dtype == DType::Bool {
                    continue;
                }

                let sum = encoding.scalar_arena.add(lhs, rhs);
                let prod = encoding.scalar_arena.mul(lhs, rhs);
                let neg = encoding.scalar_arena.neg(lhs);
                ensure_scalar_meta(egraph, sum, &encoding.scalar_arena, &mut seen_scalar_meta)?;
                ensure_scalar_meta(egraph, prod, &encoding.scalar_arena, &mut seen_scalar_meta)?;
                ensure_scalar_meta(egraph, neg, &encoding.scalar_arena, &mut seen_scalar_meta)?;

                let sum_term = scalar_term_for_id(sum, encoding);
                let prod_term = scalar_term_for_id(prod, encoding);
                let neg_term = scalar_term_for_id(neg, encoding);

                egraph
                    .parse_and_run_program(
                        None,
                        &format!("(scalar-add {lhs_term} {rhs_term} {sum_term})"),
                    )
                    .map_err(|err| format!("failed to assert scalar-add: {err}"))?;
                egraph
                    .parse_and_run_program(
                        None,
                        &format!("(scalar-mul {lhs_term} {rhs_term} {prod_term})"),
                    )
                    .map_err(|err| format!("failed to assert scalar-mul: {err}"))?;
                egraph
                    .parse_and_run_program(None, &format!("(scalar-neg {lhs_term} {neg_term})"))
                    .map_err(|err| format!("failed to assert scalar-neg: {err}"))?;
            }
        }
    }

    Ok(())
}

fn scalar_term_for_id(id: super::facts::ScalarId, encoding: &StructuralEncoding) -> String {
    if let Some(expr) = encoding.scalar_term(id) {
        return render_expr(expr);
    }
    format!("(SConst {})", id.0)
}

fn ensure_scalar_meta(
    egraph: &mut egglog::EGraph,
    id: super::facts::ScalarId,
    arena: &super::facts::ScalarArena,
    seen: &mut Vec<super::facts::ScalarId>,
) -> Result<(), String> {
    if seen.contains(&id) {
        return Ok(());
    }
    seen.push(id);

    let scalar_term = format!("(SConst {})", id.0);
    let scalar = arena.value(id);
    egraph
        .parse_and_run_program(
            None,
            &format!(
                "(set (scalar-dtype {scalar_term}) {})",
                dtype_ctor(scalar.dtype)
            ),
        )
        .map_err(|err| format!("failed to set closure scalar-dtype: {err}"))?;
    egraph
        .parse_and_run_program(
            None,
            &format!("(set (scalar-value-id {scalar_term}) {})", id.0),
        )
        .map_err(|err| format!("failed to set closure scalar-value-id: {err}"))?;
    if arena.is_zero(id) {
        egraph
            .parse_and_run_program(None, &format!("(scalar-zero {scalar_term})"))
            .map_err(|err| format!("failed to set closure scalar-zero: {err}"))?;
    }
    if arena.is_one(id) {
        egraph
            .parse_and_run_program(None, &format!("(scalar-one {scalar_term})"))
            .map_err(|err| format!("failed to set closure scalar-one: {err}"))?;
    }
    Ok(())
}

fn emit_permute_relations(
    egraph: &mut egglog::EGraph,
    rel: &str,
    triples: &std::collections::HashSet<(
        super::facts::ShapeId,
        super::facts::AxesId,
        super::facts::ShapeId,
    )>,
    encoding: &StructuralEncoding,
) -> Result<(), String> {
    for (s0, axes, s1) in triples {
        let s0_term = render_expr(
            encoding
                .shape_term(*s0)
                .ok_or_else(|| format!("missing shape term for {s0:?}"))?,
        );
        let axes_term = render_expr(
            encoding
                .axes_term(*axes)
                .ok_or_else(|| format!("missing axes term for {axes:?}"))?,
        );
        let s1_term = render_expr(
            encoding
                .shape_term(*s1)
                .ok_or_else(|| format!("missing shape term for {s1:?}"))?,
        );
        egraph
            .parse_and_run_program(None, &format!("({rel} {s0_term} {axes_term} {s1_term})"))
            .map_err(|err| format!("failed to assert {rel}: {err}"))?;
    }
    Ok(())
}

fn dtype_ctor(dtype: super::super::DType) -> &'static str {
    match dtype {
        super::super::DType::F32 => "(F32)",
        super::super::DType::F16 => "(F16)",
        super::super::DType::BF16 => "(BF16)",
        super::super::DType::F64 => "(F64)",
        super::super::DType::I8 => "(I8)",
        super::super::DType::I16 => "(I16)",
        super::super::DType::I32 => "(I32)",
        super::super::DType::I64 => "(I64)",
        super::super::DType::U8 => "(U8)",
        super::super::DType::U16 => "(U16)",
        super::super::DType::U32 => "(U32)",
        super::super::DType::U64 => "(U64)",
        super::super::DType::Bool => "(Bool)",
    }
}

fn layout_ctor(layout: super::facts::LayoutFact) -> &'static str {
    match layout {
        super::facts::LayoutFact::Contiguous => "(Contiguous)",
        super::facts::LayoutFact::Broadcasted => "(Broadcasted)",
        super::facts::LayoutFact::Strided => "(Strided)",
    }
}

fn zero_scalar(dtype: DType) -> Scalar {
    match dtype {
        DType::F32 => Scalar::F32(0.0),
        DType::F16 => Scalar::F16(0),
        DType::BF16 => Scalar::BF16(0),
        DType::F64 => Scalar::F64(0.0),
        DType::I8 => Scalar::I8(0),
        DType::I16 => Scalar::I16(0),
        DType::I32 => Scalar::I32(0),
        DType::I64 => Scalar::I64(0),
        DType::U8 => Scalar::U8(0),
        DType::U16 => Scalar::U16(0),
        DType::U32 => Scalar::U32(0),
        DType::U64 => Scalar::U64(0),
        DType::Bool => Scalar::Bool(false),
    }
}

fn one_scalar(dtype: DType) -> Scalar {
    match dtype {
        DType::F32 => Scalar::F32(1.0),
        DType::F16 => Scalar::F16(0x3C00),
        DType::BF16 => Scalar::BF16(0x3F80),
        DType::F64 => Scalar::F64(1.0),
        DType::I8 => Scalar::I8(1),
        DType::I16 => Scalar::I16(1),
        DType::I32 => Scalar::I32(1),
        DType::I64 => Scalar::I64(1),
        DType::U8 => Scalar::U8(1),
        DType::U16 => Scalar::U16(1),
        DType::U32 => Scalar::U32(1),
        DType::U64 => Scalar::U64(1),
        DType::Bool => Scalar::Bool(true),
    }
}

fn emit_pair_shape_relations(
    egraph: &mut egglog::EGraph,
    rel: &str,
    pairs: &std::collections::HashSet<(super::facts::ShapeId, super::facts::ShapeId)>,
    encoding: &StructuralEncoding,
) -> Result<(), String> {
    for (a, b) in pairs {
        let sa = render_expr(
            encoding
                .shape_term(*a)
                .ok_or_else(|| format!("missing shape term for {a:?}"))?,
        );
        let sb = render_expr(
            encoding
                .shape_term(*b)
                .ok_or_else(|| format!("missing shape term for {b:?}"))?,
        );
        egraph
            .parse_and_run_program(None, &format!("({rel} {sa} {sb})"))
            .map_err(|err| format!("failed to assert {rel}: {err}"))?;
    }
    Ok(())
}

fn emit_binary_shape_relations(
    egraph: &mut egglog::EGraph,
    rel: &str,
    triples: &std::collections::HashSet<(
        super::facts::ShapeId,
        super::facts::ShapeId,
        super::facts::ShapeId,
    )>,
    encoding: &StructuralEncoding,
) -> Result<(), String> {
    for (a, b, c) in triples {
        let sa = render_expr(
            encoding
                .shape_term(*a)
                .ok_or_else(|| format!("missing shape term for {a:?}"))?,
        );
        let sb = render_expr(
            encoding
                .shape_term(*b)
                .ok_or_else(|| format!("missing shape term for {b:?}"))?,
        );
        let sc = render_expr(
            encoding
                .shape_term(*c)
                .ok_or_else(|| format!("missing shape term for {c:?}"))?,
        );
        egraph
            .parse_and_run_program(None, &format!("({rel} {sa} {sb} {sc})"))
            .map_err(|err| format!("failed to assert {rel}: {err}"))?;
    }
    Ok(())
}
