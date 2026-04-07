//! E-graph algebraic simplification and reshape sinking via egglog.
//!
//! Encodes the algebraic subset of the HLIR (elementwise ops, Reshape, Cast)
//! into egglog terms, runs saturation with rewrite rules, extracts the
//! optimal (smallest) term, and decodes back to an HLIRGraph.
//!
//! Non-algebraic ops (Reduce, Slice, Concat, etc.) are
//! treated as opaque barriers — their inputs are optimized recursively but
//! the ops themselves are not represented in the e-graph.

use std::collections::HashMap;

use super::dim::Dim;
use super::graph::HLIRGraph;
use super::op::Op;
use super::types::{DType, NodeId, Scalar, TensorType};

type EggExpr = egglog::ast::Expr;

/// Leaf metadata stored on the Rust side, keyed by integer id used in egglog.
#[derive(Clone, Debug)]
enum LeafKind {
    /// An HLIR node that's already been rebuilt into the output graph.
    Node(NodeId),
    /// A Const whose original metadata we preserve.
    Const { ty: TensorType, value: Scalar },
}

const EGGLOG_SCHEMA: &str = include_str!("algebra.egg");

/// Run egglog algebraic simplification on the HLIR graph.
///
/// Returns a new graph and a mapping from old root NodeIds to new ones.
pub fn egglog_algebraic(
    graph: &HLIRGraph,
    roots: &[NodeId],
) -> (HLIRGraph, HashMap<NodeId, NodeId>) {
    let mut ctx = EgglogContext::new(graph);

    // Recursively encode each root, treating non-algebraic ops as barriers.
    // For barrier roots, register them as barriers; for algebraic roots, encode directly.
    for &id in roots {
        ctx.encode(id);
    }

    // Collect all terms that need extraction: algebraic user roots + algebraic sub-roots for barriers.
    let mut extract_entries: Vec<(EggExpr, NodeId)> = Vec::new();

    // User roots that are algebraic (have egglog terms).
    for &id in roots {
        if let Some(term) = ctx.term_memo.get(&id)
            && !EgglogContext::is_barrier_encoded(&ctx, id)
        {
            extract_entries.push((term.clone(), id));
        }
    }
    // Algebraic sub-roots needed by barriers.
    for (term, src_id) in &ctx.algebraic_roots {
        extract_entries.push((term.clone(), *src_id));
    }

    if extract_entries.is_empty() {
        // Nothing algebraic — just rebuild barriers directly.
        let mut out = ctx.out;
        let remap =
            rebuild_all_barriers(graph, roots, &ctx.barriers, &[], &ctx.built_map, &mut out);
        return (out, remap);
    }

    // Run egglog.
    let mut egraph = egglog::EGraph::default();
    if egraph.parse_and_run_program(None, EGGLOG_SCHEMA).is_err() {
        let identity_map = roots.iter().map(|&id| (id, id)).collect();
        return (graph.clone(), identity_map);
    }

    let mut extracted_inputs: Vec<(NodeId, egglog::ArcSort, egglog::Value)> =
        Vec::with_capacity(extract_entries.len());
    for (expr, src_id) in &extract_entries {
        match egraph.eval_expr(expr) {
            Ok((sort, value)) => extracted_inputs.push((*src_id, sort, value)),
            Err(_) => {
                let identity_map = roots.iter().map(|&id| (id, id)).collect();
                return (graph.clone(), identity_map);
            }
        }
    }

    if egraph.parse_and_run_program(None, "(run 10)").is_err() {
        let identity_map = roots.iter().map(|&id| (id, id)).collect();
        return (graph.clone(), identity_map);
    }

    let extracted: Vec<(NodeId, egglog::TermDag, egglog::Term)> = match extracted_inputs
        .into_iter()
        .map(|(src_id, sort, value)| {
            egraph
                .extract_value(&sort, value)
                .map(|(termdag, term, _)| (src_id, termdag, term))
        })
        .collect::<Result<_, _>>()
    {
        Err(_) => {
            let identity_map = roots.iter().map(|&id| (id, id)).collect();
            return (graph.clone(), identity_map);
        }
        Ok(extracted) => extracted,
    };

    let mut out = ctx.out;
    let mut decode_memo: HashMap<egglog::TermId, NodeId> = HashMap::new();
    let mut decoded_map: HashMap<NodeId, NodeId> = HashMap::new();

    // Decode all extracted terms.
    for (src_id, termdag, term) in &extracted {
        let new_id = decode_term(
            termdag,
            term,
            &ctx.leaves,
            &ctx.shape_meta,
            &ctx.dtype_meta,
            &ctx.permute_meta,
            &mut out,
            &mut decode_memo,
        );
        decoded_map.insert(*src_id, new_id);
    }

    // Build alg_decoded vector in the order of algebraic_roots.
    let alg_decoded: Vec<NodeId> = ctx
        .algebraic_roots
        .iter()
        .map(|(_, src_id)| {
            decoded_map
                .get(src_id)
                .or_else(|| ctx.built_map.get(src_id))
                .copied()
                .unwrap_or(*src_id)
        })
        .collect();

    // Merge with built_map for a full source→output mapping.
    for (&src, &out_id) in &ctx.built_map {
        decoded_map.entry(src).or_insert(out_id);
    }

    // Rebuild barrier nodes using decoded inputs.
    let mut remap = rebuild_all_barriers(
        graph,
        roots,
        &ctx.barriers,
        &alg_decoded,
        &decoded_map,
        &mut out,
    );

    // Add algebraic user roots to remap.
    for (&src, &out_id) in &decoded_map {
        remap.entry(src).or_insert(out_id);
    }

    (out, remap)
}

/// Tracks which source NodeIds are algebraic roots that feed into barriers.
/// After egglog decode, barriers are rebuilt referencing decoded NodeIds.
struct BarrierInfo {
    src_id: NodeId,
    /// Indices into `algebraic_roots` for each input that is algebraic.
    /// Non-algebraic inputs map to `None` and are rebuilt recursively.
    input_alg_indices: Vec<Option<usize>>,
}

struct EgglogContext<'a> {
    src: &'a HLIRGraph,
    out: HLIRGraph,
    /// Maps source NodeId → egglog expression for algebraic nodes.
    term_memo: HashMap<NodeId, EggExpr>,
    /// Leaf metadata, keyed by the integer id used in egglog terms.
    leaves: Vec<LeafKind>,
    /// Shape metadata, keyed by integer id used in EReshape/EExpand terms.
    shape_meta: Vec<Vec<Dim>>,
    /// DType metadata, keyed by integer id used in ECast terms.
    dtype_meta: Vec<DType>,
    /// Permute axes metadata, keyed by integer id used in EPermute terms.
    permute_meta: Vec<Vec<usize>>,
    /// Dedup map for shape ids.
    shape_dedup: HashMap<Vec<Dim>, i64>,
    /// Dedup map for dtype ids.
    dtype_dedup: HashMap<DType, i64>,
    /// Dedup map for permute axes ids.
    permute_dedup: HashMap<Vec<usize>, i64>,
    /// Algebraic sub-roots that need egglog extraction (fed to barriers).
    algebraic_roots: Vec<(EggExpr, NodeId)>,
    /// Barrier nodes that need post-decode reconstruction.
    barriers: Vec<BarrierInfo>,
    /// Maps source NodeId → output NodeId for already-rebuilt barrier/leaf nodes.
    built_map: HashMap<NodeId, NodeId>,
}

impl<'a> EgglogContext<'a> {
    fn new(src: &'a HLIRGraph) -> Self {
        Self {
            src,
            out: HLIRGraph::new(),
            term_memo: HashMap::new(),
            leaves: Vec::new(),
            shape_meta: Vec::new(),
            dtype_meta: Vec::new(),
            permute_meta: Vec::new(),
            shape_dedup: HashMap::new(),
            dtype_dedup: HashMap::new(),
            permute_dedup: HashMap::new(),
            algebraic_roots: Vec::new(),
            barriers: Vec::new(),
            built_map: HashMap::new(),
        }
    }

    fn alloc_leaf(&mut self, kind: LeafKind) -> i64 {
        let id = self.leaves.len() as i64;
        self.leaves.push(kind);
        id
    }

    fn alloc_shape(&mut self, shape: Vec<Dim>) -> i64 {
        if let Some(&id) = self.shape_dedup.get(&shape) {
            return id;
        }
        let id = self.shape_meta.len() as i64;
        self.shape_dedup.insert(shape.clone(), id);
        self.shape_meta.push(shape);
        id
    }

    fn alloc_dtype(&mut self, dtype: DType) -> i64 {
        if let Some(&id) = self.dtype_dedup.get(&dtype) {
            return id;
        }
        let id = self.dtype_meta.len() as i64;
        self.dtype_dedup.insert(dtype, id);
        self.dtype_meta.push(dtype);
        id
    }

    fn alloc_permute(&mut self, axes: Vec<usize>) -> i64 {
        if let Some(&id) = self.permute_dedup.get(&axes) {
            return id;
        }
        let id = self.permute_meta.len() as i64;
        self.permute_dedup.insert(axes.clone(), id);
        self.permute_meta.push(axes);
        id
    }

    fn is_algebraic(op: &Op) -> bool {
        matches!(
            op,
            Op::Const { .. }
                | Op::Load { .. }
                | Op::Neg(_)
                | Op::Recip(_)
                | Op::Exp(_)
                | Op::Log(_)
                | Op::Sqrt(_)
                | Op::Sin(_)
                | Op::Cast { .. }
                | Op::Add(_, _)
                | Op::Mul(_, _)
                | Op::Max(_, _)
                | Op::Min(_, _)
                | Op::Reshape { .. }
                | Op::Permute { .. }
                | Op::Expand { .. }
        )
    }

    /// Check if a node was encoded as a barrier (not a true algebraic term).
    fn is_barrier_encoded(ctx: &EgglogContext, id: NodeId) -> bool {
        ctx.barriers.iter().any(|b| b.src_id == id)
    }

    /// Recursively process a barrier node: register it and encode/process its inputs.
    fn encode_barrier(&mut self, id: NodeId) {
        if self.barriers.iter().any(|b| b.src_id == id) {
            return; // Already processed.
        }

        let node = self.src.node(id);
        let inputs = node.op.inputs();
        let mut input_alg_indices = Vec::with_capacity(inputs.len());

        for &inp in &inputs {
            let inp_node = self.src.node(inp);
            if Self::is_algebraic(&inp_node.op) {
                let term = self.encode(inp);
                let idx = self.algebraic_roots.len();
                self.algebraic_roots.push((term, inp));
                input_alg_indices.push(Some(idx));
            } else {
                // Non-algebraic input: recursively process as barrier.
                self.encode_barrier(inp);
                input_alg_indices.push(None);
            }
        }

        self.barriers.push(BarrierInfo {
            src_id: id,
            input_alg_indices,
        });
    }

    /// Encode an HLIR node as an egglog term string.
    /// Non-algebraic nodes are recorded as barriers for post-decode rebuild.
    fn encode(&mut self, id: NodeId) -> EggExpr {
        if let Some(term) = self.term_memo.get(&id) {
            return term.clone();
        }

        let node = self.src.node(id);
        let term = match &node.op {
            // Constants: classify as Zero/One/Leaf
            Op::Const { value, .. } => {
                let kind = LeafKind::Const {
                    ty: node.ty.clone(),
                    value: value.clone(),
                };
                let leaf_id = self.alloc_leaf(kind);
                if value.is_exact_zero() {
                    expr_call("Zero", vec![expr_i64(leaf_id)])
                } else if value.is_exact_one() {
                    expr_call("One", vec![expr_i64(leaf_id)])
                } else {
                    expr_call("Leaf", vec![expr_i64(leaf_id)])
                }
            }

            // Load: opaque leaf
            Op::Load { buffer } => {
                let out_id = self.out.load(*buffer, node.ty.clone());
                self.built_map.insert(id, out_id);
                let leaf_id = self.alloc_leaf(LeafKind::Node(out_id));
                expr_call("Leaf", vec![expr_i64(leaf_id)])
            }

            // Algebraic unary ops
            Op::Neg(input) => {
                let inner = self.encode(*input);
                expr_call("ENeg", vec![inner])
            }
            Op::Recip(input) => {
                let inner = self.encode(*input);
                expr_call("ERecip", vec![inner])
            }
            Op::Exp(input) => {
                let inner = self.encode(*input);
                expr_call("EExp", vec![inner])
            }
            Op::Log(input) => {
                let inner = self.encode(*input);
                expr_call("ELog", vec![inner])
            }
            Op::Sqrt(input) => {
                let inner = self.encode(*input);
                expr_call("ESqrt", vec![inner])
            }
            Op::Sin(input) => {
                let inner = self.encode(*input);
                expr_call("ESin", vec![inner])
            }

            // Cast
            Op::Cast { input, to } => {
                let inner = self.encode(*input);
                let did = self.alloc_dtype(*to);
                expr_call("ECast", vec![inner, expr_i64(did)])
            }

            // Algebraic binary ops
            Op::Add(a, b) => {
                let la = self.encode(*a);
                let lb = self.encode(*b);
                expr_call("EAdd", vec![la, lb])
            }
            Op::Mul(a, b) => {
                let la = self.encode(*a);
                let lb = self.encode(*b);
                expr_call("EMul", vec![la, lb])
            }
            Op::Max(a, b) => {
                let la = self.encode(*a);
                let lb = self.encode(*b);
                expr_call("EMax", vec![la, lb])
            }
            Op::Min(a, b) => {
                let la = self.encode(*a);
                let lb = self.encode(*b);
                expr_call("EMin", vec![la, lb])
            }

            // Reshape: encode with identity elimination
            Op::Reshape { input, shape } => {
                let inner = self.encode(*input);
                if self.src.ty(*input).shape == *shape {
                    // Identity reshape — skip, but still memoize.
                    self.term_memo.insert(id, inner.clone());
                    return inner;
                }
                let sid = self.alloc_shape(shape.clone());
                expr_call("EReshape", vec![inner, expr_i64(sid)])
            }

            // Permute: encode with identity elimination
            Op::Permute { input, axes } => {
                let inner = self.encode(*input);
                // Identity permute: [0, 1, 2, ...] for rank dimensions
                let rank = self.src.ty(*input).shape.len();
                let is_identity =
                    axes.len() == rank && axes.iter().enumerate().all(|(i, &a)| a == i);
                if is_identity {
                    self.term_memo.insert(id, inner.clone());
                    return inner;
                }
                let pid = self.alloc_permute(axes.clone());
                expr_call("EPermute", vec![inner, expr_i64(pid)])
            }

            // Expand: encode with identity elimination
            Op::Expand { input, shape } => {
                let inner = self.encode(*input);
                if self.src.ty(*input).shape == *shape {
                    // Identity expand — skip.
                    self.term_memo.insert(id, inner.clone());
                    return inner;
                }
                let sid = self.alloc_shape(shape.clone());
                expr_call("EExpand", vec![inner, expr_i64(sid)])
            }

            // Barrier ops: don't create egglog terms. Process recursively.
            _ => {
                self.encode_barrier(id);
                // Return a dummy — this won't be used for extraction since
                // barrier roots are handled separately.
                expr_call("Leaf", vec![expr_i64(-1)])
            }
        };

        self.term_memo.insert(id, term.clone());
        term
    }
}

/// Rebuild all barrier nodes, mapping their inputs through decoded algebraic
/// roots and previously-rebuilt barriers.
fn rebuild_all_barriers(
    src: &HLIRGraph,
    roots: &[NodeId],
    barriers: &[BarrierInfo],
    alg_decoded: &[NodeId],
    input_map: &HashMap<NodeId, NodeId>,
    out: &mut HLIRGraph,
) -> HashMap<NodeId, NodeId> {
    let mut remap: HashMap<NodeId, NodeId> = HashMap::new();

    // Copy input_map entries that are roots.
    for &r in roots {
        if let Some(&out_id) = input_map.get(&r) {
            remap.insert(r, out_id);
        }
    }

    for barrier in barriers {
        let src_node = src.node(barrier.src_id);
        let src_inputs = src_node.op.inputs();
        let mut new_inputs = Vec::with_capacity(src_inputs.len());

        for (k, &inp) in src_inputs.iter().enumerate() {
            if let Some(alg_idx) = barrier.input_alg_indices[k] {
                // This input was algebraic — use the decoded result.
                new_inputs.push(alg_decoded[alg_idx]);
            } else {
                // Non-algebraic input — look up in remap (other barriers) or input_map.
                let built = remap
                    .get(&inp)
                    .or_else(|| input_map.get(&inp))
                    .copied()
                    .unwrap_or(inp);
                new_inputs.push(built);
            }
        }

        let new_op = super::optimize::remap_op_inputs(&src_node.op, &new_inputs);
        let out_id = out.add_node(new_op, src_node.ty.clone());
        remap.insert(barrier.src_id, out_id);
    }

    remap
}

fn expr_i64(v: i64) -> EggExpr {
    EggExpr::Lit(egglog::ast::Span::Panic, egglog::ast::Literal::Int(v))
}

fn expr_call(name: &str, args: Vec<EggExpr>) -> EggExpr {
    EggExpr::Call(egglog::ast::Span::Panic, name.to_string(), args)
}

fn decode_term(
    termdag: &egglog::TermDag,
    term: &egglog::Term,
    leaves: &[LeafKind],
    shapes: &[Vec<Dim>],
    dtypes: &[DType],
    permutes: &[Vec<usize>],
    out: &mut HLIRGraph,
    memo: &mut HashMap<egglog::TermId, NodeId>,
) -> NodeId {
    let term_id = termdag.lookup(term);
    if let Some(&id) = memo.get(&term_id) {
        return id;
    }
    let id = decode_inner(termdag, term, leaves, shapes, dtypes, permutes, out, memo);
    memo.insert(term_id, id);
    id
}

fn decode_inner(
    termdag: &egglog::TermDag,
    term: &egglog::Term,
    leaves: &[LeafKind],
    shapes: &[Vec<Dim>],
    dtypes: &[DType],
    permutes: &[Vec<usize>],
    out: &mut HLIRGraph,
    memo: &mut HashMap<egglog::TermId, NodeId>,
) -> NodeId {
    match term {
        egglog::Term::App(ctor, args) if ctor == "Leaf" || ctor == "Zero" || ctor == "One" => {
            let leaf_id = decode_i64_arg(termdag, args, 0, "leaf id") as usize;
            match &leaves[leaf_id] {
                LeafKind::Node(nid) => *nid,
                LeafKind::Const { ty, value } => {
                    out.constant(value.clone(), ty.shape.clone(), ty.dtype)
                }
            }
        }
        egglog::Term::App(ctor, args) if ctor == "EAdd" => {
            let la = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            let lb = decode_term(
                termdag,
                termdag.get(args[1]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            out.binary(la, lb, Op::Add)
        }
        egglog::Term::App(ctor, args) if ctor == "EMul" => {
            let la = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            let lb = decode_term(
                termdag,
                termdag.get(args[1]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            out.binary(la, lb, Op::Mul)
        }
        egglog::Term::App(ctor, args) if ctor == "EMax" => {
            let la = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            let lb = decode_term(
                termdag,
                termdag.get(args[1]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            out.binary(la, lb, Op::Max)
        }
        egglog::Term::App(ctor, args) if ctor == "EMin" => {
            let la = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            let lb = decode_term(
                termdag,
                termdag.get(args[1]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            out.binary(la, lb, Op::Min)
        }
        egglog::Term::App(ctor, args) if ctor == "ENeg" => {
            let c = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            out.unary(c, Op::Neg)
        }
        egglog::Term::App(ctor, args) if ctor == "ERecip" => {
            let c = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            out.unary(c, Op::Recip)
        }
        egglog::Term::App(ctor, args) if ctor == "EExp" => {
            let c = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            out.unary(c, Op::Exp)
        }
        egglog::Term::App(ctor, args) if ctor == "ELog" => {
            let c = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            out.unary(c, Op::Log)
        }
        egglog::Term::App(ctor, args) if ctor == "ESqrt" => {
            let c = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            out.unary(c, Op::Sqrt)
        }
        egglog::Term::App(ctor, args) if ctor == "ESin" => {
            let c = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            out.unary(c, Op::Sin)
        }
        egglog::Term::App(ctor, args) if ctor == "EReshape" => {
            let c = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            let sid = decode_i64_arg(termdag, args, 1, "shape id") as usize;
            out.reshape(c, shapes[sid].clone())
        }
        egglog::Term::App(ctor, args) if ctor == "ECast" => {
            let c = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            let did = decode_i64_arg(termdag, args, 1, "dtype id") as usize;
            out.cast(c, dtypes[did])
        }
        egglog::Term::App(ctor, args) if ctor == "EPermute" => {
            let c = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            let pid = decode_i64_arg(termdag, args, 1, "permute id") as usize;
            out.permute(c, permutes[pid].clone())
        }
        egglog::Term::App(ctor, args) if ctor == "EExpand" => {
            let c = decode_term(
                termdag,
                termdag.get(args[0]),
                leaves,
                shapes,
                dtypes,
                permutes,
                out,
                memo,
            );
            let sid = decode_i64_arg(termdag, args, 1, "shape id") as usize;
            out.expand(c, shapes[sid].clone())
        }
        egglog::Term::App(other, _) => panic!("unknown egglog constructor: {other}"),
        other => panic!("malformed egglog term: {other:?}"),
    }
}

fn decode_i64_arg(
    termdag: &egglog::TermDag,
    args: &[egglog::TermId],
    idx: usize,
    what: &str,
) -> i64 {
    let arg = termdag.get(args[idx]);
    match arg {
        egglog::Term::Lit(egglog::ast::Literal::Int(v)) => *v,
        _ => panic!("expected int literal for {what}, got {arg:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::hlir::{BufferId, DType, Dim, TensorType};

    fn f32_ty(dims: &[i64]) -> TensorType {
        TensorType::contiguous(dims.iter().map(|&d| Dim::Const(d)).collect(), DType::F32)
    }

    #[test]
    fn add_zero_eliminated() {
        let mut g = HLIRGraph::new();
        let x = g.load(BufferId(0), f32_ty(&[4]));
        let zero = g.constant(Scalar::F32(0.0), vec![Dim::Const(4)], DType::F32);
        let add = g.binary(x, zero, Op::Add);

        let (opt, remap) = egglog_algebraic(&g, &[add]);
        let out_root = remap[&add];

        // The optimized graph should just be the load — no add, no const.
        assert!(
            matches!(opt.node(out_root).op, Op::Load { .. }),
            "Add(x, 0) should simplify to x, got {:?}",
            opt.node(out_root).op.name()
        );
    }

    #[test]
    fn mul_one_eliminated() {
        let mut g = HLIRGraph::new();
        let x = g.load(BufferId(0), f32_ty(&[4]));
        let one = g.constant(Scalar::F32(1.0), vec![Dim::Const(4)], DType::F32);
        let mul = g.binary(x, one, Op::Mul);

        let (opt, remap) = egglog_algebraic(&g, &[mul]);
        let out_root = remap[&mul];

        assert!(
            matches!(opt.node(out_root).op, Op::Load { .. }),
            "Mul(x, 1) should simplify to x, got {:?}",
            opt.node(out_root).op.name()
        );
    }

    #[test]
    fn mul_zero_becomes_zero() {
        let mut g = HLIRGraph::new();
        let x = g.load(BufferId(0), f32_ty(&[4]));
        let zero = g.constant(Scalar::F32(0.0), vec![Dim::Const(4)], DType::F32);
        let mul = g.binary(x, zero, Op::Mul);

        let (opt, remap) = egglog_algebraic(&g, &[mul]);
        let out_root = remap[&mul];

        assert!(
            matches!(opt.node(out_root).op, Op::Const { .. }),
            "Mul(x, 0) should simplify to Const(0), got {:?}",
            opt.node(out_root).op.name()
        );
    }

    #[test]
    fn neg_neg_eliminated() {
        let mut g = HLIRGraph::new();
        let x = g.load(BufferId(0), f32_ty(&[4]));
        let n1 = g.unary(x, Op::Neg);
        let n2 = g.unary(n1, Op::Neg);

        let (opt, remap) = egglog_algebraic(&g, &[n2]);
        let out_root = remap[&n2];

        assert!(
            matches!(opt.node(out_root).op, Op::Load { .. }),
            "Neg(Neg(x)) should simplify to x, got {:?}",
            opt.node(out_root).op.name()
        );
    }

    #[test]
    fn exp_log_eliminated() {
        let mut g = HLIRGraph::new();
        let x = g.load(BufferId(0), f32_ty(&[4]));
        let log = g.unary(x, Op::Log);
        let exp = g.unary(log, Op::Exp);

        let (opt, remap) = egglog_algebraic(&g, &[exp]);
        let out_root = remap[&exp];

        assert!(
            matches!(opt.node(out_root).op, Op::Load { .. }),
            "Exp(Log(x)) should simplify to x, got {:?}",
            opt.node(out_root).op.name()
        );
    }

    #[test]
    fn barrier_op_inputs_optimized() {
        // Reshape(Add(x, Const(0)), shape) -> Reshape(x, shape)
        let mut g = HLIRGraph::new();
        let x = g.load(BufferId(0), f32_ty(&[4]));
        let zero = g.constant(Scalar::F32(0.0), vec![Dim::Const(4)], DType::F32);
        let add = g.binary(x, zero, Op::Add);
        let reshape = g.reshape(add, vec![Dim::Const(2), Dim::Const(2)]);

        let (opt, remap) = egglog_algebraic(&g, &[reshape]);
        let out_root = remap[&reshape];

        // The reshape should still exist but its input should be simplified.
        assert!(
            matches!(opt.node(out_root).op, Op::Reshape { .. }),
            "barrier Reshape should be preserved, got {:?}",
            opt.node(out_root).op.name()
        );
        // And the graph should have fewer nodes (no Add, no Const(0)).
        assert!(
            opt.len() <= 2,
            "expected <= 2 nodes (Load + Reshape), got {}",
            opt.len()
        );
    }

    #[test]
    fn permute_add_zero_eliminated() {
        // Permute(Add(x, 0), axes) -> Permute(x, axes) (via add-zero elimination)
        let mut g = HLIRGraph::new();
        let x = g.load(BufferId(0), f32_ty(&[2, 3]));
        let zero = g.constant(
            Scalar::F32(0.0),
            vec![Dim::Const(2), Dim::Const(3)],
            DType::F32,
        );
        let add = g.binary(x, zero, Op::Add);
        let perm = g.permute(add, vec![1, 0]);

        let (opt, remap) = egglog_algebraic(&g, &[perm]);
        let out_root = remap[&perm];

        // Should simplify to Permute(x, axes) without the Add.
        assert!(
            matches!(opt.node(out_root).op, Op::Permute { .. }),
            "expected Permute at root, got {:?}",
            opt.node(out_root).op.name()
        );
        // Graph should be just Load + Permute (no Add, no Const)
        assert!(
            opt.len() <= 2,
            "expected <= 2 nodes (Load + Permute), got {}",
            opt.len()
        );
    }

    #[test]
    fn expand_add_zero_eliminated() {
        // Expand(Add(x, 0), shape) -> Expand(x, shape) (via add-zero elimination)
        let mut g = HLIRGraph::new();
        let x = g.load(BufferId(0), f32_ty(&[1, 4]));
        let zero = g.constant(
            Scalar::F32(0.0),
            vec![Dim::Const(1), Dim::Const(4)],
            DType::F32,
        );
        let add = g.binary(x, zero, Op::Add);
        let expanded = g.expand(add, vec![Dim::Const(4), Dim::Const(4)]);

        let (opt, remap) = egglog_algebraic(&g, &[expanded]);
        let out_root = remap[&expanded];

        // Should simplify to Expand(x, shape) without the Add.
        assert!(
            matches!(opt.node(out_root).op, Op::Expand { .. }),
            "expected Expand at root, got {:?}",
            opt.node(out_root).op.name()
        );
        // Graph should be just Load + Expand (no Add, no Const)
        assert!(
            opt.len() <= 2,
            "expected <= 2 nodes (Load + Expand), got {}",
            opt.len()
        );
    }

    #[test]
    fn expand_expand_collapsed() {
        // Expand(Expand(x, s1), s2) -> Expand(x, s2)
        let mut g = HLIRGraph::new();
        let x = g.load(BufferId(0), f32_ty(&[1, 4]));
        let e1 = g.expand(x, vec![Dim::Const(2), Dim::Const(4)]);
        let e2 = g.expand(e1, vec![Dim::Const(4), Dim::Const(4)]);

        let (opt, remap) = egglog_algebraic(&g, &[e2]);
        let out_root = remap[&e2];

        // Should have collapsed to a single Expand directly from Load.
        assert!(
            matches!(opt.node(out_root).op, Op::Expand { .. }),
            "expected Expand at root, got {:?}",
            opt.node(out_root).op.name()
        );
        if let Op::Expand { input, .. } = &opt.node(out_root).op {
            assert!(
                matches!(opt.node(*input).op, Op::Load { .. }),
                "expected Load directly under collapsed Expand, got {:?}",
                opt.node(*input).op.name()
            );
        }
    }
}
