//! Structural HLIR-to-egglog encoding for the typed pipeline.

use std::collections::HashMap;

use super::super::{DType, HLIRGraph, NodeId, Op};
use super::facts::{AxesId, FactDatabase, ScalarArena, ScalarId, ShapeId};
use super::region::{plan_regions, LeafRef, RegionPlan};

pub type EggExpr = egglog::ast::Expr;

#[allow(dead_code)]
pub struct StructuralEncoding {
    pub plan: RegionPlan,
    pub facts: FactDatabase,
    pub scalar_arena: ScalarArena,
    expr_terms: HashMap<NodeId, EggExpr>,
    leaf_ids: HashMap<LeafRef, i64>,
    shape_terms: HashMap<ShapeId, EggExpr>,
    axes_terms: HashMap<AxesId, EggExpr>,
    scalar_terms: HashMap<ScalarId, EggExpr>,
    constant_scalars: HashMap<NodeId, ScalarId>,
}

#[allow(dead_code)]
impl StructuralEncoding {
    pub fn expr(&self, id: NodeId) -> Option<&EggExpr> {
        self.expr_terms.get(&id)
    }

    pub fn region_root_terms(&self) -> Vec<(NodeId, &EggExpr)> {
        self.plan
            .region_roots
            .iter()
            .filter_map(|&id| self.expr(id).map(|expr| (id, expr)))
            .collect()
    }

    pub fn leaf_id(&self, leaf: &LeafRef) -> Option<i64> {
        self.leaf_ids.get(leaf).copied()
    }

    pub fn shape_term(&self, shape: ShapeId) -> Option<&EggExpr> {
        self.shape_terms.get(&shape)
    }

    pub fn axes_term(&self, axes: AxesId) -> Option<&EggExpr> {
        self.axes_terms.get(&axes)
    }

    pub fn scalar_term(&self, scalar: ScalarId) -> Option<&EggExpr> {
        self.scalar_terms.get(&scalar)
    }

    pub fn scalar_id_for_const(&self, id: NodeId) -> Option<ScalarId> {
        self.constant_scalars.get(&id).copied()
    }

    pub fn expr_terms_sorted(&self) -> Vec<(NodeId, &EggExpr)> {
        let mut pairs = self
            .expr_terms
            .iter()
            .map(|(id, expr)| (*id, expr))
            .collect::<Vec<_>>();
        pairs.sort_by_key(|(id, _)| id.0);
        pairs
    }
}

pub fn encode_algebraic(graph: &HLIRGraph, roots: &[NodeId]) -> StructuralEncoding {
    StructuralEncoder::new(graph, roots).encode()
}

struct StructuralEncoder<'a> {
    graph: &'a HLIRGraph,
    plan: RegionPlan,
    facts: FactDatabase,
    scalar_arena: ScalarArena,
    expr_terms: HashMap<NodeId, EggExpr>,
    leaf_ids: HashMap<LeafRef, i64>,
    shape_terms: HashMap<ShapeId, EggExpr>,
    axes_terms: HashMap<AxesId, EggExpr>,
    scalar_terms: HashMap<ScalarId, EggExpr>,
    constant_scalars: HashMap<NodeId, ScalarId>,
    axes_lookup: HashMap<Vec<usize>, AxesId>,
}

impl<'a> StructuralEncoder<'a> {
    fn new(graph: &'a HLIRGraph, roots: &[NodeId]) -> Self {
        let plan = plan_regions(graph, roots);
        let facts = FactDatabase::from_graph(graph);
        let leaf_ids = plan
            .symbolic_leaves
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, leaf)| {
                (
                    leaf,
                    i64::try_from(index).expect("symbolic leaf count exceeded egglog i64 ids"),
                )
            })
            .collect();
        let axes_lookup = facts
            .axes
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, axes)| (axes, AxesId(index)))
            .collect();

        Self {
            graph,
            plan,
            facts,
            scalar_arena: ScalarArena::new(),
            expr_terms: HashMap::new(),
            leaf_ids,
            shape_terms: HashMap::new(),
            axes_terms: HashMap::new(),
            scalar_terms: HashMap::new(),
            constant_scalars: HashMap::new(),
            axes_lookup,
        }
    }

    fn encode(mut self) -> StructuralEncoding {
        let shape_ids: Vec<_> = (0..self.facts.shapes.len()).map(ShapeId).collect();
        for shape_id in shape_ids {
            let _ = self.encode_shape(shape_id);
        }

        let axes_ids: Vec<_> = (0..self.facts.axes.len()).map(AxesId).collect();
        for axes_id in axes_ids {
            let _ = self.encode_axes(axes_id);
        }

        let region_roots = self.plan.region_roots.clone();
        for root in region_roots {
            let _ = self.encode_node(root);
        }

        StructuralEncoding {
            plan: self.plan,
            facts: self.facts,
            scalar_arena: self.scalar_arena,
            expr_terms: self.expr_terms,
            leaf_ids: self.leaf_ids,
            shape_terms: self.shape_terms,
            axes_terms: self.axes_terms,
            scalar_terms: self.scalar_terms,
            constant_scalars: self.constant_scalars,
        }
    }

    fn encode_node(&mut self, id: NodeId) -> EggExpr {
        if let Some(term) = self.expr_terms.get(&id) {
            return term.clone();
        }

        let facts = self
            .facts
            .expr(id)
            .copied()
            .unwrap_or_else(|| panic!("missing fact entry for node {id:?}"));
        let op = self.graph.node(id).op.clone();
        let term = match op {
            Op::Load { .. } => self.leaf_expr(&LeafRef::Load(id)),
            Op::Const { value, .. } => {
                let scalar_id = self.scalar_arena.intern(value);
                self.constant_scalars.insert(id, scalar_id);
                let scalar = self.encode_scalar_id(scalar_id);
                let shape = self.encode_shape(facts.shape);
                let dtype = dtype_expr(facts.dtype);
                expr_call("Const", vec![scalar, shape, dtype])
            }
            Op::Neg(input) => expr_call("Neg", vec![self.encode_node(input)]),
            Op::Recip(input) => expr_call("Recip", vec![self.encode_node(input)]),
            Op::Exp(input) => expr_call("Exp", vec![self.encode_node(input)]),
            Op::Log(input) => expr_call("Log", vec![self.encode_node(input)]),
            Op::Sqrt(input) => expr_call("Sqrt", vec![self.encode_node(input)]),
            Op::Sin(input) => expr_call("Sin", vec![self.encode_node(input)]),
            Op::Cast { input, to } => {
                expr_call("Cast", vec![self.encode_node(input), dtype_expr(to)])
            }
            Op::Add(lhs, rhs) => {
                expr_call("Add", vec![self.encode_node(lhs), self.encode_node(rhs)])
            }
            Op::Mul(lhs, rhs) => {
                expr_call("Mul", vec![self.encode_node(lhs), self.encode_node(rhs)])
            }
            Op::Max(lhs, rhs) => {
                expr_call("Max", vec![self.encode_node(lhs), self.encode_node(rhs)])
            }
            Op::Min(lhs, rhs) => {
                expr_call("Min", vec![self.encode_node(lhs), self.encode_node(rhs)])
            }
            Op::Reshape { input, .. } => expr_call(
                "Reshape",
                vec![self.encode_node(input), self.encode_shape(facts.shape)],
            ),
            Op::Permute { input, axes } => {
                let axes = self.encode_axes_id(&axes);
                expr_call("Permute", vec![self.encode_node(input), axes])
            }
            Op::Expand { input, .. } => expr_call(
                "Expand",
                vec![self.encode_node(input), self.encode_shape(facts.shape)],
            ),
            Op::Broadcast { input, .. } => expr_call(
                "Broadcast",
                vec![self.encode_node(input), self.encode_shape(facts.shape)],
            ),
            Op::Store { .. }
            | Op::Cmp { .. }
            | Op::Where { .. }
            | Op::Reduce { .. }
            | Op::Slice { .. }
            | Op::Concat { .. } => self.leaf_expr(&LeafRef::BarrierOutput(id)),
        };

        if is_algebraic(&self.graph.node(id).op) {
            self.expr_terms.insert(id, term.clone());
        }
        term
    }

    fn encode_shape(&mut self, shape_id: ShapeId) -> EggExpr {
        if let Some(term) = self.shape_terms.get(&shape_id) {
            return term.clone();
        }

        let dims = self.facts.shape(shape_id).to_vec();
        let mut term = expr_call("ShapeNil", vec![]);
        for dim_id in dims.into_iter().rev() {
            term = expr_call("ShapeCons", vec![expr_usize(dim_id.0), term]);
        }

        self.shape_terms.insert(shape_id, term.clone());
        term
    }

    fn encode_axes(&mut self, axes_id: AxesId) -> EggExpr {
        if let Some(term) = self.axes_terms.get(&axes_id) {
            return term.clone();
        }

        let axes = self.facts.axes(axes_id).to_vec();
        let mut term = expr_call("AxesNil", vec![]);
        for axis in axes.into_iter().rev() {
            term = expr_call("AxesCons", vec![expr_usize(axis), term]);
        }

        self.axes_terms.insert(axes_id, term.clone());
        term
    }

    fn encode_axes_id(&mut self, axes: &[usize]) -> EggExpr {
        let axes_id = self
            .axes_lookup
            .get(axes)
            .copied()
            .unwrap_or_else(|| panic!("missing interned axes facts for {axes:?}"));
        self.encode_axes(axes_id)
    }

    fn encode_scalar_id(&mut self, scalar_id: ScalarId) -> EggExpr {
        if let Some(term) = self.scalar_terms.get(&scalar_id) {
            return term.clone();
        }

        let term = expr_call("SConst", vec![expr_usize(scalar_id.0)]);
        self.scalar_terms.insert(scalar_id, term.clone());
        term
    }

    fn leaf_expr(&self, leaf: &LeafRef) -> EggExpr {
        let leaf_id = self
            .leaf_ids
            .get(leaf)
            .copied()
            .unwrap_or_else(|| panic!("missing symbolic leaf id for {leaf:?}"));
        expr_call("Leaf", vec![expr_i64(leaf_id)])
    }
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
            | Op::Broadcast { .. }
    )
}

fn dtype_expr(dtype: DType) -> EggExpr {
    let name = match dtype {
        DType::F32 => "F32",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F64 => "F64",
        DType::I8 => "I8",
        DType::I16 => "I16",
        DType::I32 => "I32",
        DType::I64 => "I64",
        DType::U8 => "U8",
        DType::U16 => "U16",
        DType::U32 => "U32",
        DType::U64 => "U64",
        DType::Bool => "Bool",
    };
    expr_call(name, vec![])
}

fn expr_usize(value: usize) -> EggExpr {
    expr_i64(i64::try_from(value).expect("value exceeded egglog i64 literal range"))
}

fn expr_i64(value: i64) -> EggExpr {
    EggExpr::Lit(egglog::ast::Span::Panic, egglog::ast::Literal::Int(value))
}

fn expr_call(name: &str, args: Vec<EggExpr>) -> EggExpr {
    EggExpr::Call(egglog::ast::Span::Panic, name.into(), args)
}

pub fn render_expr(expr: &EggExpr) -> String {
    match expr {
        EggExpr::Lit(_, egglog::ast::Literal::Int(value)) => value.to_string(),
        EggExpr::Call(_, name, args) if args.is_empty() => format!("({name})"),
        EggExpr::Call(_, name, args) => {
            let rendered_args = args.iter().map(render_expr).collect::<Vec<_>>().join(" ");
            format!("({name} {rendered_args})")
        }
        other => format!("{other:?}"),
    }
}
