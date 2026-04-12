//! Structural decode and barrier rebuild for the typed egglog pipeline.

use std::collections::HashMap;

use super::super::{DType, Dim, HLIRGraph, NodeId, Op};
use super::encode::{EggExpr, StructuralEncoding};
use super::facts::{DimFact, ScalarId};
use super::region::LeafRef;

#[derive(Debug)]
#[allow(dead_code)]
pub enum DecodeError {
    Parse(String),
    Malformed(String),
    MissingExtraction(NodeId),
    Unresolved(String),
}

pub fn decode_with_extracted(
    src: &HLIRGraph,
    roots: &[NodeId],
    encoding: &StructuralEncoding,
    extracted_terms: &[(NodeId, String)],
) -> Result<(HLIRGraph, HashMap<NodeId, NodeId>), DecodeError> {
    let mut parser = egglog::ast::Parser::default();
    let mut extracted_exprs = HashMap::new();
    for (src_id, term) in extracted_terms {
        let expr = parser
            .get_expr_from_string(None, term)
            .map_err(|err| DecodeError::Parse(format!("failed to parse extracted term: {err}")))?;
        extracted_exprs.insert(*src_id, expr);
    }

    let mut state = DecodeState::new(src, encoding);
    for &region_root in &encoding.plan.region_roots {
        if !extracted_exprs.contains_key(&region_root) {
            return Err(DecodeError::MissingExtraction(region_root));
        }
    }

    let mut pending_roots = encoding.plan.region_roots.clone();
    let mut pending_barriers: Vec<_> = encoding.plan.barriers.iter().map(|b| b.src_id).collect();

    loop {
        let mut progress = false;

        let mut next_pending = Vec::new();
        for src_id in pending_roots {
            if state.remap.contains_key(&src_id) {
                progress = true;
                continue;
            }
            let expr = extracted_exprs
                .get(&src_id)
                .ok_or(DecodeError::MissingExtraction(src_id))?;
            match state.decode_expr(expr) {
                Ok(decoded) => {
                    state.remap.insert(src_id, decoded);
                    progress = true;
                }
                Err(DecodeError::Unresolved(_)) => next_pending.push(src_id),
                Err(err) => return Err(err),
            }
        }
        pending_roots = next_pending;

        let mut next_barriers = Vec::new();
        for barrier_id in pending_barriers {
            if state.try_rebuild_barrier(barrier_id)? {
                progress = true;
            } else {
                next_barriers.push(barrier_id);
            }
        }
        pending_barriers = next_barriers;

        if pending_roots.is_empty() && pending_barriers.is_empty() {
            break;
        }
        if !progress {
            return Err(DecodeError::Unresolved(
                "could not resolve all extracted roots/barriers".to_string(),
            ));
        }
    }

    for &root in roots {
        if !state.remap.contains_key(&root) {
            let copied = state.copy_source_subtree(root)?;
            state.remap.insert(root, copied);
        }
    }

    let root_map = roots
        .iter()
        .map(|&root| {
            let out = state.remap.get(&root).copied().ok_or_else(|| {
                DecodeError::Unresolved(format!("missing root mapping for {root:?}"))
            })?;
            Ok((root, out))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;

    Ok((state.out, root_map))
}

struct DecodeState<'a> {
    src: &'a HLIRGraph,
    encoding: &'a StructuralEncoding,
    out: HLIRGraph,
    remap: HashMap<NodeId, NodeId>,
    expr_memo: HashMap<String, NodeId>,
}

impl<'a> DecodeState<'a> {
    fn new(src: &'a HLIRGraph, encoding: &'a StructuralEncoding) -> Self {
        Self {
            src,
            encoding,
            out: HLIRGraph::new(),
            remap: HashMap::new(),
            expr_memo: HashMap::new(),
        }
    }

    fn decode_expr(&mut self, expr: &EggExpr) -> Result<NodeId, DecodeError> {
        let key = render_expr(expr);
        if let Some(&cached) = self.expr_memo.get(&key) {
            return Ok(cached);
        }

        let out_id = match expr {
            EggExpr::Call(_, name, args) if name.to_string() == "Leaf" => {
                let leaf_id = int_arg(args, 0, "Leaf")?;
                let leaf = self
                    .encoding
                    .plan
                    .symbolic_leaves
                    .get(usize::try_from(leaf_id).map_err(|_| {
                        DecodeError::Malformed(format!("negative leaf id: {leaf_id}"))
                    })?)
                    .ok_or_else(|| DecodeError::Malformed(format!("unknown leaf id: {leaf_id}")))?;
                match leaf {
                    LeafRef::Load(id) => self.copy_source_subtree(*id)?,
                    LeafRef::BarrierOutput(id) => self.remap.get(id).copied().ok_or_else(|| {
                        DecodeError::Unresolved(format!("unresolved barrier leaf {id:?}"))
                    })?,
                }
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Const" => {
                let scalar_id = int_arg(call_args(&args[0], "SConst")?, 0, "SConst")?;
                let scalar_index = usize::try_from(scalar_id).map_err(|_| {
                    DecodeError::Malformed(format!("negative scalar id: {scalar_id}"))
                })?;
                let scalar = if let Some(value) =
                    self.encoding.scalar_arena.try_value(ScalarId(scalar_index))
                {
                    value.bits.clone()
                } else {
                    let dtype = decode_dtype(&args[2])?;
                    synthesize_scalar_from_id(ScalarId(scalar_index), dtype)?
                };
                let shape = self.decode_shape(&args[1])?;
                let dtype = decode_dtype(&args[2])?;
                self.out.constant(scalar, shape, dtype)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Neg" => {
                let input = self.decode_expr(&args[0])?;
                self.out.unary(input, Op::Neg)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Recip" => {
                let input = self.decode_expr(&args[0])?;
                self.out.unary(input, Op::Recip)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Exp" => {
                let input = self.decode_expr(&args[0])?;
                self.out.unary(input, Op::Exp)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Log" => {
                let input = self.decode_expr(&args[0])?;
                self.out.unary(input, Op::Log)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Sqrt" => {
                let input = self.decode_expr(&args[0])?;
                self.out.unary(input, Op::Sqrt)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Sin" => {
                let input = self.decode_expr(&args[0])?;
                self.out.unary(input, Op::Sin)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Cast" => {
                let input = self.decode_expr(&args[0])?;
                let dtype = decode_dtype(&args[1])?;
                self.out.cast(input, dtype)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Add" => {
                let mut terms = Vec::new();
                collect_associative_terms("Add", expr, &mut terms);
                let decoded = terms
                    .into_iter()
                    .map(|term| self.decode_expr(term))
                    .collect::<Result<Vec<_>, _>>()?;
                build_balanced(&mut self.out, decoded, Op::Add)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Mul" => {
                let mut terms = Vec::new();
                collect_associative_terms("Mul", expr, &mut terms);
                let decoded = terms
                    .into_iter()
                    .map(|term| self.decode_expr(term))
                    .collect::<Result<Vec<_>, _>>()?;
                build_balanced(&mut self.out, decoded, Op::Mul)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Max" => {
                let lhs = self.decode_expr(&args[0])?;
                let rhs = self.decode_expr(&args[1])?;
                self.out.binary(lhs, rhs, Op::Max)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Min" => {
                let lhs = self.decode_expr(&args[0])?;
                let rhs = self.decode_expr(&args[1])?;
                self.out.binary(lhs, rhs, Op::Min)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Reshape" => {
                let input = self.decode_expr(&args[0])?;
                let shape = self.decode_shape(&args[1])?;
                self.out.reshape(input, shape)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Permute" => {
                let input = self.decode_expr(&args[0])?;
                let axes = decode_axes(&args[1])?;
                self.out.permute(input, axes)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Expand" => {
                let input = self.decode_expr(&args[0])?;
                let shape = self.decode_shape(&args[1])?;
                self.out.expand(input, shape)
            }
            EggExpr::Call(_, name, args) if name.to_string() == "Broadcast" => {
                let input = self.decode_expr(&args[0])?;
                let shape = self.decode_shape(&args[1])?;
                self.out.broadcast(input, shape)
            }
            other => {
                return Err(DecodeError::Malformed(format!(
                    "unsupported decoded expr: {other:?}"
                )))
            }
        };

        self.expr_memo.insert(key, out_id);
        Ok(out_id)
    }

    fn decode_shape(&self, shape_expr: &EggExpr) -> Result<Vec<Dim>, DecodeError> {
        let dim_ids = decode_shape_dim_ids(shape_expr)?;
        dim_ids
            .into_iter()
            .map(|dim_id| self.decode_dim(dim_id))
            .collect()
    }

    fn decode_dim(&self, dim_id: usize) -> Result<Dim, DecodeError> {
        let fact = self
            .encoding
            .facts
            .dims
            .get(dim_id)
            .ok_or_else(|| DecodeError::Malformed(format!("unknown dim id {dim_id}")))?;
        Ok(match fact {
            DimFact::Const(v) => Dim::Const(*v),
            DimFact::Sym(sym) => Dim::Sym(*sym),
            DimFact::Add(lhs, rhs) => Dim::Add(
                Box::new(self.decode_dim(lhs.0)?),
                Box::new(self.decode_dim(rhs.0)?),
            ),
            DimFact::Mul(lhs, rhs) => Dim::Mul(
                Box::new(self.decode_dim(lhs.0)?),
                Box::new(self.decode_dim(rhs.0)?),
            ),
            DimFact::Div(lhs, rhs) => Dim::Div(
                Box::new(self.decode_dim(lhs.0)?),
                Box::new(self.decode_dim(rhs.0)?),
            ),
            DimFact::Mod(lhs, rhs) => Dim::Mod(
                Box::new(self.decode_dim(lhs.0)?),
                Box::new(self.decode_dim(rhs.0)?),
            ),
        })
    }

    fn copy_source_subtree(&mut self, src_id: NodeId) -> Result<NodeId, DecodeError> {
        if let Some(&id) = self.remap.get(&src_id) {
            return Ok(id);
        }
        let src_node = self.src.node(src_id);
        let remapped_inputs = src_node
            .op
            .inputs()
            .iter()
            .map(|input| self.copy_source_subtree(*input))
            .collect::<Result<Vec<_>, _>>()?;
        let new_op = super::super::optimize::remap_op_inputs(&src_node.op, &remapped_inputs);
        let out_id = self.out.add_node(new_op, src_node.ty.clone());
        self.remap.insert(src_id, out_id);
        Ok(out_id)
    }

    fn try_rebuild_barrier(&mut self, barrier_id: NodeId) -> Result<bool, DecodeError> {
        if self.remap.contains_key(&barrier_id) {
            return Ok(true);
        }

        let src_node = self.src.node(barrier_id);
        if is_algebraic(&src_node.op) {
            return Ok(false);
        }

        let mut new_inputs = Vec::with_capacity(src_node.op.inputs().len());
        for input in src_node.op.inputs() {
            if let Some(&remapped) = self.remap.get(&input) {
                new_inputs.push(remapped);
                continue;
            }
            if is_algebraic(&self.src.node(input).op) {
                return Ok(false);
            }
            let copied = self.copy_source_subtree(input)?;
            new_inputs.push(copied);
        }

        let new_op = super::super::optimize::remap_op_inputs(&src_node.op, &new_inputs);
        let out_id = self.out.add_node(new_op, src_node.ty.clone());
        self.remap.insert(barrier_id, out_id);
        Ok(true)
    }
}

fn decode_dtype(expr: &EggExpr) -> Result<DType, DecodeError> {
    let name = call_name(expr)?;
    let dtype = match name.as_str() {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F64" => DType::F64,
        "I8" => DType::I8,
        "I16" => DType::I16,
        "I32" => DType::I32,
        "I64" => DType::I64,
        "U8" => DType::U8,
        "U16" => DType::U16,
        "U32" => DType::U32,
        "U64" => DType::U64,
        "Bool" => DType::Bool,
        other => {
            return Err(DecodeError::Malformed(format!(
                "unknown dtype ctor: {other}"
            )))
        }
    };
    Ok(dtype)
}

fn decode_shape_dim_ids(expr: &EggExpr) -> Result<Vec<usize>, DecodeError> {
    match expr {
        EggExpr::Call(_, name, args) if name.to_string() == "ShapeNil" => Ok(Vec::new()),
        EggExpr::Call(_, name, args) if name.to_string() == "ShapeCons" => {
            let dim_id = usize::try_from(int_arg(args, 0, "ShapeCons")?)
                .map_err(|_| DecodeError::Malformed("negative shape dim id".to_string()))?;
            let mut rest = decode_shape_dim_ids(&args[1])?;
            let mut out = Vec::with_capacity(rest.len() + 1);
            out.push(dim_id);
            out.append(&mut rest);
            Ok(out)
        }
        other => Err(DecodeError::Malformed(format!(
            "expected Shape term, got {other:?}"
        ))),
    }
}

fn decode_axes(expr: &EggExpr) -> Result<Vec<usize>, DecodeError> {
    match expr {
        EggExpr::Call(_, name, args) if name.to_string() == "AxesNil" => Ok(Vec::new()),
        EggExpr::Call(_, name, args) if name.to_string() == "AxesCons" => {
            let axis = usize::try_from(int_arg(args, 0, "AxesCons")?)
                .map_err(|_| DecodeError::Malformed("negative axis id".to_string()))?;
            let mut rest = decode_axes(&args[1])?;
            let mut out = Vec::with_capacity(rest.len() + 1);
            out.push(axis);
            out.append(&mut rest);
            Ok(out)
        }
        other => Err(DecodeError::Malformed(format!(
            "expected Axes term, got {other:?}"
        ))),
    }
}

fn int_arg(args: &[EggExpr], idx: usize, what: &str) -> Result<i64, DecodeError> {
    match args.get(idx) {
        Some(EggExpr::Lit(_, egglog::ast::Literal::Int(v))) => Ok(*v),
        Some(other) => Err(DecodeError::Malformed(format!(
            "expected int literal for {what}, got {other:?}"
        ))),
        None => Err(DecodeError::Malformed(format!(
            "missing argument {idx} for {what}"
        ))),
    }
}

fn call_name(expr: &EggExpr) -> Result<String, DecodeError> {
    match expr {
        EggExpr::Call(_, name, args) if args.is_empty() => Ok(name.to_string()),
        other => Err(DecodeError::Malformed(format!(
            "expected nullary constructor, got {other:?}"
        ))),
    }
}

fn call_args<'a>(expr: &'a EggExpr, expected: &str) -> Result<&'a [EggExpr], DecodeError> {
    match expr {
        EggExpr::Call(_, name, args) if name.to_string() == expected => Ok(args),
        other => Err(DecodeError::Malformed(format!(
            "expected {expected} call, got {other:?}"
        ))),
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

fn render_expr(expr: &EggExpr) -> String {
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

fn synthesize_scalar_from_id(
    id: ScalarId,
    dtype: DType,
) -> Result<super::super::Scalar, DecodeError> {
    let as_i64 = i64::try_from(id.0)
        .map_err(|_| DecodeError::Malformed(format!("scalar id too large: {}", id.0)))?;
    let scalar = match dtype {
        DType::F32 => super::super::Scalar::F32(as_i64 as f32),
        DType::F16 => {
            super::super::Scalar::F16(super::super::types::f32_to_f16_bits(as_i64 as f32))
        }
        DType::BF16 => {
            super::super::Scalar::BF16(super::super::types::f32_to_bf16_bits(as_i64 as f32))
        }
        DType::F64 => super::super::Scalar::F64(as_i64 as f64),
        DType::I8 => super::super::Scalar::I8(as_i64 as i8),
        DType::I16 => super::super::Scalar::I16(as_i64 as i16),
        DType::I32 => super::super::Scalar::I32(as_i64 as i32),
        DType::I64 => super::super::Scalar::I64(as_i64),
        DType::U8 => super::super::Scalar::U8(as_i64 as u8),
        DType::U16 => super::super::Scalar::U16(as_i64 as u16),
        DType::U32 => super::super::Scalar::U32(as_i64 as u32),
        DType::U64 => super::super::Scalar::U64(as_i64 as u64),
        DType::Bool => super::super::Scalar::Bool(as_i64 != 0),
    };
    Ok(scalar)
}

fn collect_associative_terms<'a>(op_name: &str, expr: &'a EggExpr, out: &mut Vec<&'a EggExpr>) {
    match expr {
        EggExpr::Call(_, name, args) if name.to_string() == op_name && args.len() == 2 => {
            collect_associative_terms(op_name, &args[0], out);
            collect_associative_terms(op_name, &args[1], out);
        }
        _ => out.push(expr),
    }
}

fn build_balanced(
    graph: &mut HLIRGraph,
    mut terms: Vec<NodeId>,
    op: fn(NodeId, NodeId) -> Op,
) -> NodeId {
    if terms.len() == 1 {
        return terms[0];
    }
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        let mut i = 0;
        while i + 1 < terms.len() {
            next.push(graph.binary(terms[i], terms[i + 1], op));
            i += 2;
        }
        if i < terms.len() {
            next.push(terms[i]);
        }
        terms = next;
    }
    terms[0]
}
