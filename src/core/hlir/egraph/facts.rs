//! Shape, dtype, layout, and scalar fact emission for the upcoming typed
//! egglog pipeline.

use std::collections::{HashMap, HashSet};

use super::super::types::{
    bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits, DType,
    Layout as HLIRLayout, Scalar, TensorType,
};
use super::super::{Dim, HLIRGraph, NodeId, Op, Symbol};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DimId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShapeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AxesId(pub usize);

/// Symbolic dimensions are serialized into stable integer ids and emitted as a
/// separate structural arena. Shapes then refer only to those `DimId`s, which
/// keeps shape facts egglog-compatible even when dimensions are symbolic trees.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DimFact {
    Const(i64),
    Sym(Symbol),
    Add(DimId, DimId),
    Mul(DimId, DimId),
    Div(DimId, DimId),
    Mod(DimId, DimId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayoutFact {
    Contiguous,
    Broadcasted,
    Strided,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExprFacts {
    pub shape: ShapeId,
    pub dtype: DType,
    pub layout: LayoutFact,
}

#[derive(Clone, Debug, Default)]
pub struct FactDatabase {
    pub dims: Vec<DimFact>,
    pub shapes: Vec<Vec<DimId>>,
    pub axes: Vec<Vec<usize>>,
    pub expr_facts: HashMap<NodeId, ExprFacts>,
    pub scalar_shapes: HashSet<ShapeId>,
    pub rank_of: HashMap<ShapeId, usize>,
    pub same_shape: HashSet<(ShapeId, ShapeId)>,
    pub same_numel: HashSet<(ShapeId, ShapeId)>,
    pub reshape_ok: HashSet<(ShapeId, ShapeId)>,
    pub broadcast_ok: HashSet<(ShapeId, ShapeId)>,
    pub expand_ok: HashSet<(ShapeId, ShapeId)>,
    pub permute_ok: HashSet<(ShapeId, AxesId, ShapeId)>,
    pub axes_compose: HashSet<(AxesId, AxesId, AxesId)>,
    pub axes_identity: HashSet<AxesId>,
    pub legal_add_shapes: HashSet<(ShapeId, ShapeId, ShapeId)>,
    pub legal_mul_shapes: HashSet<(ShapeId, ShapeId, ShapeId)>,
    pub legal_max_shapes: HashSet<(ShapeId, ShapeId, ShapeId)>,
    pub legal_min_shapes: HashSet<(ShapeId, ShapeId, ShapeId)>,
}

#[derive(Default)]
struct FactBuilder {
    facts: FactDatabase,
    dim_intern: HashMap<DimFact, DimId>,
    shape_intern: HashMap<Vec<DimId>, ShapeId>,
    axes_intern: HashMap<Vec<usize>, AxesId>,
    numel_of: HashMap<ShapeId, NumelKey>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum NumelKey {
    Constant(i128),
    Product {
        constant: i128,
        symbolic: Vec<DimId>,
    },
}

impl FactDatabase {
    pub fn from_graph(graph: &HLIRGraph) -> Self {
        FactBuilder::new().build(graph)
    }

    pub fn expr(&self, id: NodeId) -> Option<&ExprFacts> {
        self.expr_facts.get(&id)
    }

    pub fn dim(&self, id: DimId) -> &DimFact {
        &self.dims[id.0]
    }

    pub fn shape(&self, id: ShapeId) -> &[DimId] {
        &self.shapes[id.0]
    }

    pub fn axes(&self, id: AxesId) -> &[usize] {
        &self.axes[id.0]
    }
}

impl FactBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn build(mut self, graph: &HLIRGraph) -> FactDatabase {
        for (id, node) in graph.topo_iter() {
            self.record_node(id, &node.op, &node.ty);
        }

        self.ensure_identity_axes();
        self.derive_shape_relations();
        self.derive_axes_relations();
        self.derive_permute_relations();

        self.facts
    }

    fn record_node(&mut self, id: NodeId, op: &Op, ty: &TensorType) {
        let shape = self.intern_shape(&ty.shape);
        self.facts.expr_facts.insert(
            id,
            ExprFacts {
                shape,
                dtype: ty.dtype,
                layout: layout_fact(ty),
            },
        );

        match op {
            Op::Const { shape, .. }
            | Op::Reshape { shape, .. }
            | Op::Expand { shape, .. }
            | Op::Broadcast { shape, .. } => {
                let _ = self.intern_shape(shape);
            }
            Op::Permute { axes, .. } => {
                let _ = self.intern_axes(axes);
            }
            _ => {}
        }
    }

    fn ensure_identity_axes(&mut self) {
        let mut ranks: Vec<_> = self.facts.rank_of.values().copied().collect();
        ranks.sort_unstable();
        ranks.dedup();

        for rank in ranks {
            let axes: Vec<usize> = (0..rank).collect();
            let axes_id = self.intern_axes(&axes);
            self.facts.axes_identity.insert(axes_id);
        }
    }

    fn derive_shape_relations(&mut self) {
        let shape_ids: Vec<_> = (0..self.facts.shapes.len()).map(ShapeId).collect();

        for &shape_id in &shape_ids {
            self.facts.same_shape.insert((shape_id, shape_id));
        }

        for &lhs in &shape_ids {
            for &rhs in &shape_ids {
                if self.numel_of[&lhs] == self.numel_of[&rhs] {
                    self.facts.same_numel.insert((lhs, rhs));
                    self.facts.reshape_ok.insert((lhs, rhs));
                }

                if self.is_broadcast_ok(lhs, rhs) {
                    self.facts.broadcast_ok.insert((lhs, rhs));
                }

                if self.is_expand_ok(lhs, rhs) {
                    self.facts.expand_ok.insert((lhs, rhs));
                }

                if let Some(out_shape) = self.binary_output_shape(lhs, rhs) {
                    self.facts.legal_add_shapes.insert((lhs, rhs, out_shape));
                    self.facts.legal_mul_shapes.insert((lhs, rhs, out_shape));
                    self.facts.legal_max_shapes.insert((lhs, rhs, out_shape));
                    self.facts.legal_min_shapes.insert((lhs, rhs, out_shape));
                }
            }
        }
    }

    fn derive_axes_relations(&mut self) {
        let axes_ids: Vec<_> = (0..self.facts.axes.len()).map(AxesId).collect();

        for &axes_id in &axes_ids {
            if is_identity_axes(self.facts.axes(axes_id)) {
                self.facts.axes_identity.insert(axes_id);
            }
        }

        for &lhs in &axes_ids {
            for &rhs in &axes_ids {
                if let Some(composed) = compose_axes(self.facts.axes(lhs), self.facts.axes(rhs)) {
                    let composed_id = self.intern_axes(&composed);
                    self.facts.axes_compose.insert((lhs, rhs, composed_id));
                    if is_identity_axes(&composed) {
                        self.facts.axes_identity.insert(composed_id);
                    }
                }
            }
        }
    }

    fn derive_permute_relations(&mut self) {
        let shape_ids: Vec<_> = (0..self.facts.shapes.len()).map(ShapeId).collect();
        let axes_ids: Vec<_> = (0..self.facts.axes.len()).map(AxesId).collect();

        for &shape_id in &shape_ids {
            for &axes_id in &axes_ids {
                if let Some(permuted_dims) =
                    permuted_shape(self.facts.shape(shape_id), self.facts.axes(axes_id))
                {
                    let out_shape = self.intern_shape_ids(permuted_dims);
                    self.facts.permute_ok.insert((shape_id, axes_id, out_shape));
                }
            }
        }
    }

    fn intern_dim(&mut self, dim: &Dim) -> DimId {
        let key = match dim {
            Dim::Const(value) => DimFact::Const(*value),
            Dim::Sym(symbol) => DimFact::Sym(*symbol),
            Dim::Add(lhs, rhs) => {
                let lhs = self.intern_dim(lhs);
                let rhs = self.intern_dim(rhs);
                DimFact::Add(lhs, rhs)
            }
            Dim::Mul(lhs, rhs) => {
                let lhs = self.intern_dim(lhs);
                let rhs = self.intern_dim(rhs);
                DimFact::Mul(lhs, rhs)
            }
            Dim::Div(lhs, rhs) => {
                let lhs = self.intern_dim(lhs);
                let rhs = self.intern_dim(rhs);
                DimFact::Div(lhs, rhs)
            }
            Dim::Mod(lhs, rhs) => {
                let lhs = self.intern_dim(lhs);
                let rhs = self.intern_dim(rhs);
                DimFact::Mod(lhs, rhs)
            }
        };

        if let Some(&id) = self.dim_intern.get(&key) {
            return id;
        }

        let id = DimId(self.facts.dims.len());
        self.facts.dims.push(key.clone());
        self.dim_intern.insert(key, id);
        id
    }

    fn intern_shape(&mut self, shape: &[Dim]) -> ShapeId {
        let dim_ids: Vec<_> = shape.iter().map(|dim| self.intern_dim(dim)).collect();
        self.intern_shape_ids(dim_ids)
    }

    fn intern_shape_ids(&mut self, shape: Vec<DimId>) -> ShapeId {
        if let Some(&id) = self.shape_intern.get(&shape) {
            return id;
        }

        let id = ShapeId(self.facts.shapes.len());
        self.facts.shapes.push(shape.clone());
        self.shape_intern.insert(shape.clone(), id);
        self.facts.rank_of.insert(id, shape.len());
        if shape.is_empty() {
            self.facts.scalar_shapes.insert(id);
        }
        self.numel_of.insert(id, self.numel_key(&shape));
        id
    }

    fn intern_axes(&mut self, axes: &[usize]) -> AxesId {
        if let Some(&id) = self.axes_intern.get(axes) {
            return id;
        }

        let id = AxesId(self.facts.axes.len());
        let owned = axes.to_vec();
        self.facts.axes.push(owned.clone());
        self.axes_intern.insert(owned, id);
        id
    }

    fn numel_key(&self, shape: &[DimId]) -> NumelKey {
        let mut constant = 1_i128;
        let mut symbolic = Vec::new();

        for &dim_id in shape {
            self.collect_numel_factors(dim_id, &mut constant, &mut symbolic);
            if constant == 0 {
                return NumelKey::Constant(0);
            }
        }

        if symbolic.is_empty() {
            NumelKey::Constant(constant)
        } else {
            symbolic.sort_unstable();
            NumelKey::Product { constant, symbolic }
        }
    }

    fn collect_numel_factors(&self, dim_id: DimId, constant: &mut i128, symbolic: &mut Vec<DimId>) {
        match self.facts.dim(dim_id) {
            DimFact::Const(value) => {
                *constant = constant.saturating_mul(*value as i128);
            }
            DimFact::Mul(lhs, rhs) => {
                self.collect_numel_factors(*lhs, constant, symbolic);
                self.collect_numel_factors(*rhs, constant, symbolic);
            }
            _ => symbolic.push(dim_id),
        }
    }

    fn is_broadcast_ok(&self, from: ShapeId, to: ShapeId) -> bool {
        let from_dims = self.facts.shape(from);
        let to_dims = self.facts.shape(to);
        if from_dims.len() > to_dims.len() {
            return false;
        }

        let rank_offset = to_dims.len() - from_dims.len();
        for (to_index, &to_dim) in to_dims.iter().enumerate() {
            if to_index < rank_offset {
                continue;
            }

            let from_dim = from_dims[to_index - rank_offset];
            if from_dim == to_dim || self.is_const_one(from_dim) {
                continue;
            }

            return false;
        }

        true
    }

    fn is_expand_ok(&self, from: ShapeId, to: ShapeId) -> bool {
        let from_dims = self.facts.shape(from);
        let to_dims = self.facts.shape(to);
        if from_dims.len() != to_dims.len() {
            return false;
        }

        from_dims
            .iter()
            .zip(to_dims.iter())
            .all(|(&from_dim, &to_dim)| from_dim == to_dim || self.is_const_one(from_dim))
    }

    fn binary_output_shape(&mut self, lhs: ShapeId, rhs: ShapeId) -> Option<ShapeId> {
        let lhs_dims = self.facts.shape(lhs);
        let rhs_dims = self.facts.shape(rhs);
        let out_rank = lhs_dims.len().max(rhs_dims.len());
        let lhs_offset = out_rank - lhs_dims.len();
        let rhs_offset = out_rank - rhs_dims.len();
        let mut out_dims = Vec::with_capacity(out_rank);

        for out_index in 0..out_rank {
            let lhs_dim = (out_index >= lhs_offset).then(|| lhs_dims[out_index - lhs_offset]);
            let rhs_dim = (out_index >= rhs_offset).then(|| rhs_dims[out_index - rhs_offset]);

            match (lhs_dim, rhs_dim) {
                (Some(lhs_dim), Some(rhs_dim)) if lhs_dim == rhs_dim => out_dims.push(lhs_dim),
                (Some(lhs_dim), Some(rhs_dim)) if self.is_const_one(lhs_dim) => {
                    out_dims.push(rhs_dim)
                }
                (Some(lhs_dim), Some(rhs_dim)) if self.is_const_one(rhs_dim) => {
                    out_dims.push(lhs_dim)
                }
                (Some(_), Some(_)) => return None,
                (Some(lhs_dim), None) => out_dims.push(lhs_dim),
                (None, Some(rhs_dim)) => out_dims.push(rhs_dim),
                (None, None) => unreachable!("binary output rank cannot exceed both inputs"),
            }
        }

        Some(self.intern_shape_ids(out_dims))
    }

    fn is_const_one(&self, dim_id: DimId) -> bool {
        matches!(self.facts.dim(dim_id), DimFact::Const(1))
    }
}

fn layout_fact(ty: &TensorType) -> LayoutFact {
    if ty.is_contiguous() {
        return LayoutFact::Contiguous;
    }

    match &ty.layout {
        HLIRLayout::Contiguous => LayoutFact::Contiguous,
        HLIRLayout::Strided(strides) if is_broadcast_layout(&ty.shape, strides) => {
            LayoutFact::Broadcasted
        }
        HLIRLayout::Strided(_) => LayoutFact::Strided,
    }
}

fn is_broadcast_layout(shape: &[Dim], strides: &[Dim]) -> bool {
    let contiguous = TensorType::compute_contiguous_strides(shape);
    let mut saw_zero_stride = false;

    for (stride, contiguous_stride) in strides.iter().zip(contiguous.iter()) {
        if matches!(stride, Dim::Const(0)) {
            saw_zero_stride = true;
            continue;
        }

        if stride != contiguous_stride {
            return false;
        }
    }

    saw_zero_stride
}

fn permuted_shape(shape: &[DimId], axes: &[usize]) -> Option<Vec<DimId>> {
    if !is_valid_permutation(axes, shape.len()) {
        return None;
    }

    Some(axes.iter().map(|&axis| shape[axis]).collect())
}

fn compose_axes(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    if lhs.len() != rhs.len()
        || !is_valid_permutation(lhs, lhs.len())
        || !is_valid_permutation(rhs, rhs.len())
    {
        return None;
    }

    Some(rhs.iter().map(|&axis| lhs[axis]).collect())
}

fn is_identity_axes(axes: &[usize]) -> bool {
    axes.iter().enumerate().all(|(index, axis)| *axis == index)
}

fn is_valid_permutation(axes: &[usize], rank: usize) -> bool {
    if axes.len() != rank {
        return false;
    }

    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank || seen[axis] {
            return false;
        }
        seen[axis] = true;
    }

    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScalarId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarValue {
    pub dtype: DType,
    pub bits: Scalar,
}

#[derive(Clone, Default)]
pub struct ScalarArena {
    values: Vec<ScalarValue>,
    interned: HashMap<ScalarKey, ScalarId>,
    add_cache: HashMap<(ScalarId, ScalarId), ScalarId>,
    mul_cache: HashMap<(ScalarId, ScalarId), ScalarId>,
    neg_cache: HashMap<ScalarId, ScalarId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ScalarKey {
    F32(u32),
    F16(u16),
    BF16(u16),
    F64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Bool(bool),
}

impl ScalarArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, value: Scalar) -> ScalarId {
        let key = ScalarKey::from(&value);
        if let Some(&id) = self.interned.get(&key) {
            return id;
        }

        let id = ScalarId(self.values.len());
        self.values.push(ScalarValue {
            dtype: value.dtype(),
            bits: value,
        });
        self.interned.insert(key, id);
        id
    }

    pub fn cast(&mut self, id: ScalarId, dtype: DType) -> ScalarId {
        self.intern(self.value(id).bits.cast(dtype))
    }

    pub fn value(&self, id: ScalarId) -> &ScalarValue {
        &self.values[id.0]
    }

    pub fn try_value(&self, id: ScalarId) -> Option<&ScalarValue> {
        self.values.get(id.0)
    }

    pub fn scalar_ids(&self) -> Vec<ScalarId> {
        (0..self.values.len()).map(ScalarId).collect()
    }

    pub fn neg(&mut self, id: ScalarId) -> ScalarId {
        if let Some(&cached) = self.neg_cache.get(&id) {
            return cached;
        }

        let value = exact_neg(&self.value(id).bits);
        let result = self.intern(value);
        self.neg_cache.insert(id, result);
        result
    }

    pub fn add(&mut self, lhs: ScalarId, rhs: ScalarId) -> ScalarId {
        if let Some(&cached) = self.add_cache.get(&(lhs, rhs)) {
            return cached;
        }

        let value = exact_add(&self.value(lhs).bits, &self.value(rhs).bits);
        let result = self.intern(value);
        self.add_cache.insert((lhs, rhs), result);
        result
    }

    pub fn mul(&mut self, lhs: ScalarId, rhs: ScalarId) -> ScalarId {
        if let Some(&cached) = self.mul_cache.get(&(lhs, rhs)) {
            return cached;
        }

        let value = exact_mul(&self.value(lhs).bits, &self.value(rhs).bits);
        let result = self.intern(value);
        self.mul_cache.insert((lhs, rhs), result);
        result
    }

    pub fn is_zero(&self, id: ScalarId) -> bool {
        self.value(id).bits.is_exact_zero()
    }

    pub fn is_one(&self, id: ScalarId) -> bool {
        self.value(id).bits.is_exact_one()
    }
}

impl From<&Scalar> for ScalarKey {
    fn from(value: &Scalar) -> Self {
        match value {
            Scalar::F32(v) => Self::F32(v.to_bits()),
            Scalar::F16(bits) => Self::F16(*bits),
            Scalar::BF16(bits) => Self::BF16(*bits),
            Scalar::F64(v) => Self::F64(v.to_bits()),
            Scalar::I8(v) => Self::I8(*v),
            Scalar::I16(v) => Self::I16(*v),
            Scalar::I32(v) => Self::I32(*v),
            Scalar::I64(v) => Self::I64(*v),
            Scalar::U8(v) => Self::U8(*v),
            Scalar::U16(v) => Self::U16(*v),
            Scalar::U32(v) => Self::U32(*v),
            Scalar::U64(v) => Self::U64(*v),
            Scalar::Bool(v) => Self::Bool(*v),
        }
    }
}

fn exact_neg(value: &Scalar) -> Scalar {
    match value {
        Scalar::F32(v) => Scalar::F32(f32::from_bits(v.to_bits() ^ 0x8000_0000)),
        Scalar::F16(bits) => Scalar::F16(bits ^ 0x8000),
        Scalar::BF16(bits) => Scalar::BF16(bits ^ 0x8000),
        Scalar::F64(v) => Scalar::F64(f64::from_bits(v.to_bits() ^ 0x8000_0000_0000_0000)),
        Scalar::I8(v) => Scalar::I8(v.wrapping_neg()),
        Scalar::I16(v) => Scalar::I16(v.wrapping_neg()),
        Scalar::I32(v) => Scalar::I32(v.wrapping_neg()),
        Scalar::I64(v) => Scalar::I64(v.wrapping_neg()),
        Scalar::U8(v) => Scalar::U8(v.wrapping_neg()),
        Scalar::U16(v) => Scalar::U16(v.wrapping_neg()),
        Scalar::U32(v) => Scalar::U32(v.wrapping_neg()),
        Scalar::U64(v) => Scalar::U64(v.wrapping_neg()),
        Scalar::Bool(_) => panic!("boolean scalar negation is not supported"),
    }
}

fn exact_add(lhs: &Scalar, rhs: &Scalar) -> Scalar {
    assert_same_dtype(lhs, rhs);
    match (lhs, rhs) {
        (Scalar::F32(a), Scalar::F32(b)) => Scalar::F32(*a + *b),
        (Scalar::F16(a), Scalar::F16(b)) => {
            Scalar::F16(f32_to_f16_bits(f16_bits_to_f32(*a) + f16_bits_to_f32(*b)))
        }
        (Scalar::BF16(a), Scalar::BF16(b)) => Scalar::BF16(f32_to_bf16_bits(
            bf16_bits_to_f32(*a) + bf16_bits_to_f32(*b),
        )),
        (Scalar::F64(a), Scalar::F64(b)) => Scalar::F64(*a + *b),
        (Scalar::I8(a), Scalar::I8(b)) => Scalar::I8(a.wrapping_add(*b)),
        (Scalar::I16(a), Scalar::I16(b)) => Scalar::I16(a.wrapping_add(*b)),
        (Scalar::I32(a), Scalar::I32(b)) => Scalar::I32(a.wrapping_add(*b)),
        (Scalar::I64(a), Scalar::I64(b)) => Scalar::I64(a.wrapping_add(*b)),
        (Scalar::U8(a), Scalar::U8(b)) => Scalar::U8(a.wrapping_add(*b)),
        (Scalar::U16(a), Scalar::U16(b)) => Scalar::U16(a.wrapping_add(*b)),
        (Scalar::U32(a), Scalar::U32(b)) => Scalar::U32(a.wrapping_add(*b)),
        (Scalar::U64(a), Scalar::U64(b)) => Scalar::U64(a.wrapping_add(*b)),
        (Scalar::Bool(_), Scalar::Bool(_)) => panic!("boolean scalar addition is not supported"),
        _ => unreachable!("dtype mismatch slipped past scalar arena"),
    }
}

fn exact_mul(lhs: &Scalar, rhs: &Scalar) -> Scalar {
    assert_same_dtype(lhs, rhs);
    match (lhs, rhs) {
        (Scalar::F32(a), Scalar::F32(b)) => Scalar::F32(*a * *b),
        (Scalar::F16(a), Scalar::F16(b)) => {
            Scalar::F16(f32_to_f16_bits(f16_bits_to_f32(*a) * f16_bits_to_f32(*b)))
        }
        (Scalar::BF16(a), Scalar::BF16(b)) => Scalar::BF16(f32_to_bf16_bits(
            bf16_bits_to_f32(*a) * bf16_bits_to_f32(*b),
        )),
        (Scalar::F64(a), Scalar::F64(b)) => Scalar::F64(*a * *b),
        (Scalar::I8(a), Scalar::I8(b)) => Scalar::I8(a.wrapping_mul(*b)),
        (Scalar::I16(a), Scalar::I16(b)) => Scalar::I16(a.wrapping_mul(*b)),
        (Scalar::I32(a), Scalar::I32(b)) => Scalar::I32(a.wrapping_mul(*b)),
        (Scalar::I64(a), Scalar::I64(b)) => Scalar::I64(a.wrapping_mul(*b)),
        (Scalar::U8(a), Scalar::U8(b)) => Scalar::U8(a.wrapping_mul(*b)),
        (Scalar::U16(a), Scalar::U16(b)) => Scalar::U16(a.wrapping_mul(*b)),
        (Scalar::U32(a), Scalar::U32(b)) => Scalar::U32(a.wrapping_mul(*b)),
        (Scalar::U64(a), Scalar::U64(b)) => Scalar::U64(a.wrapping_mul(*b)),
        (Scalar::Bool(_), Scalar::Bool(_)) => {
            panic!("boolean scalar multiplication is not supported")
        }
        _ => unreachable!("dtype mismatch slipped past scalar arena"),
    }
}

fn assert_same_dtype(lhs: &Scalar, rhs: &Scalar) {
    assert_eq!(
        lhs.dtype(),
        rhs.dtype(),
        "scalar arena arithmetic requires matching dtypes: {:?} vs {:?}",
        lhs.dtype(),
        rhs.dtype()
    );
}
