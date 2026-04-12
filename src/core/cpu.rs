use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::core::compile::OutputRemapper;
use crate::core::hlir::{BufferId, DType, Dim, NodeId, Op, Scalar, TensorType, op::CmpOp};
use crate::core::llir::LLIRProgram;
use crate::core::traits::CodeGenerator;

#[derive(Clone, Debug, PartialEq)]
pub enum Buffer {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

#[derive(Clone, Debug)]
pub struct CpuModule {
    pub program: LLIRProgram,
    pub outputs: Vec<NodeId>,
}

#[derive(Clone, Debug)]
pub struct CpuCodeGenerator {
    pub outputs: Vec<NodeId>,
}

impl CpuCodeGenerator {
    pub fn new(outputs: Vec<NodeId>) -> Self {
        Self { outputs }
    }
}

impl CodeGenerator for CpuCodeGenerator {
    type Output = CpuModule;

    fn generate(&self, program: &LLIRProgram) -> Result<Self::Output> {
        Ok(CpuModule {
            program: program.clone(),
            outputs: self.outputs.clone(),
        })
    }
}

impl OutputRemapper for CpuCodeGenerator {
    fn with_remapped_outputs(&self, outputs: Vec<NodeId>) -> Self {
        Self { outputs }
    }
}

impl CpuModule {
    pub fn execute(&self, inputs: &[(BufferId, Buffer)]) -> Result<Vec<Buffer>> {
        let mut input_map: HashMap<BufferId, Buffer> = HashMap::new();
        for (id, buf) in inputs {
            input_map.insert(*id, buf.clone());
        }

        let mut values: HashMap<NodeId, TensorValue> = HashMap::new();
        let kernel_map: HashMap<NodeId, &crate::core::llir::program::Kernel> =
            self.program.kernels.iter().map(|k| (k.root, k)).collect();

        for kernel in &self.program.kernels {
            let _ = eval_node(
                kernel.root,
                &kernel_map,
                &mut values,
                &input_map,
                &mut HashMap::new(),
            )?;
        }

        let mut out = Vec::with_capacity(self.outputs.len());
        for id in &self.outputs {
            let v = values
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("missing output node {:?}", id))?;
            out.push(v.to_buffer());
        }
        Ok(out)
    }
}

fn eval_node(
    id: NodeId,
    kernels: &HashMap<NodeId, &crate::core::llir::program::Kernel>,
    values: &mut HashMap<NodeId, TensorValue>,
    inputs: &HashMap<BufferId, Buffer>,
    visiting: &mut HashMap<NodeId, bool>,
) -> Result<TensorValue> {
    if let Some(v) = values.get(&id) {
        return Ok(v.clone());
    }
    if visiting.get(&id).copied().unwrap_or(false) {
        bail!("cycle detected in kernel graph at node {:?}", id);
    }
    visiting.insert(id, true);

    let k = kernels
        .get(&id)
        .ok_or_else(|| anyhow::anyhow!("missing kernel for node {:?}", id))?;

    for dep in op_inputs(&k.op) {
        if !values.contains_key(&dep) {
            let _ = eval_node(dep, kernels, values, inputs, visiting)?;
        }
    }

    let v = eval_op(&k.op, &k.ty, values, inputs)?;
    values.insert(id, v.clone());
    visiting.insert(id, false);
    Ok(v)
}

fn op_inputs(op: &Op) -> smallvec::SmallVec<[NodeId; 3]> {
    op.inputs()
}

#[derive(Clone, Debug)]
struct TensorValue {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    dtype: DType,
    data: Vec<f64>,
}

impl TensorValue {
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    fn to_buffer(&self) -> Buffer {
        let materialized = self.materialize_contiguous();
        match self.dtype {
            DType::F32 => Buffer::F32(materialized.iter().map(|x| *x as f32).collect()),
            DType::F64 => Buffer::F64(materialized),
            DType::I32 => Buffer::I32(materialized.iter().map(|x| *x as i32).collect()),
            DType::I64 => Buffer::I64(materialized.iter().map(|x| *x as i64).collect()),
            DType::Bool => Buffer::Bool(materialized.iter().map(|x| *x != 0.0).collect()),
            _ => Buffer::F64(materialized),
        }
    }

    fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        compute_strides(shape)
    }

    fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides == Self::contiguous_strides(&self.shape)
    }

    fn get_flat(&self, logical_flat: usize) -> f64 {
        let coord = unravel_index(logical_flat, &self.shape);
        self.get_coord(&coord)
    }

    fn get_coord(&self, coord: &[usize]) -> f64 {
        let storage_index = self.offset
            + coord
                .iter()
                .zip(&self.strides)
                .map(|(c, s)| c * s)
                .sum::<usize>();
        self.data[storage_index]
    }

    fn materialize_contiguous(&self) -> Vec<f64> {
        let numel = self.numel();
        let mut out = vec![0.0; numel];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.get_flat(i);
        }
        out
    }

    fn as_contiguous(&self) -> TensorValue {
        if self.is_contiguous() {
            self.clone()
        } else {
            TensorValue {
                shape: self.shape.clone(),
                strides: Self::contiguous_strides(&self.shape),
                offset: 0,
                dtype: self.dtype,
                data: self.materialize_contiguous(),
            }
        }
    }
}

fn eval_op(
    op: &Op,
    ty: &TensorType,
    values: &HashMap<NodeId, TensorValue>,
    inputs: &HashMap<BufferId, Buffer>,
) -> Result<TensorValue> {
    let shape = dims_to_shape(&ty.shape)?;
    let dtype = ty.dtype;
    let out = match op {
        Op::Load { buffer } => from_input(*buffer, &shape, dtype, inputs)?,
        Op::Const { value, .. } => {
            let numel: usize = shape.iter().product();
            TensorValue {
                shape: shape.clone(),
                strides: TensorValue::contiguous_strides(&shape),
                offset: 0,
                dtype,
                data: vec![scalar_to_f64(value); numel],
            }
        }
        Op::Add(a, b) => binary(values, *a, *b, |x, y| x + y)?,
        Op::Mul(a, b) => binary(values, *a, *b, |x, y| x * y)?,
        Op::Max(a, b) => binary(values, *a, *b, |x, y| x.max(y))?,
        Op::Min(a, b) => binary(values, *a, *b, |x, y| x.min(y))?,
        Op::Neg(a) => unary(values, *a, |x| -x)?,
        Op::Recip(a) => unary(values, *a, |x| 1.0 / x)?,
        Op::Exp(a) => unary(values, *a, |x| x.exp())?,
        Op::Log(a) => unary(values, *a, |x| x.ln())?,
        Op::Sqrt(a) => unary(values, *a, |x| x.sqrt())?,
        Op::Sin(a) => unary(values, *a, |x| x.sin())?,
        Op::Reshape { input, .. } => {
            let v = values
                .get(input)
                .ok_or_else(|| anyhow::anyhow!("missing reshape input"))?
                .clone();
            if v.is_contiguous() {
                TensorValue {
                    shape: shape.clone(),
                    strides: TensorValue::contiguous_strides(&shape),
                    offset: 0,
                    dtype: v.dtype,
                    data: v.data,
                }
            } else {
                TensorValue {
                    shape: shape.clone(),
                    strides: TensorValue::contiguous_strides(&shape),
                    offset: 0,
                    dtype: v.dtype,
                    data: v.materialize_contiguous(),
                }
            }
        }
        Op::Expand { input, .. } => {
            let v = values
                .get(input)
                .ok_or_else(|| anyhow::anyhow!("missing expand input"))?;
            expand(v, &shape)?
        }
        Op::Broadcast { input, .. } => {
            let v = values
                .get(input)
                .ok_or_else(|| anyhow::anyhow!("missing broadcast input"))?;
            broadcast(v, &shape)?
        }
        Op::Permute { input, axes } => {
            let v = values
                .get(input)
                .ok_or_else(|| anyhow::anyhow!("missing permute input"))?;
            permute(v, axes)?
        }
        Op::Slice { input, ranges } => {
            let v = values
                .get(input)
                .ok_or_else(|| anyhow::anyhow!("missing slice input"))?;
            slice(v, ranges)?
        }
        Op::Concat { inputs: ins, axis } => {
            let mut parts = Vec::new();
            for id in ins {
                parts.push(
                    values
                        .get(id)
                        .ok_or_else(|| anyhow::anyhow!("missing concat input"))?
                        .clone(),
                );
            }
            concat(&parts, *axis)?
        }
        Op::Reduce {
            input,
            axes,
            op,
            keepdim,
        } => {
            let v = values
                .get(input)
                .ok_or_else(|| anyhow::anyhow!("missing reduce input"))?;
            reduce(v, axes, *op, *keepdim)?
        }
        Op::Cast { input, to } => {
            let mut v = values
                .get(input)
                .ok_or_else(|| anyhow::anyhow!("missing cast input"))?
                .clone();
            v.dtype = *to;
            v
        }
        Op::Cmp { lhs, rhs, op } => {
            let l = values
                .get(lhs)
                .ok_or_else(|| anyhow::anyhow!("missing cmp lhs"))?;
            let r = values
                .get(rhs)
                .ok_or_else(|| anyhow::anyhow!("missing cmp rhs"))?;
            if l.shape != r.shape {
                bail!("cmp shape mismatch");
            }
            let mut data = Vec::with_capacity(l.numel());
            for i in 0..l.numel() {
                let a = l.get_flat(i);
                let b = r.get_flat(i);
                data.push(match op {
                    CmpOp::Eq => (a == b) as i32 as f64,
                    CmpOp::Ne => (a != b) as i32 as f64,
                    CmpOp::Lt => (a < b) as i32 as f64,
                    CmpOp::Le => (a <= b) as i32 as f64,
                    CmpOp::Gt => (a > b) as i32 as f64,
                    CmpOp::Ge => (a >= b) as i32 as f64,
                });
            }
            TensorValue {
                shape: l.shape.clone(),
                strides: TensorValue::contiguous_strides(&l.shape),
                offset: 0,
                dtype: DType::Bool,
                data,
            }
        }
        Op::Where {
            cond,
            then_val,
            else_val,
        } => {
            let c = values
                .get(cond)
                .ok_or_else(|| anyhow::anyhow!("missing where cond"))?;
            let t = values
                .get(then_val)
                .ok_or_else(|| anyhow::anyhow!("missing where then"))?;
            let e = values
                .get(else_val)
                .ok_or_else(|| anyhow::anyhow!("missing where else"))?;
            let mut data = Vec::with_capacity(t.numel());
            for i in 0..t.numel() {
                data.push(if c.get_flat(i) != 0.0 {
                    t.get_flat(i)
                } else {
                    e.get_flat(i)
                });
            }
            TensorValue {
                shape: t.shape.clone(),
                strides: TensorValue::contiguous_strides(&t.shape),
                offset: 0,
                dtype: t.dtype,
                data,
            }
        }
        Op::Store { value, .. } => values
            .get(value)
            .ok_or_else(|| anyhow::anyhow!("missing store value"))?
            .clone(),
    };
    Ok(out)
}

fn dims_to_shape(dims: &[Dim]) -> Result<Vec<usize>> {
    dims.iter()
        .map(|d| match d {
            Dim::Const(v) => usize::try_from(*v).map_err(|_| anyhow::anyhow!("invalid dim {v}")),
            Dim::Sym(s) => Err(anyhow::anyhow!(
                "runtime does not support symbolic dims (Sym({}))",
                s.0
            )),
            Dim::Add(a, b) => {
                let av = dims_to_shape(std::slice::from_ref(a))?;
                let bv = dims_to_shape(std::slice::from_ref(b))?;
                Ok(av[0].saturating_add(bv[0]))
            }
            Dim::Mul(a, b) => {
                let av = dims_to_shape(std::slice::from_ref(a))?;
                let bv = dims_to_shape(std::slice::from_ref(b))?;
                Ok(av[0].saturating_mul(bv[0]))
            }
            Dim::Div(a, b) => {
                let av = dims_to_shape(std::slice::from_ref(a))?;
                let bv = dims_to_shape(std::slice::from_ref(b))?;
                if bv[0] == 0 {
                    Err(anyhow::anyhow!("division by zero"))
                } else {
                    Ok(av[0].saturating_div(bv[0]))
                }
            }
            Dim::Mod(a, b) => {
                let av = dims_to_shape(std::slice::from_ref(a))?;
                let bv = dims_to_shape(std::slice::from_ref(b))?;
                if bv[0] == 0 {
                    Err(anyhow::anyhow!("modulo by zero"))
                } else {
                    Ok(av[0] % bv[0])
                }
            }
        })
        .collect()
}

fn scalar_to_f64(s: &Scalar) -> f64 {
    s.to_f64()
}

fn from_input(
    id: BufferId,
    shape: &[usize],
    dtype: DType,
    inputs: &HashMap<BufferId, Buffer>,
) -> Result<TensorValue> {
    let b = inputs
        .get(&id)
        .ok_or_else(|| anyhow::anyhow!("missing input buffer {:?}", id))?;
    let data: Vec<f64> = match b {
        Buffer::F32(v) => v.iter().map(|x| *x as f64).collect(),
        Buffer::F64(v) => v.clone(),
        Buffer::I32(v) => v.iter().map(|x| *x as f64).collect(),
        Buffer::I64(v) => v.iter().map(|x| *x as f64).collect(),
        Buffer::Bool(v) => v.iter().map(|x| if *x { 1.0 } else { 0.0 }).collect(),
    };
    if data.len() != shape.iter().product::<usize>() {
        bail!("input {:?} size mismatch", id);
    }
    Ok(TensorValue {
        shape: shape.to_vec(),
        strides: TensorValue::contiguous_strides(shape),
        offset: 0,
        dtype,
        data,
    })
}

fn unary(
    values: &HashMap<NodeId, TensorValue>,
    a: NodeId,
    f: impl Fn(f64) -> f64,
) -> Result<TensorValue> {
    let v = values
        .get(&a)
        .ok_or_else(|| anyhow::anyhow!("missing unary input"))?;
    let vc = v.as_contiguous();
    Ok(TensorValue {
        shape: vc.shape.clone(),
        strides: TensorValue::contiguous_strides(&vc.shape),
        offset: 0,
        dtype: vc.dtype,
        data: vc.data.iter().map(|x| f(*x)).collect(),
    })
}

fn binary(
    values: &HashMap<NodeId, TensorValue>,
    a: NodeId,
    b: NodeId,
    f: impl Fn(f64, f64) -> f64,
) -> Result<TensorValue> {
    let l = values
        .get(&a)
        .ok_or_else(|| anyhow::anyhow!("missing binary lhs"))?;
    let r = values
        .get(&b)
        .ok_or_else(|| anyhow::anyhow!("missing binary rhs"))?;
    if l.numel() != r.numel() {
        bail!("binary size mismatch");
    }
    let mut out = vec![0.0; l.numel()];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = f(l.get_flat(i), r.get_flat(i));
    }
    Ok(TensorValue {
        shape: l.shape.clone(),
        strides: TensorValue::contiguous_strides(&l.shape),
        offset: 0,
        dtype: l.dtype,
        data: out,
    })
}

fn expand(v: &TensorValue, out_shape: &[usize]) -> Result<TensorValue> {
    let in_rank = v.shape.len();
    let out_rank = out_shape.len();
    if in_rank != out_rank {
        bail!("expand rank mismatch");
    }
    let mut out_strides = v.strides.clone();
    for i in 0..in_rank {
        if v.shape[i] != 1 && v.shape[i] != out_shape[i] {
            bail!(
                "expand incompatible dim at axis {}: in={}, out={}",
                i,
                v.shape[i],
                out_shape[i]
            );
        }
        if v.shape[i] == 1 && out_shape[i] > 1 {
            out_strides[i] = 0;
        }
    }
    Ok(TensorValue {
        shape: out_shape.to_vec(),
        strides: out_strides,
        offset: v.offset,
        dtype: v.dtype,
        data: v.data.clone(),
    })
}

fn broadcast(v: &TensorValue, out_shape: &[usize]) -> Result<TensorValue> {
    let in_rank = v.shape.len();
    let out_rank = out_shape.len();
    if in_rank > out_rank {
        bail!("broadcast rank mismatch");
    }

    let rank_offset = out_rank - in_rank;
    let mut out_strides = vec![0; out_rank];
    for out_axis in 0..out_rank {
        if out_axis < rank_offset {
            continue;
        }

        let in_axis = out_axis - rank_offset;
        if v.shape[in_axis] != 1 && v.shape[in_axis] != out_shape[out_axis] {
            bail!(
                "broadcast incompatible dim at axis {}: in={}, out={}",
                out_axis,
                v.shape[in_axis],
                out_shape[out_axis]
            );
        }

        out_strides[out_axis] = if v.shape[in_axis] == 1 {
            0
        } else {
            v.strides[in_axis]
        };
    }

    Ok(TensorValue {
        shape: out_shape.to_vec(),
        strides: out_strides,
        offset: v.offset,
        dtype: v.dtype,
        data: v.data.clone(),
    })
}

fn permute(v: &TensorValue, axes: &[usize]) -> Result<TensorValue> {
    if axes.len() != v.shape.len() {
        bail!("permute rank mismatch");
    }
    let out_shape: Vec<usize> = axes.iter().map(|&i| v.shape[i]).collect();
    let out_strides: Vec<usize> = axes.iter().map(|&i| v.strides[i]).collect();
    Ok(TensorValue {
        shape: out_shape,
        strides: out_strides,
        offset: v.offset,
        dtype: v.dtype,
        data: v.data.clone(),
    })
}

fn slice(v: &TensorValue, ranges: &[crate::core::hlir::Range]) -> Result<TensorValue> {
    if ranges.len() != v.shape.len() {
        bail!("slice rank mismatch");
    }
    let mut starts = Vec::with_capacity(ranges.len());
    let mut out_shape = Vec::with_capacity(ranges.len());
    for (i, r) in ranges.iter().enumerate() {
        let s = if let Dim::Const(v) = r.start {
            v as usize
        } else {
            return Err(anyhow::anyhow!("slice requires const ranges"));
        };
        let e = if let Dim::Const(v) = r.end {
            v as usize
        } else {
            return Err(anyhow::anyhow!("slice requires const ranges"));
        };
        if s > e || e > v.shape[i] {
            bail!("slice out of bounds");
        }
        starts.push(s);
        out_shape.push(e - s);
    }
    let mut offset = v.offset;
    for (start, stride) in starts.iter().zip(&v.strides) {
        offset += start * stride;
    }
    Ok(TensorValue {
        shape: out_shape,
        strides: v.strides.clone(),
        offset,
        dtype: v.dtype,
        data: v.data.clone(),
    })
}

fn concat(parts: &[TensorValue], axis: usize) -> Result<TensorValue> {
    if parts.is_empty() {
        bail!("concat inputs empty");
    }
    let dtype = parts[0].dtype;
    let mut out_shape = parts[0].shape.clone();
    out_shape[axis] = parts.iter().map(|p| p.shape[axis]).sum();
    let out_numel: usize = out_shape.iter().product();
    let mut out = vec![0.0; out_numel];
    let out_strides = compute_strides(&out_shape);

    let mut axis_offset = 0usize;
    for p in parts {
        for idx in 0..p.numel() {
            let mut coord = unravel_index(idx, &p.shape);
            coord[axis] += axis_offset;
            let out_idx = ravel_index(&coord, &out_strides);
            out[out_idx] = p.get_flat(idx);
        }
        axis_offset += p.shape[axis];
    }

    let out_strides = compute_strides(&out_shape);
    Ok(TensorValue {
        shape: out_shape,
        strides: out_strides,
        offset: 0,
        dtype,
        data: out,
    })
}

fn reduce(
    v: &TensorValue,
    axes: &[usize],
    op: crate::core::hlir::ReduceOp,
    keepdim: bool,
) -> Result<TensorValue> {
    let mut out_shape = Vec::new();
    for (i, d) in v.shape.iter().copied().enumerate() {
        if axes.contains(&i) {
            if keepdim {
                out_shape.push(1);
            }
        } else {
            out_shape.push(d);
        }
    }
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let out_numel: usize = out_shape.iter().product();
    let init = match op {
        crate::core::hlir::ReduceOp::Sum => 0.0,
        crate::core::hlir::ReduceOp::Prod => 1.0,
        crate::core::hlir::ReduceOp::Max => f64::NEG_INFINITY,
        crate::core::hlir::ReduceOp::Min => f64::INFINITY,
    };
    let mut out = vec![init; out_numel];
    let out_strides = compute_strides(&out_shape);

    for idx in 0..v.numel() {
        let val = v.get_flat(idx);
        let coord = unravel_index(idx, &v.shape);
        let mut out_coord = Vec::new();
        for (i, c) in coord.iter().copied().enumerate() {
            if axes.contains(&i) {
                if keepdim {
                    out_coord.push(0);
                }
            } else {
                out_coord.push(c);
            }
        }
        if out_coord.is_empty() {
            out_coord.push(0);
        }
        let oi = ravel_index(&out_coord, &out_strides);
        out[oi] = match op {
            crate::core::hlir::ReduceOp::Sum => out[oi] + val,
            crate::core::hlir::ReduceOp::Prod => out[oi] * val,
            crate::core::hlir::ReduceOp::Max => out[oi].max(val),
            crate::core::hlir::ReduceOp::Min => out[oi].min(val),
        };
    }

    let out_strides = compute_strides(&out_shape);
    Ok(TensorValue {
        shape: out_shape,
        strides: out_strides,
        offset: 0,
        dtype: v.dtype,
        data: out,
    })
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn unravel_index(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let strides = compute_strides(shape);
    let mut coord = vec![0usize; shape.len()];
    for i in 0..shape.len() {
        coord[i] = idx / strides[i];
        idx %= strides[i];
    }
    coord
}

fn ravel_index(coord: &[usize], strides: &[usize]) -> usize {
    coord.iter().zip(strides).map(|(c, s)| c * s).sum()
}
