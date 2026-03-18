use super::{
    context::{Context, KernelCache},
    dtype::{Buffer, DType, RealizedTensor, Scalar},
    graph::{Graph, NodeId, Op},
    jit::{compile_kernel, KernelSignature},
    optimize, render,
    schedule::{build_schedule, ScheduleItem},
};
use crate::core::{
    errors::{ExpansionError, ReshapeError, TransposeError, UnsqueezeError},
    iters::Indexer,
};
use anyhow::{anyhow, bail, Result};
use std::{cmp::Ordering, sync::Arc};

#[derive(Clone)]
pub struct Tensor {
    graph: Arc<std::sync::Mutex<Graph>>,
    kernel_cache: KernelCache,
    id: NodeId,
    shape: Vec<usize>,
    dtype: DType,
}

impl Tensor {
    fn with_graph_mut<R>(&self, f: impl FnOnce(&mut Graph) -> R) -> R {
        f(&mut self.graph.lock().unwrap())
    }

    fn derived(&self, id: NodeId, shape: Vec<usize>) -> Self {
        Self {
            graph: Arc::clone(&self.graph),
            kernel_cache: Arc::clone(&self.kernel_cache),
            id,
            shape,
            dtype: self.dtype,
        }
    }

    pub fn from_slice(cx: &Context, data: &[f32], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_f32_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self {
            graph,
            kernel_cache: cx.kernel_cache(),
            id,
            shape,
            dtype: DType::F32,
        }
    }

    pub fn from_slice_f64(cx: &Context, data: &[f64], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_f64_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self {
            graph,
            kernel_cache: cx.kernel_cache(),
            id,
            shape,
            dtype: DType::F64,
        }
    }

    pub fn from_slice_i32(cx: &Context, data: &[i32], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_i32_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self {
            graph,
            kernel_cache: cx.kernel_cache(),
            id,
            shape,
            dtype: DType::I32,
        }
    }

    pub fn from_slice_i64(cx: &Context, data: &[i64], shape: Vec<usize>) -> Self {
        let buffer = Buffer::from_i64_vec(data.to_vec());
        let graph = cx.graph();
        let id = graph.lock().unwrap().load(buffer, shape.clone());
        Self {
            graph,
            kernel_cache: cx.kernel_cache(),
            id,
            shape,
            dtype: DType::I64,
        }
    }

    pub fn constant(cx: &Context, value: f32, shape: Vec<usize>) -> Self {
        let graph = cx.graph();
        let id = graph
            .lock()
            .unwrap()
            .constant(Scalar::F32(value), shape.clone());
        Self {
            graph,
            kernel_cache: cx.kernel_cache(),
            id,
            shape,
            dtype: DType::F32,
        }
    }

    pub fn constant_scalar(cx: &Context, value: Scalar, shape: Vec<usize>) -> Self {
        let dtype = value.dtype();
        let graph = cx.graph();
        let id = graph.lock().unwrap().constant(value, shape.clone());
        Self {
            graph,
            kernel_cache: cx.kernel_cache(),
            id,
            shape,
            dtype,
        }
    }

    pub fn arange(cx: &Context, start: f32, end: f32, step: f32) -> anyhow::Result<Self> {
        use std::cmp::Ordering;

        let ascending = match step
            .partial_cmp(&0.0)
            .ok_or(anyhow::anyhow!("Cannot compare step value"))?
        {
            Ordering::Greater if end > start => Ok(true),
            Ordering::Less if start > end => Ok(false),
            Ordering::Greater => Err(anyhow::anyhow!("step is positive but end <= start")),
            Ordering::Less => Err(anyhow::anyhow!("step is negative but start <= end")),
            Ordering::Equal => Err(anyhow::anyhow!("step cannot be zero")),
        }?;

        let mut data = Vec::new();
        let mut curr = start;
        while (ascending && curr < end) || (!ascending && curr > end) {
            data.push(curr);
            curr += step;
        }

        let shape = vec![data.len()];
        Ok(Self::from_slice(cx, &data, shape))
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    // -------- elementwise lazy ops --------

    fn binary_op(&self, rhs: &Tensor, op: Op) -> Result<Tensor> {
        if !Arc::ptr_eq(&self.graph, &rhs.graph) {
            bail!("binary_op requires both tensors to share the same Context");
        }
        if self.dtype != rhs.dtype {
            bail!(
                "binary_op requires matching dtypes, got {:?} and {:?}",
                self.dtype,
                rhs.dtype
            );
        }
        let id = self.with_graph_mut(|g| g.binary(op, self.id, rhs.id));
        Ok(self.derived(id, self.shape.clone()))
    }

    fn unary_op(&self, op: Op) -> Tensor {
        let id = self.with_graph_mut(|g| g.unary(op, self.id));
        self.derived(id, self.shape.clone())
    }

    pub fn exp(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("exp requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Exp))
    }

    pub fn ln(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("ln requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Ln))
    }

    pub fn sqrt(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("sqrt requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Sqrt))
    }

    pub fn neg(&self) -> Tensor {
        self.unary_op(Op::Neg)
    }

    // -------- shape lazy ops --------

    pub fn reshape(&self, sizes: Vec<usize>) -> Result<Tensor> {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = sizes.iter().product();
        if old_numel != new_numel {
            bail!(ReshapeError {
                current_shape: self.shape.clone(),
                new_shape: sizes
            });
        }

        let id = self.with_graph_mut(|g| g.reshape(self.id, sizes.clone()));
        Ok(self.derived(id, sizes))
    }

    pub fn permute(&self, permutation: Vec<usize>) -> Result<Tensor> {
        if permutation.len() != self.shape.len() {
            bail!(ReshapeError {
                current_shape: self.shape.clone(),
                new_shape: self.shape.clone()
            });
        }

        let mut seen = vec![false; permutation.len()];
        for &p in &permutation {
            if p >= permutation.len() || seen[p] {
                bail!(ReshapeError {
                    current_shape: self.shape.clone(),
                    new_shape: self.shape.clone()
                });
            }
            seen[p] = true;
        }

        let shape: Vec<_> = permutation.iter().map(|&i| self.shape[i]).collect();
        let id = self.with_graph_mut(|g| g.permute(self.id, permutation, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if rank < 2 || dim_1 >= rank || dim_2 >= rank {
            bail!(TransposeError);
        }

        let mut shape = self.shape.clone();
        shape.swap(dim_1, dim_2);

        let id = self.with_graph_mut(|g| g.transpose(self.id, dim_1, dim_2, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn expand(&self, expansions: Vec<usize>) -> Result<Tensor> {
        if expansions.len() != self.shape.len() {
            bail!(ExpansionError {
                size: self.shape.len(),
                expansion: expansions.len()
            });
        }

        for (&size, &exp) in self.shape.iter().zip(expansions.iter()) {
            if !(exp == size || size == 1) {
                bail!(ExpansionError {
                    size,
                    expansion: exp
                });
            }
        }

        let id = self.with_graph_mut(|g| g.expand(self.id, expansions.clone(), expansions.clone()));
        Ok(self.derived(id, expansions))
    }

    pub fn slice(&self, ranges: Vec<(usize, usize)>) -> Result<Tensor> {
        if ranges.len() != self.shape.len() {
            bail!(
                "slice ranges rank mismatch: got {}, expected {}",
                ranges.len(),
                self.shape.len()
            );
        }

        let mut out = Vec::with_capacity(self.shape.len());
        for (dim, &(start, end_raw)) in ranges.iter().enumerate() {
            let size = self.shape[dim];
            let end = if end_raw == 0 { size } else { end_raw };
            if start > end || end > size {
                bail!(
                    "slice range {:?} out of bounds for dim {} with size {}",
                    (start, end),
                    dim,
                    size
                );
            }
            out.push(end - start);
        }

        let id = self.with_graph_mut(|g| g.slice(self.id, ranges, out.clone()));
        Ok(self.derived(id, out))
    }

    pub fn flip(&self, flips: Vec<usize>) -> Result<Tensor> {
        for &d in &flips {
            if d >= self.shape.len() {
                bail!(
                    "flip dimension {} out of bounds for rank {}",
                    d,
                    self.shape.len()
                );
            }
        }

        let shape = self.shape.clone();
        let id = self.with_graph_mut(|g| g.flip(self.id, flips, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn squeeze(&self) -> Result<Tensor> {
        let mut shape: Vec<usize> = self.shape.iter().copied().filter(|&s| s != 1).collect();
        if shape.is_empty() {
            shape.push(1);
        }

        let id = self.with_graph_mut(|g| g.squeeze(self.id, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn unsqueeze(&self, new_rank: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if new_rank < rank {
            bail!(UnsqueezeError {
                current: rank,
                new_rank
            });
        }
        if new_rank == rank {
            return Ok(self.clone());
        }

        let mut shape = vec![1; new_rank - rank];
        shape.extend_from_slice(&self.shape);

        let id = self.with_graph_mut(|g| g.unsqueeze(self.id, new_rank, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn pad(&self, constant: f32, padding: Vec<(usize, usize)>) -> Result<Tensor> {
        self.pad_scalar(Scalar::F32(constant), padding)
    }

    pub fn pad_scalar(&self, constant: Scalar, padding: Vec<(usize, usize)>) -> Result<Tensor> {
        let mut pad = padding;
        pad.resize(self.shape.len(), (0, 0));

        let mut shape = Vec::with_capacity(self.shape.len());
        for (s, (l, r)) in self.shape.iter().copied().zip(pad.iter().copied()) {
            shape.push(l + s + r);
        }

        let id = self.with_graph_mut(|g| g.pad(self.id, constant, pad, shape.clone()));
        Ok(self.derived(id, shape))
    }

    // -------- reduce lazy ops --------

    pub fn sum_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let id = self.with_graph_mut(|g| g.sum(self.id, dimensions, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn product_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let id = self.with_graph_mut(|g| g.prod(self.id, dimensions, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn max_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let id = self.with_graph_mut(|g| g.max(self.id, dimensions, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn min_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let id = self.with_graph_mut(|g| g.min(self.id, dimensions, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn sum(&self) -> Result<Tensor> {
        self.sum_dims((0..self.shape.len()).collect(), true)
    }

    pub fn product(&self) -> Result<Tensor> {
        self.product_dims((0..self.shape.len()).collect(), true)
    }

    pub fn max(&self) -> Result<Tensor> {
        self.max_dims((0..self.shape.len()).collect(), true)
    }

    pub fn min(&self) -> Result<Tensor> {
        self.min_dims((0..self.shape.len()).collect(), true)
    }

    // -------- composite ops --------

    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        let a_shape = &self.shape;
        let b_shape = &rhs.shape;

        if a_shape.len() < 2 || b_shape.len() < 2 {
            bail!(
                "matmul requires at least 2D tensors, got {:?} and {:?}",
                a_shape,
                b_shape
            );
        }

        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let k2 = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];

        if k != k2 {
            bail!(
                "matmul inner dimensions mismatch: {:?} vs {:?}",
                a_shape,
                b_shape
            );
        }

        let batch_a = &a_shape[..a_shape.len() - 2];
        let batch_b = &b_shape[..b_shape.len() - 2];
        let batch = broadcast_batch(batch_a, batch_b)?;
        let blen = batch.len();

        // Pad batch dims with leading 1s to match broadcast rank, then add
        // the extra dim for the dot-product axis, then expand everything.

        // a: [batch_a..., M, K] -> reshape [1..., batch_a..., M, K, 1]
        //                       -> expand  [batch...,         M, K, N]
        let mut a_rs = vec![1usize; blen - batch_a.len()];
        a_rs.extend_from_slice(batch_a);
        a_rs.extend_from_slice(&[m, k, 1]);
        let mut a_exp: Vec<usize> = batch.clone();
        a_exp.extend_from_slice(&[m, k, n]);
        let lhs = self.reshape(a_rs)?.expand(a_exp)?;

        // b: [batch_b..., K, N] -> reshape [1..., batch_b..., 1, K, N]
        //                       -> expand  [batch...,         M, K, N]
        let mut b_rs = vec![1usize; blen - batch_b.len()];
        b_rs.extend_from_slice(batch_b);
        b_rs.extend_from_slice(&[1, k, n]);
        let mut b_exp: Vec<usize> = batch.clone();
        b_exp.extend_from_slice(&[m, k, n]);
        let rhs = rhs.reshape(b_rs)?.expand(b_exp)?;

        // elementwise mul then sum-reduce over K
        (&lhs * &rhs)?.sum_dims(vec![blen + 1], false)
    }

    // -------- visualization --------

    pub fn render_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        render::render_dag(&graph, self.id)
    }

    pub fn render_fused_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        render::render_fused_dag(&graph, self.id)
    }

    pub fn render_optimized_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        if !is_optimize_safe(&graph, self.id) {
            return render::render_dag(&graph, self.id);
        }
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
        render::render_dag(&opt_graph, opt_root)
    }

    pub fn render_optimized_fused_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        if !is_optimize_safe(&graph, self.id) {
            return render::render_fused_dag(&graph, self.id);
        }
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
        render::render_fused_dag(&opt_graph, opt_root)
    }

    pub fn render_kernels(&self) -> String {
        use std::fmt::Write;

        let graph = self.graph.lock().unwrap();
        let optimize_safe = is_optimize_safe(&graph, self.id);
        let (exec_graph, exec_root) = if optimize_safe {
            optimize::optimize(&graph, self.id)
        } else {
            clone_reachable_subgraph(&graph, self.id)
        };
        drop(graph);

        let schedule = build_schedule(&exec_graph, exec_root);
        let mut out = String::new();

        writeln!(out, "Schedule: {} items", schedule.len()).unwrap();
        writeln!(out, "{}", "=".repeat(60)).unwrap();

        for (idx, item) in schedule.iter().enumerate() {
            match item {
                ScheduleItem::Shape(s) => {
                    writeln!(out, "\n[{}] Shape {:?} → {:?}", idx, s.op, s.shape).unwrap();
                }
                ScheduleItem::Reduce(r) => {
                    writeln!(out, "\n[{}] Reduce {:?} → {:?}", idx, r.op, r.shape).unwrap();
                }
                ScheduleItem::Fused(kernel) => {
                    writeln!(
                        out,
                        "\n[{}] Fused kernel  numel={}  output_shape={:?}",
                        idx, kernel.numel, kernel.output_shape
                    )
                    .unwrap();
                    writeln!(out, "    inputs: {} buffers", kernel.input_buffers.len()).unwrap();

                    for &buf_id in &kernel.input_buffers {
                        let node = exec_graph.node(buf_id);
                        if let Some(tracker) = kernel.input_trackers.get(&buf_id) {
                            writeln!(
                                out,
                                "      {:?} {:?} → tracker shape={:?} strides={:?} offset={}",
                                buf_id, node.shape, tracker.shape, tracker.strides, tracker.offset
                            )
                            .unwrap();
                        } else {
                            writeln!(out, "      {:?} {:?} (flat)", buf_id, node.shape).unwrap();
                        }
                    }

                    if !kernel.shape_source_map.is_empty() {
                        writeln!(
                            out,
                            "    absorbed {} shape ops",
                            kernel.shape_source_map.len()
                        )
                        .unwrap();
                    }

                    match compile_kernel(&exec_graph, kernel, true) {
                        Ok(compiled) => {
                            if let Some(ref ir) = compiled.clif_ir {
                                writeln!(out, "\n    --- CLIF IR ---").unwrap();
                                for line in ir.lines() {
                                    writeln!(out, "    {}", line).unwrap();
                                }
                            }
                        }
                        Err(e) => {
                            writeln!(out, "    (compilation error: {})", e).unwrap();
                        }
                    }
                }
            }
        }

        out
    }

    // -------- realize --------

    pub fn realize(&self) -> Result<RealizedTensor> {
        let graph = self.graph.lock().unwrap();

        // Fast path: already-backed leaf.
        let root_node = graph.node(self.id);
        if let Some(ref buffer) = root_node.buffer {
            return Ok(RealizedTensor::new(
                buffer.clone(),
                root_node.shape.clone(),
            ));
        }

        // Only optimize pure elementwise roots for now.
        let optimize_safe = is_optimize_safe(&graph, self.id);
        let (exec_graph, exec_root) = if optimize_safe {
            optimize::optimize(&graph, self.id)
        } else {
            clone_reachable_subgraph(&graph, self.id)
        };
        drop(graph);

        let graph = &exec_graph;
        let root = exec_root;

        let root_node = graph.node(root);
        if let Some(ref buffer) = root_node.buffer {
            return Ok(RealizedTensor::new(
                buffer.clone(),
                root_node.shape.clone(),
            ));
        }
        if let Op::Const(val) = root_node.op {
            let numel: usize = self.shape.iter().product();
            let buffer = scalar_fill_buffer(val, numel);
            return Ok(RealizedTensor::new(buffer, self.shape.clone()));
        }

        let schedule = build_schedule(graph, root);
        let mut realized: std::collections::HashMap<NodeId, Buffer> =
            std::collections::HashMap::new();

        for item in &schedule {
            match item {
                ScheduleItem::Fused(kernel) => {
                    let dtype = graph.node(kernel.root).dtype;
                    let sig = KernelSignature::from_kernel(graph, kernel);
                    let compiled = {
                        let mut cache = self.kernel_cache.lock().unwrap();
                        if let Some(cached) = cache.get(&sig) {
                            Arc::clone(cached)
                        } else {
                            let compiled = Arc::new(compile_kernel(graph, kernel, false)?);
                            cache.insert(sig, Arc::clone(&compiled));
                            compiled
                        }
                    };

                    let input_ptrs: Vec<*const u8> = kernel
                        .input_buffers
                        .iter()
                        .map(|&buf_id| -> Result<*const u8> {
                            let node = graph.node(buf_id);
                            if let Some(ref buffer) = node.buffer {
                                Ok(buffer.as_ptr_u8())
                            } else if let Some(buf) = realized.get(&buf_id) {
                                Ok(buf.as_ptr_u8())
                            } else {
                                bail!("Input buffer {:?} not realized and has no data", buf_id);
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let mut output_bytes =
                        super::dtype::zeros(dtype, kernel.numel);
                    unsafe {
                        compiled.execute(
                            &input_ptrs,
                            output_bytes.as_mut_ptr(),
                            kernel.numel,
                        );
                    }
                    realized.insert(kernel.root, buffer_from_bytes(output_bytes, dtype));
                }
                ScheduleItem::Shape(shape_item) => {
                    let input_node = graph.node(shape_item.input);
                    let input_shape = &input_node.shape;
                    let dtype = input_node.dtype;

                    let input_buf = get_realized_buffer(
                        graph,
                        shape_item.input,
                        &realized,
                        "Shape op",
                    )?;
                    let output = execute_shape_op_typed(
                        &shape_item.op,
                        &input_buf,
                        input_shape,
                        &shape_item.shape,
                        dtype,
                    )?;
                    realized.insert(shape_item.root, output);
                }
                ScheduleItem::Reduce(reduce_item) => {
                    let input_node = graph.node(reduce_item.input);
                    let input_shape = &input_node.shape;
                    let dtype = input_node.dtype;

                    let input_buf = get_realized_buffer(
                        graph,
                        reduce_item.input,
                        &realized,
                        "Reduce op",
                    )?;
                    let output = execute_reduce_op_typed(
                        &reduce_item.op,
                        &input_buf,
                        input_shape,
                        &reduce_item.shape,
                        dtype,
                    )?;
                    realized.insert(reduce_item.root, output);
                }
            }
        }

        let buf = realized
            .remove(&root)
            .ok_or_else(|| anyhow!("Root node was not realized"))?;

        Ok(RealizedTensor::new(buf, graph.node(root).shape.clone()))
    }
}

fn get_realized_buffer<'a>(
    graph: &'a Graph,
    node_id: NodeId,
    realized: &'a std::collections::HashMap<NodeId, Buffer>,
    context: &str,
) -> Result<std::borrow::Cow<'a, Buffer>> {
    if let Some(buf) = realized.get(&node_id) {
        Ok(std::borrow::Cow::Borrowed(buf))
    } else if let Some(ref buf) = graph.node(node_id).buffer {
        Ok(std::borrow::Cow::Borrowed(buf))
    } else {
        bail!("{} input {:?} is not realized", context, node_id);
    }
}

fn scalar_fill_buffer(val: Scalar, numel: usize) -> Buffer {
    match val {
        Scalar::F32(v) => Buffer::from_f32_vec(vec![v; numel]),
        Scalar::F64(v) => Buffer::from_f64_vec(vec![v; numel]),
        Scalar::I32(v) => Buffer::from_i32_vec(vec![v; numel]),
        Scalar::I64(v) => Buffer::from_i64_vec(vec![v; numel]),
    }
}

fn buffer_from_bytes(bytes: Vec<u8>, dtype: DType) -> Buffer {
    match dtype {
        DType::F32 => {
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            Buffer::from_f32_vec(data)
        }
        DType::F64 => {
            let data: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            Buffer::from_f64_vec(data)
        }
        DType::I32 => {
            let data: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            Buffer::from_i32_vec(data)
        }
        DType::I64 => {
            let data: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            Buffer::from_i64_vec(data)
        }
    }
}

// -------- dtype-dispatched shape/reduce execution --------

macro_rules! dispatch_shape_op {
    ($buf:expr, $op:expr, $input_shape:expr, $output_shape:expr, $dtype:expr) => {
        match $buf {
            Buffer::F32(v) => {
                let out = execute_shape_op($op, v, $input_shape, $output_shape)?;
                Ok(Buffer::from_f32_vec(out))
            }
            Buffer::F64(v) => {
                let out = execute_shape_op($op, v, $input_shape, $output_shape)?;
                Ok(Buffer::from_f64_vec(out))
            }
            Buffer::I32(v) => {
                let out = execute_shape_op($op, v, $input_shape, $output_shape)?;
                Ok(Buffer::from_i32_vec(out))
            }
            Buffer::I64(v) => {
                let out = execute_shape_op($op, v, $input_shape, $output_shape)?;
                Ok(Buffer::from_i64_vec(out))
            }
        }
    };
}

fn execute_shape_op_typed(
    op: &Op,
    input_buf: &Buffer,
    input_shape: &[usize],
    output_shape: &[usize],
    _dtype: DType,
) -> Result<Buffer> {
    dispatch_shape_op!(input_buf, op, input_shape, output_shape, _dtype)
}

fn execute_reduce_op_typed(
    op: &Op,
    input_buf: &Buffer,
    input_shape: &[usize],
    output_shape: &[usize],
    _dtype: DType,
) -> Result<Buffer> {
    match input_buf {
        Buffer::F32(v) => {
            let out = execute_reduce_op_f(op, v, input_shape, output_shape)?;
            Ok(Buffer::from_f32_vec(out))
        }
        Buffer::F64(v) => {
            let out = execute_reduce_op_f(op, v, input_shape, output_shape)?;
            Ok(Buffer::from_f64_vec(out))
        }
        Buffer::I32(v) => {
            let out = execute_reduce_op_i(op, v, input_shape, output_shape)?;
            Ok(Buffer::from_i32_vec(out))
        }
        Buffer::I64(v) => {
            let out = execute_reduce_op_i(op, v, input_shape, output_shape)?;
            Ok(Buffer::from_i64_vec(out))
        }
    }
}

// -------- shape-op execution helpers (generic over Copy types) --------

fn execute_shape_op<T: Copy + Default>(
    op: &Op,
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>> {
    match op {
        Op::Reshape(_) => execute_reshape(input_data, input_shape, output_shape),
        Op::Permute(perm) => execute_permute(input_data, input_shape, output_shape, perm),
        Op::Transpose(d1, d2) => execute_transpose(input_data, input_shape, output_shape, *d1, *d2),
        Op::Expand(_) => execute_expand(input_data, input_shape, output_shape),
        Op::Slice(ranges) => execute_slice(input_data, input_shape, output_shape, ranges),
        Op::Flip(flips) => execute_flip(input_data, input_shape, output_shape, flips),
        Op::Squeeze => execute_reshape(input_data, input_shape, output_shape),
        Op::Unsqueeze(_) => execute_reshape(input_data, input_shape, output_shape),
        Op::Pad(constant, padding) => {
            let fill = scalar_to_typed::<T>(*constant);
            execute_pad(input_data, input_shape, output_shape, fill, padding)
        }
        _ => bail!("execute_shape_op called with non-shape op: {:?}", op),
    }
}

fn scalar_to_typed<T: Copy + Default>(s: Scalar) -> T {
    // Safety: we know the Scalar variant matches the Buffer variant that called us.
    // Use byte reinterpretation to avoid needing trait bounds for From.
    unsafe {
        let mut val = T::default();
        let src = match s {
            Scalar::F32(v) => std::slice::from_raw_parts(&v as *const f32 as *const u8, 4),
            Scalar::F64(v) => std::slice::from_raw_parts(&v as *const f64 as *const u8, 8),
            Scalar::I32(v) => std::slice::from_raw_parts(&v as *const i32 as *const u8, 4),
            Scalar::I64(v) => std::slice::from_raw_parts(&v as *const i64 as *const u8, 8),
        };
        let dst = std::slice::from_raw_parts_mut(&mut val as *mut T as *mut u8, std::mem::size_of::<T>());
        dst.copy_from_slice(src);
        val
    }
}

fn execute_reduce_op_f<T>(
    op: &Op,
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + PartialOrd,
    T: std::iter::Sum + std::iter::Product,
{
    match op {
        Op::Sum(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().sum(),
        ),
        Op::Prod(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().product(),
        ),
        Op::Max(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| {
                slice
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(T::default())
            },
        ),
        Op::Min(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| {
                slice
                    .iter()
                    .copied()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(T::default())
            },
        ),
        _ => bail!("execute_reduce_op called with non-reduce op: {:?}", op),
    }
}

fn execute_reduce_op_i<T>(
    op: &Op,
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Ord,
    T: std::iter::Sum + std::iter::Product,
{
    match op {
        Op::Sum(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().sum(),
        ),
        Op::Prod(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().product(),
        ),
        Op::Max(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().max().unwrap_or(T::default()),
        ),
        Op::Min(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().min().unwrap_or(T::default()),
        ),
        _ => bail!("execute_reduce_op called with non-reduce op: {:?}", op),
    }
}

fn execute_reduce<T: Copy>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    dimensions: &[usize],
    keepdims: bool,
    reducer: impl Fn(&[T]) -> T,
) -> Result<Vec<T>> {
    validate_reduce_dims(input_shape, dimensions)?;

    let expected_shape = reduced_shape(input_shape, dimensions, keepdims)?;
    if expected_shape != output_shape {
        bail!(
            "reduce output shape mismatch: expected {:?}, got {:?}",
            expected_shape,
            output_shape
        );
    }

    let rank = input_shape.len();
    let mut is_reduce_dim = vec![false; rank];
    for &d in dimensions {
        is_reduce_dim[d] = true;
    }

    let reduce_numel: usize = dimensions.iter().map(|&d| input_shape[d]).product();
    let mut out = Vec::with_capacity(output_shape.iter().product());
    let mut fixed = vec![0usize; rank];
    let mut current = vec![0usize; rank];
    let mut reduced_values: Vec<T> = Vec::with_capacity(reduce_numel);

    for out_idx in Indexer::new(output_shape) {
        if keepdims {
            for d in 0..rank {
                if !is_reduce_dim[d] {
                    fixed[d] = out_idx[d];
                }
            }
        } else {
            let mut out_pos = 0usize;
            for d in 0..rank {
                if !is_reduce_dim[d] {
                    fixed[d] = out_idx[out_pos];
                    out_pos += 1;
                }
            }
        }

        reduced_values.clear();
        collect_reduce_values(
            input_data,
            input_shape,
            &is_reduce_dim,
            &fixed,
            0,
            &mut current,
            &mut reduced_values,
        );

        out.push(reducer(&reduced_values));
    }

    Ok(out)
}

fn collect_reduce_values<T: Copy>(
    input_data: &[T],
    input_shape: &[usize],
    is_reduce_dim: &[bool],
    fixed: &[usize],
    dim: usize,
    current: &mut [usize],
    values: &mut Vec<T>,
) {
    if dim == input_shape.len() {
        let off = idx_to_offset(current, input_shape);
        values.push(input_data[off]);
        return;
    }

    if is_reduce_dim[dim] {
        for i in 0..input_shape[dim] {
            current[dim] = i;
            collect_reduce_values(
                input_data,
                input_shape,
                is_reduce_dim,
                fixed,
                dim + 1,
                current,
                values,
            );
        }
    } else {
        current[dim] = fixed[dim];
        collect_reduce_values(
            input_data,
            input_shape,
            is_reduce_dim,
            fixed,
            dim + 1,
            current,
            values,
        );
    }
}

fn reduced_shape(shape: &[usize], dimensions: &[usize], keepdims: bool) -> Result<Vec<usize>> {
    validate_reduce_dims(shape, dimensions)?;

    let mut is_reduce_dim = vec![false; shape.len()];
    for &d in dimensions {
        is_reduce_dim[d] = true;
    }

    let mut out = Vec::new();
    for (d, &size) in shape.iter().enumerate() {
        if is_reduce_dim[d] {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(size);
        }
    }

    if out.is_empty() {
        out.push(1);
    }

    Ok(out)
}

fn validate_reduce_dims(shape: &[usize], dimensions: &[usize]) -> Result<()> {
    let mut seen = vec![false; shape.len()];
    for &d in dimensions {
        if d >= shape.len() {
            bail!(
                "reduce dimension {} out of bounds for rank {}",
                d,
                shape.len()
            );
        }
        if seen[d] {
            bail!("duplicate reduce dimension {}", d);
        }
        seen[d] = true;
    }
    Ok(())
}

fn execute_reshape<T: Copy>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>> {
    let in_numel: usize = input_shape.iter().product();
    let out_numel: usize = output_shape.iter().product();
    if in_numel != out_numel || input_data.len() != in_numel {
        bail!(
            "invalid reshape numel: input {:?}, output {:?}",
            input_shape,
            output_shape
        );
    }
    Ok(input_data.to_vec())
}

fn execute_permute<T: Copy + Default>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    perm: &[usize],
) -> Result<Vec<T>> {
    if perm.len() != input_shape.len() || output_shape.len() != input_shape.len() {
        bail!("invalid permute rank");
    }
    let mut out = vec![T::default(); output_shape.iter().product()];

    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = vec![0usize; input_shape.len()];
        for (out_dim, &in_dim) in perm.iter().enumerate() {
            in_idx[in_dim] = out_idx[out_dim];
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_transpose<T: Copy + Default>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    dim_1: usize,
    dim_2: usize,
) -> Result<Vec<T>> {
    if dim_1 >= input_shape.len()
        || dim_2 >= input_shape.len()
        || input_shape.len() != output_shape.len()
    {
        bail!("invalid transpose dims");
    }

    let mut perm: Vec<usize> = (0..input_shape.len()).collect();
    perm.swap(dim_1, dim_2);
    execute_permute(input_data, input_shape, output_shape, &perm)
}

fn execute_expand<T: Copy + Default>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<T>> {
    if input_shape.len() != output_shape.len() {
        bail!("expand rank mismatch");
    }

    let mut out = vec![T::default(); output_shape.iter().product()];
    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = vec![0usize; input_shape.len()];
        for d in 0..input_shape.len() {
            if input_shape[d] == output_shape[d] {
                in_idx[d] = out_idx[d];
            } else if input_shape[d] == 1 {
                in_idx[d] = 0;
            } else {
                bail!(
                    "cannot expand dim {} from {} to {}",
                    d,
                    input_shape[d],
                    output_shape[d]
                );
            }
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_slice<T: Copy + Default>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<T>> {
    if ranges.len() != input_shape.len() || output_shape.len() != input_shape.len() {
        bail!("slice rank mismatch");
    }

    let mut out = vec![T::default(); output_shape.iter().product()];
    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = vec![0usize; input_shape.len()];
        for d in 0..input_shape.len() {
            let (start, end_raw) = ranges[d];
            let end = if end_raw == 0 {
                input_shape[d]
            } else {
                end_raw
            };
            if start > end || end > input_shape[d] {
                bail!("invalid slice range {:?} for dim {}", ranges[d], d);
            }
            in_idx[d] = start + out_idx[d];
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_flip<T: Copy + Default>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    flips: &[usize],
) -> Result<Vec<T>> {
    if input_shape != output_shape {
        bail!("flip must preserve shape");
    }

    let mut flip_mask = vec![false; input_shape.len()];
    for &d in flips {
        if d >= input_shape.len() {
            bail!("flip dim {} out of range", d);
        }
        flip_mask[d] = true;
    }

    let mut out = vec![T::default(); output_shape.iter().product()];
    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = out_idx.clone();
        for d in 0..in_idx.len() {
            if flip_mask[d] {
                in_idx[d] = input_shape[d] - 1 - in_idx[d];
            }
        }

        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_pad<T: Copy>(
    input_data: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    constant: T,
    padding: &[(usize, usize)],
) -> Result<Vec<T>> {
    if input_shape.len() != output_shape.len() {
        bail!("pad rank mismatch");
    }

    let mut pad = padding.to_vec();
    pad.resize(input_shape.len(), (0, 0));

    let mut expected = Vec::with_capacity(input_shape.len());
    for d in 0..input_shape.len() {
        expected.push(pad[d].0 + input_shape[d] + pad[d].1);
    }
    if expected != output_shape {
        bail!(
            "pad output shape mismatch: expected {:?}, got {:?}",
            expected,
            output_shape
        );
    }

    let mut out = vec![constant; output_shape.iter().product()];
    for in_idx in Indexer::new(input_shape) {
        let mut out_idx = vec![0usize; input_shape.len()];
        for d in 0..input_shape.len() {
            out_idx[d] = pad[d].0 + in_idx[d];
        }

        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn idx_to_offset(index: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut off = 0usize;
    for d in (0..shape.len()).rev() {
        off += index[d] * stride;
        stride *= shape[d];
    }
    off
}

// -------- graph import helpers --------

fn import_node(
    src_graph: &Graph,
    src_id: NodeId,
    dst_graph: &mut Graph,
    id_map: &mut std::collections::HashMap<NodeId, NodeId>,
    buffer_map: &std::collections::HashMap<usize, NodeId>,
) -> NodeId {
    if let Some(&mapped) = id_map.get(&src_id) {
        return mapped;
    }

    let node = src_graph.node(src_id);

    if let (Op::Load, Some(ref buf)) = (&node.op, &node.buffer) {
        let ptr_key = buf.as_ptr_u8() as usize;
        if let Some(&existing_id) = buffer_map.get(&ptr_key) {
            id_map.insert(src_id, existing_id);
            return existing_id;
        }
    }

    let new_inputs: Vec<NodeId> = node
        .inputs
        .iter()
        .map(|&input_id| import_node(src_graph, input_id, dst_graph, id_map, buffer_map))
        .collect();

    let new_id = dst_graph.add_node(super::graph::Node {
        op: node.op.clone(),
        inputs: new_inputs,
        shape: node.shape.clone(),
        dtype: node.dtype,
        buffer: node.buffer.clone(),
    });

    id_map.insert(src_id, new_id);
    new_id
}

// -------- optimization helpers --------

fn is_optimize_safe(graph: &Graph, root: NodeId) -> bool {
    // Only optimize float dtypes — egglog rules use float constants.
    if !graph.node(root).dtype.is_float() {
        return false;
    }

    fn dfs(graph: &Graph, id: NodeId, seen: &mut std::collections::HashSet<NodeId>) -> bool {
        if !seen.insert(id) {
            return true;
        }
        let node = graph.node(id);
        let here_ok = matches!(
            node.op,
            Op::Load
                | Op::Const(_)
                | Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Exp
                | Op::Ln
                | Op::Sqrt
                | Op::Neg
        );
        if !here_ok {
            return false;
        }
        node.inputs.iter().all(|&inp| dfs(graph, inp, seen))
    }

    let mut seen = std::collections::HashSet::new();
    dfs(graph, root, &mut seen)
}

fn broadcast_batch(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        if da != db && da != 1 && db != 1 {
            bail!("batch dimensions not broadcastable: {:?} vs {:?}", a, b);
        }
        out[i] = da.max(db);
    }
    Ok(out)
}

fn clone_reachable_subgraph(src: &Graph, root: NodeId) -> (Graph, NodeId) {
    let mut dst = Graph::new();
    let mut id_map = std::collections::HashMap::new();
    let buffer_map = std::collections::HashMap::new();
    let new_root = import_node(src, root, &mut dst, &mut id_map, &buffer_map);
    (dst, new_root)
}

// -------- operator overloads --------

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $op:expr) => {
        // Core implementation: &Tensor op &Tensor
        impl<'a, 'b> std::ops::$trait<&'b Tensor> for &'a Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: &'b Tensor) -> Self::Output {
                self.binary_op(rhs, $op)
            }
        }

        // Tensor op &Tensor
        impl<'b> std::ops::$trait<&'b Tensor> for Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: &'b Tensor) -> Self::Output {
                self.binary_op(rhs, $op)
            }
        }

        // &Tensor op Tensor
        impl<'a> std::ops::$trait<Tensor> for &'a Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: Tensor) -> Self::Output {
                self.binary_op(&rhs, $op)
            }
        }

        // Tensor op Tensor
        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: Tensor) -> Self::Output {
                self.binary_op(&rhs, $op)
            }
        }
    };
}

impl_binop!(Add, add, Op::Add);
impl_binop!(Sub, sub, Op::Sub);
impl_binop!(Mul, mul, Op::Mul);
impl_binop!(Div, div, Op::Div);
