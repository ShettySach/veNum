use anyhow::{anyhow, bail, Result};

use std::{cmp::Ordering, sync::Arc};

use crate::core::{
    errors::{ExpansionError, ReshapeError, TransposeError, UnsqueezeError},
    iters::Indexer,
};

use super::{
    context::Context,
    graph::{Graph, NodeId, Op},
    jit::compile_kernel,
    optimize, render,
    schedule::{build_schedule, ScheduleItem},
};

#[derive(Clone, Debug, PartialEq)]
pub struct RealizedTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl RealizedTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Clone)]
pub struct Tensor {
    graph: Arc<std::sync::Mutex<Graph>>,
    id: NodeId,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn from_slice(cx: &Context, data: &[f32], shape: Vec<usize>) -> Self {
        let data = Arc::new(data.to_vec());

        let graph = cx.graph();
        let id = {
            let mut g = graph.lock().unwrap();
            g.load(data, shape.clone())
        };

        Self { graph, id, shape }
    }

    pub fn constant(cx: &Context, value: f32, shape: Vec<usize>) -> Self {
        let graph = cx.graph();
        let id = {
            let mut g = graph.lock().unwrap();
            g.constant(value, shape.clone())
        };

        Self { graph, id, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    // -------- elementwise lazy ops --------

    fn binary_op(&self, rhs: &Tensor, op: Op) -> Tensor {
        debug_assert!(
            Arc::ptr_eq(&self.graph, &rhs.graph),
            "binary_op requires both tensors to share the same Context"
        );
        let mut graph = self.graph.lock().unwrap();
        let id = graph.binary(op, self.id, rhs.id);
        let shape = graph.node(id).shape.clone();
        drop(graph);

        Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        }
    }

    fn unary_op(&self, op: Op) -> Tensor {
        let mut graph = self.graph.lock().unwrap();
        let id = graph.unary(op, self.id);
        let shape = graph.node(id).shape.clone();
        drop(graph);

        Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        }
    }

    pub fn exp(&self) -> Tensor {
        self.unary_op(Op::Exp)
    }

    pub fn ln(&self) -> Tensor {
        self.unary_op(Op::Ln)
    }

    pub fn sqrt(&self) -> Tensor {
        self.unary_op(Op::Sqrt)
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

        let mut graph = self.graph.lock().unwrap();
        let id = graph.reshape(self.id, sizes.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: sizes,
        })
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

        let shape = permutation
            .iter()
            .map(|&i| self.shape[i])
            .collect::<Vec<_>>();

        let mut graph = self.graph.lock().unwrap();
        let id = graph.permute(self.id, permutation, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if rank < 2 || dim_1 >= rank || dim_2 >= rank {
            bail!(TransposeError);
        }

        let mut shape = self.shape.clone();
        shape.swap(dim_1, dim_2);

        let mut graph = self.graph.lock().unwrap();
        let id = graph.transpose(self.id, dim_1, dim_2, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
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

        let mut graph = self.graph.lock().unwrap();
        let id = graph.expand(self.id, expansions.clone(), expansions.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: expansions,
        })
    }

    pub fn slice(&self, ranges: Vec<(usize, usize)>) -> Result<Tensor> {
        if ranges.len() != self.shape.len() {
            bail!(anyhow!(
                "slice ranges rank mismatch: got {}, expected {}",
                ranges.len(),
                self.shape.len()
            ));
        }

        let mut out = Vec::with_capacity(self.shape.len());
        for (dim, &(start, end_raw)) in ranges.iter().enumerate() {
            let size = self.shape[dim];
            let end = if end_raw == 0 { size } else { end_raw };
            if start > end || end > size {
                bail!(anyhow!(
                    "slice range {:?} out of bounds for dim {} with size {}",
                    (start, end),
                    dim,
                    size
                ));
            }
            out.push(end - start);
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.slice(self.id, ranges, out.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: out,
        })
    }

    pub fn flip(&self, flips: Vec<usize>) -> Result<Tensor> {
        for &d in &flips {
            if d >= self.shape.len() {
                bail!(anyhow!(
                    "flip dimension {} out of bounds for rank {}",
                    d,
                    self.shape.len()
                ));
            }
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.flip(self.id, flips, self.shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: self.shape.clone(),
        })
    }

    pub fn squeeze(&self) -> Result<Tensor> {
        let mut shape: Vec<usize> = self.shape.iter().copied().filter(|&s| s != 1).collect();
        if shape.is_empty() {
            shape.push(1);
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.squeeze(self.id, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
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

        let mut graph = self.graph.lock().unwrap();
        let id = graph.unsqueeze(self.id, new_rank, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn pad(&self, constant: f32, padding: Vec<(usize, usize)>) -> Result<Tensor> {
        let mut pad = padding;
        pad.resize(self.shape.len(), (0, 0));

        let mut shape = Vec::with_capacity(self.shape.len());
        for (s, (l, r)) in self.shape.iter().copied().zip(pad.iter().copied()) {
            shape.push(l + s + r);
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.pad(self.id, constant, pad, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    // -------- reduce lazy ops --------

    pub fn sum_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.sum(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn product_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.prod(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn max_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.max(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn min_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.min(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
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
        (&lhs * &rhs).sum_dims(vec![blen + 1], false)
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

    // -------- realize --------

    pub fn realize(&self) -> Result<RealizedTensor> {
        let graph = self.graph.lock().unwrap();

        // Fast path: already-backed leaf.
        let root_node = graph.node(self.id);
        if let Some(ref buffer) = root_node.buffer {
            let data = buffer.as_f32().to_vec();
            let shape = root_node.shape.clone();
            drop(graph);
            return Ok(RealizedTensor::new(data, shape));
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
                buffer.as_f32().to_vec(),
                root_node.shape.clone(),
            ));
        }
        if let Op::Const(val) = root_node.op {
            let numel: usize = self.shape.iter().product();
            return Ok(RealizedTensor::new(vec![val; numel], self.shape.clone()));
        }

        let schedule = build_schedule(graph, root);
        let mut realized: std::collections::HashMap<NodeId, Vec<f32>> =
            std::collections::HashMap::new();

        for item in &schedule {
            match item {
                ScheduleItem::Fused(kernel) => {
                    let compiled = compile_kernel(graph, kernel)?;

                    let input_ptrs: Vec<*const f32> = kernel
                        .input_buffers
                        .iter()
                        .map(|&buf_id| {
                            let node = graph.node(buf_id);
                            if let Some(ref buffer) = node.buffer {
                                buffer.as_f32_ptr()
                            } else if let Some(data) = realized.get(&buf_id) {
                                data.as_ptr()
                            } else {
                                panic!("Input buffer {:?} not realized and has no data", buf_id);
                            }
                        })
                        .collect();

                    let mut output = vec![0.0f32; kernel.numel];
                    unsafe {
                        compiled.execute(&input_ptrs, output.as_mut_ptr(), kernel.numel);
                    }
                    realized.insert(kernel.root, output);
                }
                ScheduleItem::Shape(shape_item) => {
                    let input_node = graph.node(shape_item.input);
                    let input_shape = input_node.shape.clone();

                    let input_data = if let Some(data) = realized.get(&shape_item.input) {
                        data.clone()
                    } else if let Some(ref buf) = input_node.buffer {
                        buf.as_f32().to_vec()
                    } else {
                        bail!("Shape op input {:?} is not realized", shape_item.input);
                    };

                    let output = execute_shape_op(
                        &shape_item.op,
                        &input_data,
                        &input_shape,
                        &shape_item.shape,
                    )?;
                    realized.insert(shape_item.root, output);
                }
                ScheduleItem::Reduce(reduce_item) => {
                    let input_node = graph.node(reduce_item.input);
                    let input_shape = input_node.shape.clone();

                    let input_data = if let Some(data) = realized.get(&reduce_item.input) {
                        data.clone()
                    } else if let Some(ref buf) = input_node.buffer {
                        buf.as_f32().to_vec()
                    } else {
                        bail!("Reduce op input {:?} is not realized", reduce_item.input);
                    };

                    let output = execute_reduce_op(
                        &reduce_item.op,
                        &input_data,
                        &input_shape,
                        &reduce_item.shape,
                    )?;
                    realized.insert(reduce_item.root, output);
                }
            }
        }

        let data = realized
            .remove(&root)
            .ok_or_else(|| anyhow!("Root node was not realized"))?;

        Ok(RealizedTensor::new(data, graph.node(root).shape.clone()))
    }
}

// -------- shape-op execution helpers --------

fn execute_shape_op(
    op: &Op,
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
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
            execute_pad(input_data, input_shape, output_shape, *constant, padding)
        }
        _ => bail!("execute_shape_op called with non-shape op: {:?}", op),
    }
}

fn execute_reduce_op(
    op: &Op,
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
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
                    .unwrap_or(0.0)
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
                    .unwrap_or(0.0)
            },
        ),
        _ => bail!("execute_reduce_op called with non-reduce op: {:?}", op),
    }
}

fn execute_reduce(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    dimensions: &[usize],
    keepdims: bool,
    reducer: impl Fn(&[f32]) -> f32,
) -> Result<Vec<f32>> {
    validate_reduce_dims(input_shape, dimensions)?;

    let expected_shape = reduced_shape(input_shape, dimensions, keepdims)?;
    if expected_shape != output_shape {
        bail!(
            "reduce output shape mismatch: expected {:?}, got {:?}",
            expected_shape,
            output_shape
        );
    }

    let mut out = Vec::with_capacity(output_shape.iter().product());

    for out_idx in Indexer::new(output_shape) {
        let mut fixed = vec![None; input_shape.len()];

        if keepdims {
            for d in 0..input_shape.len() {
                if !dimensions.contains(&d) {
                    fixed[d] = Some(out_idx[d]);
                }
            }
        } else {
            let mut out_pos = 0usize;
            (0..input_shape.len()).for_each(|d| {
                if !dimensions.contains(&d) {
                    fixed[d] = Some(out_idx[out_pos]);
                    out_pos += 1;
                }
            });
        }

        let mut reduced_values = Vec::new();
        collect_reduce_values(
            input_data,
            input_shape,
            dimensions,
            &fixed,
            0,
            &mut vec![0; input_shape.len()],
            &mut reduced_values,
        );

        out.push(reducer(&reduced_values));
    }

    Ok(out)
}

fn collect_reduce_values(
    input_data: &[f32],
    input_shape: &[usize],
    dimensions: &[usize],
    fixed: &[Option<usize>],
    dim: usize,
    current: &mut Vec<usize>,
    values: &mut Vec<f32>,
) {
    if dim == input_shape.len() {
        let off = idx_to_offset(current, input_shape);
        values.push(input_data[off]);
        return;
    }

    if dimensions.contains(&dim) {
        for i in 0..input_shape[dim] {
            current[dim] = i;
            collect_reduce_values(
                input_data,
                input_shape,
                dimensions,
                fixed,
                dim + 1,
                current,
                values,
            );
        }
    } else if let Some(idx) = fixed[dim] {
        current[dim] = idx;
        collect_reduce_values(
            input_data,
            input_shape,
            dimensions,
            fixed,
            dim + 1,
            current,
            values,
        );
    }
}

fn reduced_shape(shape: &[usize], dimensions: &[usize], keepdims: bool) -> Result<Vec<usize>> {
    validate_reduce_dims(shape, dimensions)?;

    let mut out = Vec::new();
    for (d, &size) in shape.iter().enumerate() {
        if dimensions.contains(&d) {
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

fn execute_reshape(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
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

fn execute_permute(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    perm: &[usize],
) -> Result<Vec<f32>> {
    if perm.len() != input_shape.len() || output_shape.len() != input_shape.len() {
        bail!("invalid permute rank");
    }
    let mut out = vec![0.0f32; output_shape.iter().product()];

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

fn execute_transpose(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    dim_1: usize,
    dim_2: usize,
) -> Result<Vec<f32>> {
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

fn execute_expand(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    if input_shape.len() != output_shape.len() {
        bail!("expand rank mismatch");
    }

    let mut out = vec![0.0f32; output_shape.iter().product()];
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

fn execute_slice(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<f32>> {
    if ranges.len() != input_shape.len() || output_shape.len() != input_shape.len() {
        bail!("slice rank mismatch");
    }

    let mut out = vec![0.0f32; output_shape.iter().product()];
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

fn execute_flip(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    flips: &[usize],
) -> Result<Vec<f32>> {
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

    let mut out = vec![0.0f32; output_shape.iter().product()];
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

fn execute_pad(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    constant: f32,
    padding: &[(usize, usize)],
) -> Result<Vec<f32>> {
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
    buffer_map: &std::collections::HashMap<*const Vec<f32>, NodeId>,
) -> NodeId {
    if let Some(&mapped) = id_map.get(&src_id) {
        return mapped;
    }

    let node = src_graph.node(src_id);

    if let (Op::Load, Some(super::dtype::Buffer::F32(ref arc))) = (&node.op, &node.buffer) {
        if let Some(&existing_id) = buffer_map.get(&Arc::as_ptr(arc)) {
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
            type Output = Tensor;

            fn $method(self, rhs: &'b Tensor) -> Tensor {
                self.binary_op(rhs, $op)
            }
        }

        // Tensor op &Tensor
        impl<'b> std::ops::$trait<&'b Tensor> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: &'b Tensor) -> Tensor {
                self.binary_op(rhs, $op)
            }
        }

        // &Tensor op Tensor
        impl<'a> std::ops::$trait<Tensor> for &'a Tensor {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Tensor {
                self.binary_op(&rhs, $op)
            }
        }

        // Tensor op Tensor
        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Tensor {
                self.binary_op(&rhs, $op)
            }
        }
    };
}

impl_binop!(Add, add, Op::Add);
impl_binop!(Sub, sub, Op::Sub);
impl_binop!(Mul, mul, Op::Mul);
impl_binop!(Div, div, Op::Div);
