use anyhow::{anyhow, bail, Result};
use std::sync::Arc;

use super::super::context::SharedBufferPool;
use super::super::dtype::{Buffer, RealizedTensor};
use super::super::exec;
use super::super::graph::{Graph, NodeId, Op};
use super::super::jit::{compile_kernel, KernelSignature};
use super::super::optimize;
use super::super::plan::{ExecItem, ExecutionPlan, GraphSignature};
use super::super::schedule::{build_schedule, ScheduleItem};
use super::helpers::{clone_reachable_subgraph, is_optimize_safe};
use super::Tensor;

impl Tensor {
    pub fn realize(&self) -> Result<RealizedTensor> {
        let graph_handle = self.cx.graph();
        let graph = graph_handle.lock().unwrap();

        // Fast path: already-backed leaf.
        let root_node = graph.node(self.id);
        if let Some(ref buffer) = root_node.buffer {
            return Ok(RealizedTensor::new(buffer.clone(), root_node.shape.clone()));
        }

        // Compute the graph signature for plan cache lookup.
        let sig = GraphSignature::from_graph(&graph, self.id);

        // Check the plan cache.
        {
            let plan_cache_handle = self.cx.plan_cache();
            let cache = plan_cache_handle.lock().unwrap();
            if let Some(plan) = cache.get(&sig) {
                let plan = Arc::clone(plan);
                drop(cache);
                drop(graph);
                return run_plan(&plan, &self.cx.buffer_pool());
            }
        }

        // Cache miss — build the execution graph and plan.
        let optimize_safe = is_optimize_safe(&graph, self.id);
        let (exec_graph, exec_root) = if optimize_safe {
            optimize::optimize(&graph, self.id)?
        } else {
            clone_reachable_subgraph(&graph, self.id)
        };
        drop(graph);

        let plan = build_plan(exec_graph, exec_root, &self.cx.kernel_cache(), &self.shape)?;
        let plan = Arc::new(plan);

        // Cache the plan.
        {
            let plan_cache_handle = self.cx.plan_cache();
            let mut cache = plan_cache_handle.lock().unwrap();
            cache.insert(sig, Arc::clone(&plan));
        }

        run_plan(&plan, &self.cx.buffer_pool())
    }
}

/// Build an `ExecutionPlan` from an execution graph.
fn build_plan(
    exec_graph: Graph,
    root: NodeId,
    kernel_cache: &std::sync::Mutex<
        std::collections::HashMap<KernelSignature, Arc<super::super::jit::CompiledKernel>>,
    >,
    tensor_shape: &[usize],
) -> Result<ExecutionPlan> {
    let root_node = exec_graph.node(root);

    // Handle already-backed root in exec graph.
    if root_node.buffer.is_some() {
        return Ok(ExecutionPlan {
            items: vec![],
            output: root,
            output_shape: tensor_shape.to_vec(),
            exec_graph,
        });
    }

    // Handle const root — egglog may collapse to a scalar Const(v) with shape [1],
    // but the tensor expects the original shape, so fill to `tensor_shape`.
    if let Op::Const(val) = root_node.op {
        let numel: usize = tensor_shape.iter().product();
        return Ok(ExecutionPlan {
            items: vec![ExecItem::ConstFill {
                value: val,
                output: root,
                numel,
            }],
            output: root,
            output_shape: tensor_shape.to_vec(),
            exec_graph,
        });
    }

    let schedule = build_schedule(&exec_graph, root);

    let mut items = Vec::with_capacity(schedule.len());

    for item in &schedule {
        match item {
            ScheduleItem::Fused(kernel) => {
                let sig = KernelSignature::from_kernel(&exec_graph, kernel);
                let compiled = {
                    let mut cache = kernel_cache.lock().unwrap();
                    if let Some(cached) = cache.get(&sig) {
                        Arc::clone(cached)
                    } else {
                        let compiled = Arc::new(compile_kernel(&exec_graph, kernel, false)?);
                        cache.insert(sig, Arc::clone(&compiled));
                        compiled
                    }
                };

                items.push(ExecItem::Kernel {
                    compiled,
                    inputs: kernel.input_buffers.clone(),
                    output: kernel.root,
                });
            }
            ScheduleItem::Shape(shape_item) => {
                items.push(ExecItem::Shape {
                    op: shape_item.op.clone(),
                    input: shape_item.input,
                    output: shape_item.root,
                });
            }
            ScheduleItem::Reduce(reduce_item) => {
                items.push(ExecItem::Reduce {
                    op: reduce_item.op.clone(),
                    input: reduce_item.input,
                    output: reduce_item.root,
                });
            }
        }
    }

    let output_shape = exec_graph.node(root).shape.clone();

    Ok(ExecutionPlan {
        items,
        output: root,
        output_shape,
        exec_graph,
    })
}

/// Compute the last step index at which each intermediate NodeId is read.
fn compute_last_use(plan: &ExecutionPlan) -> std::collections::HashMap<NodeId, usize> {
    let mut last_use = std::collections::HashMap::new();
    for (step, item) in plan.items.iter().enumerate() {
        match item {
            ExecItem::Kernel { inputs, .. } => {
                for &id in inputs {
                    last_use.insert(id, step);
                }
            }
            ExecItem::Shape { input, .. } | ExecItem::Reduce { input, .. } => {
                last_use.insert(*input, step);
            }
            ExecItem::ConstFill { .. } => {}
        }
    }
    last_use
}

/// Execute a cached plan to produce a realized tensor.
fn run_plan(plan: &ExecutionPlan, pool: &SharedBufferPool) -> Result<RealizedTensor> {
    let graph = &plan.exec_graph;

    // If plan has no items, the output must be a direct buffer in the exec graph.
    if plan.items.is_empty() {
        let node = graph.node(plan.output);
        if let Some(ref buffer) = node.buffer {
            return Ok(RealizedTensor::new(
                buffer.clone(),
                plan.output_shape.clone(),
            ));
        }
        bail!("Empty plan with no output buffer available");
    }

    let last_use = compute_last_use(plan);

    let mut realized: std::collections::HashMap<NodeId, Buffer> = std::collections::HashMap::new();

    for (step, item) in plan.items.iter().enumerate() {
        match item {
            ExecItem::Kernel {
                compiled,
                inputs,
                output,
            } => {
                let out_node = graph.node(*output);
                let dtype = out_node.dtype;
                let numel: usize = out_node.shape.iter().product();
                let input_ptrs: Vec<*const u8> = inputs
                    .iter()
                    .map(|&buf_id| -> Result<*const u8> {
                        if let Some(buf) = realized.get(&buf_id) {
                            Ok(buf.as_ptr_u8())
                        } else {
                            let node = graph.node(buf_id);
                            if let Some(ref buffer) = node.buffer {
                                Ok(buffer.as_ptr_u8())
                            } else {
                                bail!("Input buffer {:?} not realized and has no data", buf_id);
                            }
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                let mut output_bytes = pool.lock().unwrap().acquire(dtype, numel);
                unsafe {
                    compiled.execute(&input_ptrs, output_bytes.as_mut_ptr(), numel);
                }
                realized.insert(*output, exec::buffer_from_bytes(output_bytes, dtype));
            }
            ExecItem::Shape { op, input, output } => {
                let input_shape = graph.node(*input).shape.clone();
                let output_shape = graph.node(*output).shape.clone();
                let input_buf = resolve_buffer(graph, *input, &realized, "Shape op")?;
                let result =
                    exec::execute_shape_op_typed(op, &input_buf, &input_shape, &output_shape)?;
                realized.insert(*output, result);
            }
            ExecItem::Reduce { op, input, output } => {
                let input_shape = graph.node(*input).shape.clone();
                let output_shape = graph.node(*output).shape.clone();
                let input_buf = resolve_buffer(graph, *input, &realized, "Reduce op")?;
                let result =
                    exec::execute_reduce_op_typed(op, &input_buf, &input_shape, &output_shape)?;
                realized.insert(*output, result);
            }
            ExecItem::ConstFill {
                value,
                output,
                numel,
            } => {
                let buffer = exec::scalar_fill_buffer(*value, *numel);
                realized.insert(*output, buffer);
            }
        }

        // Release dead intermediates back to the pool.
        // Only release intermediates that are not the final output.
        for (&node_id, &last_step) in &last_use {
            if last_step == step && node_id != plan.output {
                if let Some(buf) = realized.remove(&node_id) {
                    let dtype = buf.dtype();
                    let numel = buf.len();
                    // We can't recover the raw Vec<u8> from a Buffer, so just drop it.
                    // The pool is used for kernel output allocations above.
                    drop(buf);
                    let _ = (dtype, numel);
                }
            }
        }
    }

    let buf = realized
        .remove(&plan.output)
        .ok_or_else(|| anyhow!("Root node was not realized"))?;

    Ok(RealizedTensor::new(buf, plan.output_shape.clone()))
}

fn resolve_buffer(
    graph: &Graph,
    node_id: NodeId,
    realized: &std::collections::HashMap<NodeId, Buffer>,
    context: &str,
) -> Result<Buffer> {
    if let Some(buf) = realized.get(&node_id) {
        Ok(buf.clone())
    } else {
        let node = graph.node(node_id);
        if let Some(ref buffer) = node.buffer {
            Ok(buffer.clone())
        } else {
            bail!(
                "{} input {:?} not realized and has no data",
                context,
                node_id
            );
        }
    }
}
