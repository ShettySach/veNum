use anyhow::{anyhow, bail, Result};
use std::sync::Arc;

use crate::core::lazy::{
    context::SharedBufferPool,
    plan::{build_plan, ExecItem, ExecutionPlan, GraphSignature},
    tensor::{
        helpers::{clone_reachable_subgraph, is_optimize_safe},
        Tensor,
    },
};
use crate::core::shared::{
    dtype::{Buffer, RealizedTensor},
    exec,
    graph::{Graph, NodeId},
    optimize,
};

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
            let mut cache = plan_cache_handle.lock().unwrap();
            if let Some(plan) = cache.get_cloned(&sig) {
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

        let backend = self.cx.backend();
        let plan = build_plan(
            exec_graph,
            exec_root,
            &self.cx.kernel_cache(),
            backend.as_ref(),
            &self.shape,
        )?;
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

/// Precompute which intermediates can be released at each plan step.
fn compute_release_lists(plan: &ExecutionPlan) -> Vec<Vec<NodeId>> {
    let mut last_use: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
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

    let mut release_at = vec![Vec::new(); plan.items.len()];
    for (node_id, step) in last_use {
        if node_id != plan.output {
            release_at[step].push(node_id);
        }
    }
    release_at
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

    let release_at = compute_release_lists(plan);

    let mut realized: std::collections::HashMap<NodeId, Buffer> = std::collections::HashMap::new();

    for (step, item) in plan.items.iter().enumerate() {
        match item {
            ExecItem::Kernel {
                kernel,
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
                kernel.execute(&input_ptrs, output_bytes.as_mut_ptr(), numel);
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

        // Drop dead intermediates.
        for &node_id in &release_at[step] {
            let _ = realized.remove(&node_id);
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
