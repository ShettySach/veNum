use anyhow::{anyhow, bail, Result};

use super::super::dtype::{Buffer, RealizedTensor};
use super::super::exec;
use super::super::graph::{NodeId, Op};
use super::super::jit::{compile_kernel, KernelSignature};
use super::super::optimize;
use super::super::schedule::{build_schedule, ScheduleItem};
use super::helpers::{clone_reachable_subgraph, is_optimize_safe};
use super::Tensor;

impl Tensor {
    pub fn realize(&self) -> Result<RealizedTensor> {
        let graph = self.graph.lock().unwrap();

        // Fast path: already-backed leaf.
        let root_node = graph.node(self.id);
        if let Some(ref buffer) = root_node.buffer {
            return Ok(RealizedTensor::new(buffer.clone(), root_node.shape.clone()));
        }

        // Only optimize pure elementwise roots for now.
        let optimize_safe = is_optimize_safe(&graph, self.id);
        let (exec_graph, exec_root) = if optimize_safe {
            optimize::optimize(&graph, self.id)?
        } else {
            clone_reachable_subgraph(&graph, self.id)
        };
        drop(graph);

        let graph = &exec_graph;
        let root = exec_root;

        let root_node = graph.node(root);
        if let Some(ref buffer) = root_node.buffer {
            return Ok(RealizedTensor::new(buffer.clone(), root_node.shape.clone()));
        }
        if let Op::Const(val) = root_node.op {
            let numel: usize = self.shape.iter().product();
            let buffer = exec::scalar_fill_buffer(val, numel);
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
                            std::sync::Arc::clone(cached)
                        } else {
                            let compiled =
                                std::sync::Arc::new(compile_kernel(graph, kernel, false)?);
                            cache.insert(sig, std::sync::Arc::clone(&compiled));
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

                    let mut output_bytes = super::super::dtype::zeros(dtype, kernel.numel);
                    unsafe {
                        compiled.execute(&input_ptrs, output_bytes.as_mut_ptr(), kernel.numel);
                    }
                    realized.insert(kernel.root, exec::buffer_from_bytes(output_bytes, dtype));
                }
                ScheduleItem::Shape(shape_item) => {
                    let input_node = graph.node(shape_item.input);
                    let input_shape = &input_node.shape;

                    let input_buf =
                        exec::get_realized_buffer(graph, shape_item.input, &realized, "Shape op")?;
                    let output = exec::execute_shape_op_typed(
                        &shape_item.op,
                        &input_buf,
                        input_shape,
                        &shape_item.shape,
                    )?;
                    realized.insert(shape_item.root, output);
                }
                ScheduleItem::Reduce(reduce_item) => {
                    let input_node = graph.node(reduce_item.input);
                    let input_shape = &input_node.shape;

                    let input_buf = exec::get_realized_buffer(
                        graph,
                        reduce_item.input,
                        &realized,
                        "Reduce op",
                    )?;
                    let output = exec::execute_reduce_op_typed(
                        &reduce_item.op,
                        &input_buf,
                        input_shape,
                        &reduce_item.shape,
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
