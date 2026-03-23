use anyhow::Result;
use std::sync::Arc;

use crate::core::liquid::backend::Backend;
use crate::core::liquid::jit::KernelSignature;
use crate::core::liquid::kernel::ExecutableKernel;
use crate::core::liquid::lru_cache::LruCache;
use crate::core::liquid::plan::{ExecItem, ExecutionPlan};
use crate::core::liquid::schedule::{build_schedule, ScheduleItem};
use crate::core::shared::graph::{Graph, NodeId, Op};

/// Build an `ExecutionPlan` from an execution graph.
pub fn build_plan(
    exec_graph: Graph,
    root: NodeId,
    kernel_cache: &std::sync::Mutex<LruCache<KernelSignature, Arc<dyn ExecutableKernel>>>,
    backend: &dyn Backend,
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
                    if let Some(cached) = cache.get_cloned(&sig) {
                        cached
                    } else {
                        let compiled = backend.compile(&exec_graph, kernel, false)?;
                        cache.insert(sig, Arc::clone(&compiled));
                        compiled
                    }
                };

                items.push(ExecItem::Kernel {
                    kernel: compiled,
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
