//! Liquid-specific tensor visualization and debugging.

use anyhow::Result;
use std::collections::HashSet;
use std::fmt::Write;

use crate::core::liquid::context::LiquidContext;
use crate::core::liquid::render;
use crate::core::liquid::schedule::{build_schedule, ScheduleItem};
use crate::core::shared::graph::{Graph, NodeId, Op};
use crate::core::shared::optimize;
use crate::core::shared::tensor::Tensor;

use std::collections::HashMap;

/// Liquid-specific visualization extensions.
impl Tensor<LiquidContext> {
    pub fn render_dag(&self) -> String {
        let graph_handle = self.cx.graph();
        let graph = graph_handle.lock().unwrap();
        render::render_dag(&graph, self.id)
    }

    pub fn render_fused_dag(&self) -> String {
        let graph_handle = self.cx.graph();
        let graph = graph_handle.lock().unwrap();
        render::render_fused_dag(&graph, self.id)
    }

    pub fn render_optimized_dag(&self) -> Result<String> {
        let graph_handle = self.cx.graph();
        let graph = graph_handle.lock().unwrap();
        if is_optimize_safe(&graph, self.id) {
            let (opt_graph, opt_root) = optimize::optimize(&graph, self.id)?;
            Ok(render::render_dag(&opt_graph, opt_root))
        } else {
            Ok(render::render_dag(&graph, self.id))
        }
    }

    pub fn render_optimized_fused_dag(&self) -> Result<String> {
        let graph_handle = self.cx.graph();
        let graph = graph_handle.lock().unwrap();
        if is_optimize_safe(&graph, self.id) {
            let (opt_graph, opt_root) = optimize::optimize(&graph, self.id)?;
            Ok(render::render_fused_dag(&opt_graph, opt_root))
        } else {
            Ok(render::render_fused_dag(&graph, self.id))
        }
    }

    pub fn render_kernels(&self) -> Result<String> {
        let graph_handle = self.cx.graph();
        let graph = graph_handle.lock().unwrap();
        let optimize_safe = is_optimize_safe(&graph, self.id);
        let (exec_graph, exec_root) = if optimize_safe {
            optimize::optimize(&graph, self.id)?
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

                    let backend = self.cx.backend();
                    match backend.compile(&exec_graph, kernel, true) {
                        Ok(compiled) => {
                            if let Some(ir) = compiled.debug_ir() {
                                writeln!(out, "\n    --- IR ---").unwrap();
                                for line in ir.lines() {
                                    writeln!(out, "    {}", line).unwrap();
                                }
                            }
                        }
                        Err(e) => writeln!(out, "    (compilation error: {})", e).unwrap(),
                    };
                }
            }
        }

        Ok(out)
    }
}

// ==================== Helper Functions ====================

fn clone_reachable_subgraph(src: &Graph, root: NodeId) -> (Graph, NodeId) {
    use crate::core::shared::graph::Node;

    let mut dst = Graph::new();
    let mut id_map = HashMap::new();

    fn import_node(
        src_graph: &Graph,
        src_id: NodeId,
        dst_graph: &mut Graph,
        id_map: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if let Some(&mapped) = id_map.get(&src_id) {
            return mapped;
        }

        let node = src_graph.node(src_id);
        let new_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|&input_id| import_node(src_graph, input_id, dst_graph, id_map))
            .collect();

        let new_id = dst_graph.add_node(Node {
            op: node.op.clone(),
            inputs: new_inputs,
            shape: node.shape.clone(),
            dtype: node.dtype,
            buffer: node.buffer.clone(),
        });

        id_map.insert(src_id, new_id);
        new_id
    }

    let new_root = import_node(src, root, &mut dst, &mut id_map);
    (dst, new_root)
}

fn is_optimize_safe(graph: &Graph, root: NodeId) -> bool {
    // Only optimize float dtypes - egglog rules use float constants.
    if !graph.node(root).dtype.is_float() {
        return false;
    }

    fn dfs(graph: &Graph, id: NodeId, seen: &mut HashSet<NodeId>) -> bool {
        if !seen.insert(id) {
            return true;
        }

        let node = graph.node(id);
        match node.op {
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
            | Op::Reshape
            | Op::Permute(_)
            | Op::Transpose(_, _)
            | Op::Expand
            | Op::Squeeze
            | Op::Unsqueeze(_) => node.inputs.iter().all(|&inp| dfs(graph, inp, seen)),
            _ => false,
        }
    }

    let mut seen = HashSet::new();
    dfs(graph, root, &mut seen)
}
