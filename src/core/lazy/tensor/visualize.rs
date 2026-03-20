use anyhow::Result;

use super::super::{optimize, render};
use super::helpers::{clone_reachable_subgraph, is_optimize_safe};
use super::Tensor;

impl Tensor {
    pub fn render_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        render::render_dag(&graph, self.id)
    }

    pub fn render_fused_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        render::render_fused_dag(&graph, self.id)
    }

    pub fn render_optimized_dag(&self) -> Result<String> {
        let graph = self.graph.lock().unwrap();
        if !is_optimize_safe(&graph, self.id) {
            return Ok(render::render_dag(&graph, self.id));
        }
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id)?;
        Ok(render::render_dag(&opt_graph, opt_root))
    }

    pub fn render_optimized_fused_dag(&self) -> Result<String> {
        let graph = self.graph.lock().unwrap();
        if !is_optimize_safe(&graph, self.id) {
            return Ok(render::render_fused_dag(&graph, self.id));
        }
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id)?;
        Ok(render::render_fused_dag(&opt_graph, opt_root))
    }

    pub fn render_kernels(&self) -> Result<String> {
        use std::fmt::Write;

        use super::super::jit::compile_kernel;
        use super::super::schedule::{build_schedule, ScheduleItem};

        let graph = self.graph.lock().unwrap();
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
                        if let Some(Some(tracker)) = kernel.input_trackers.get(buf_id.0) {
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

                    if kernel.num_absorbed_shape_ops > 0 {
                        writeln!(
                            out,
                            "    absorbed {} shape ops",
                            kernel.num_absorbed_shape_ops
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

        Ok(out)
    }
}
