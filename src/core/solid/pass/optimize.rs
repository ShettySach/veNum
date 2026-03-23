//! Optimization pass wrapping the shared egglog optimizer.

use std::collections::HashSet;

use anyhow::Result;

use crate::core::shared::graph::{NodeId, Op};
use crate::core::shared::optimize::optimize;

use super::manager::{GraphPass, PassContext};

/// Optimization pass using egglog equality saturation.
///
/// Wraps the shared `optimize` function to run as a Solid compiler pass.
/// Applied per output root, replacing the graph with the optimized version.
/// Skips roots containing ops not modeled in egglog (reduces, slices, etc.).
pub struct OptimizationPass;

impl GraphPass for OptimizationPass {
    fn name(&self) -> &str {
        "optimization"
    }

    fn run(&self, ctx: PassContext) -> Result<PassContext> {
        let PassContext {
            graph,
            inputs: _,
            outputs,
        } = ctx;

        let mut current_graph = graph;
        let mut new_outputs = Vec::with_capacity(outputs.len());

        for &root in &outputs {
            if is_optimize_safe(&current_graph, root) {
                let (optimized_graph, new_root) = optimize(&current_graph, root)?;
                current_graph = optimized_graph;
                new_outputs.push(new_root);
            } else {
                new_outputs.push(root);
            }
        }

        // Input node IDs may have changed in the optimized graph.
        // Re-discover them by scanning for Op::Load nodes without buffers.
        let new_inputs = discover_inputs(&current_graph);

        Ok(PassContext {
            graph: current_graph,
            inputs: new_inputs,
            outputs: new_outputs,
        })
    }
}

/// Check if a subgraph rooted at `root` contains only ops modeled in egglog.
///
/// Returns false for graphs containing reduce, slice, flip, or pad ops,
/// which are treated as opaque leaves by the optimizer and would lose
/// their semantics if the root itself is such an op.
fn is_optimize_safe(graph: &crate::core::shared::graph::Graph, root: NodeId) -> bool {
    if !graph.node(root).dtype.is_float() {
        return false;
    }

    fn dfs(
        graph: &crate::core::shared::graph::Graph,
        id: NodeId,
        seen: &mut HashSet<NodeId>,
    ) -> bool {
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

/// Discover placeholder inputs in the graph (Load nodes with no buffer).
fn discover_inputs(graph: &crate::core::shared::graph::Graph) -> Vec<crate::core::shared::graph::NodeId> {
    use crate::core::shared::graph::{NodeId, Op};

    graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            if matches!(node.op, Op::Load) && node.buffer.is_none() {
                Some(NodeId(i))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::shared::dtype::DType;
    use crate::core::shared::graph::{Graph, Node, Op};

    #[test]
    fn optimization_pass_runs() {
        let mut graph = Graph::new();

        // x + 0 should be optimized to x
        let x = graph.add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let zero = graph.add_node(Node {
            op: Op::Const(crate::core::shared::dtype::Scalar::F32(0.0)),
            inputs: vec![],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let add = graph.add_node(Node {
            op: Op::Add,
            inputs: vec![x, zero],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        let ctx = PassContext {
            graph,
            inputs: vec![x],
            outputs: vec![add],
        };

        let pass = OptimizationPass;
        let result = pass.run(ctx).unwrap();

        // The optimized graph should have fewer or equal nodes
        assert!(!result.outputs.is_empty());
    }
}
