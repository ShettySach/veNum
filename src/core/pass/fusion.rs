//! Global fusion pass.
//!
//! Uses the shared scheduling infrastructure with `DefaultFusionPolicy`
//! to produce a schedule of fused kernels for the entire program.

use anyhow::Result;

use crate::core::fusion_policy::DefaultFusionPolicy;
use crate::core::graph::NodeId;
use crate::core::schedule::{build_schedule_with_policy, ScheduleItem};

use super::manager::{GraphPass, PassContext};

/// Result of the fusion pass: the schedule for each output root.
pub struct FusionResult {
    /// Schedules per output root, in the same order as `PassContext::outputs`.
    pub schedules: Vec<Vec<ScheduleItem>>,
}

/// Global kernel fusion pass.
///
/// Applies the `DefaultFusionPolicy` to produce fused kernel schedules.
/// The policy aggressively inlines multi-consumer nodes, allowing
/// inlining since we have whole-program visibility.
pub struct FusionPass {
    /// Nodes that must materialize (inputs + outputs).
    boundary_nodes: Vec<NodeId>,
}

impl FusionPass {
    /// Create a new fusion pass.
    ///
    /// `boundary_nodes` are nodes that must not be inlined (inputs, outputs).
    pub fn new(boundary_nodes: Vec<NodeId>) -> Self {
        Self { boundary_nodes }
    }

    /// Build schedules for all output roots.
    pub fn build_schedules(&self, ctx: &PassContext) -> FusionResult {
        let mut policy = DefaultFusionPolicy::new();
        for &node_id in &self.boundary_nodes {
            policy.add_boundary(node_id);
        }

        let schedules = ctx
            .outputs
            .iter()
            .map(|&root| build_schedule_with_policy(&ctx.graph, root, &policy))
            .collect();

        FusionResult { schedules }
    }
}

impl GraphPass for FusionPass {
    fn name(&self) -> &str {
        "fusion"
    }

    fn run(&self, ctx: PassContext) -> Result<PassContext> {
        // The fusion pass doesn't mutate the graph; it produces schedules
        // that are consumed by later compilation stages.
        // The schedules are accessible via `build_schedules()`.
        // Pass the context through unchanged.
        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dtype::DType;
    use crate::core::graph::{Graph, Node, Op};

    #[test]
    fn fusion_pass_builds_schedules() {
        let mut graph = Graph::new();

        let a = graph.add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let b = graph.add_node(Node {
            op: Op::Exp,
            inputs: vec![a],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let c = graph.add_node(Node {
            op: Op::Neg,
            inputs: vec![b],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        let ctx = PassContext {
            graph,
            inputs: vec![a],
            outputs: vec![c],
        };

        let pass = FusionPass::new(vec![a, c]);
        let result = pass.build_schedules(&ctx);

        assert_eq!(result.schedules.len(), 1);
        // exp + neg should be fused into a single kernel
        assert!(!result.schedules[0].is_empty());
    }

    #[test]
    fn fusion_pass_multi_output() {
        let mut graph = Graph::new();

        let a = graph.add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let b = graph.add_node(Node {
            op: Op::Exp,
            inputs: vec![a],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let c = graph.add_node(Node {
            op: Op::Neg,
            inputs: vec![a],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        let ctx = PassContext {
            graph,
            inputs: vec![a],
            outputs: vec![b, c],
        };

        let pass = FusionPass::new(vec![a, b, c]);
        let result = pass.build_schedules(&ctx);

        assert_eq!(result.schedules.len(), 2);
    }
}
