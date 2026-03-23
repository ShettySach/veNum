//! Pass manager and graph pass trait.

use anyhow::Result;

use crate::core::shared::graph::{Graph, NodeId};

/// Context passed through the pass pipeline.
///
/// Holds the current state of compilation as passes transform it.
pub struct PassContext {
    /// The computation graph (mutated by passes).
    pub graph: Graph,

    /// Input node IDs (placeholders).
    pub inputs: Vec<NodeId>,

    /// Output node IDs (roots).
    pub outputs: Vec<NodeId>,
}

/// A compiler pass that transforms the graph.
///
/// Passes are applied sequentially by the `PassManager`. Each pass
/// receives the current `PassContext` and returns a (potentially modified) one.
pub trait GraphPass: Send + Sync {
    /// Name of this pass (for debugging/logging).
    fn name(&self) -> &str;

    /// Run this pass on the given context.
    fn run(&self, ctx: PassContext) -> Result<PassContext>;
}

/// Manages and runs a sequence of compiler passes.
pub struct PassManager {
    passes: Vec<Box<dyn GraphPass>>,
}

impl PassManager {
    /// Create a new empty pass manager.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the pipeline.
    pub fn add_pass(&mut self, pass: impl GraphPass + 'static) {
        self.passes.push(Box::new(pass));
    }

    /// Run all passes in order on the given context.
    pub fn run(&self, mut ctx: PassContext) -> Result<PassContext> {
        for pass in &self.passes {
            ctx = pass.run(ctx)?;
        }
        Ok(ctx)
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::shared::dtype::DType;
    use crate::core::shared::graph::{Node, Op};

    /// A no-op pass for testing.
    struct NoOpPass;

    impl GraphPass for NoOpPass {
        fn name(&self) -> &str {
            "no-op"
        }

        fn run(&self, ctx: PassContext) -> Result<PassContext> {
            Ok(ctx)
        }
    }

    #[test]
    fn pass_manager_runs_passes() {
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

        let ctx = PassContext {
            graph,
            inputs: vec![a],
            outputs: vec![b],
        };

        let mut pm = PassManager::new();
        pm.add_pass(NoOpPass);

        let result = pm.run(ctx).unwrap();
        assert_eq!(result.inputs.len(), 1);
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.graph.nodes.len(), 2);
    }
}
