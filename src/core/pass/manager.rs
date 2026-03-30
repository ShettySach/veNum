//! Pass context and graph pass trait.

use anyhow::Result;

use crate::core::graph::{Graph, NodeId};

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
/// Each pass receives the current `PassContext` and returns a (potentially modified) one.
pub trait GraphPass: Send + Sync {
    /// Name of this pass (for debugging/logging).
    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// Run this pass on the given context.
    fn run(&self, ctx: PassContext) -> Result<PassContext>;
}
