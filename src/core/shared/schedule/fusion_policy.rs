use std::collections::HashMap;

use crate::core::shared::graph::{Graph, NodeId};

/// Policy for determining when nodes can be fused together.
///
/// Different execution modes have different fusion strategies:
/// - **Liquid** (per-tensor JIT): Conservative, only inline single-consumer nodes
/// - **Solid** (whole-program AOT): Aggressive, can inline multi-consumer within compilation unit
pub trait FusionPolicy: Send + Sync {
    /// Can this node be inlined into its consumer's kernel?
    ///
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `node_id` - The node being considered for inlining
    /// * `consumer_id` - The consumer that would absorb this node
    /// * `consumer_counts` - How many times each node is used as an input
    fn can_inline(
        &self,
        graph: &Graph,
        node_id: NodeId,
        consumer_id: NodeId,
        consumer_counts: &HashMap<NodeId, usize>,
    ) -> bool;
}
