use std::collections::HashMap;

use crate::core::graph::{Graph, NodeId};

/// Policy for determining when nodes can be fused together.
pub trait FusionPolicy: Send + Sync {
    /// Can this node be inlined into its consumer's kernel?
    fn can_inline(
        &self,
        graph: &Graph,
        node_id: NodeId,
        consumer_id: NodeId,
        consumer_counts: &HashMap<NodeId, usize>,
    ) -> bool;
}
