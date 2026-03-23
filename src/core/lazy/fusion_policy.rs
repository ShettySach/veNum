use std::collections::HashMap;

use crate::core::shared::graph::{Graph, NodeId};
use crate::core::shared::schedule::FusionPolicy;

/// Liquid-style fusion policy (Tinygrad-like).
///
/// Conservative: only inline single-consumer elementwise nodes
/// with matching numel. This is safe for JIT compilation where
/// we don't have global graph visibility.
pub struct LiquidFusionPolicy;

impl FusionPolicy for LiquidFusionPolicy {
    fn can_inline(
        &self,
        graph: &Graph,
        node_id: NodeId,
        consumer_id: NodeId,
        consumer_counts: &HashMap<NodeId, usize>,
    ) -> bool {
        let node = graph.node(node_id);

        // Only inline elementwise operations
        if !node.op.is_elementwise() {
            return false;
        }

        // Only inline if single consumer (conservative for JIT)
        let count = *consumer_counts.get(&node_id).unwrap_or(&0);
        if count != 1 {
            return false;
        }

        // Must have matching numel for 1:1 element mapping
        node.numel() == graph.node(consumer_id).numel()
    }
}
