//! Fusion policy for Solid mode.

use std::collections::{HashMap, HashSet};

use crate::core::shared::graph::{Graph, NodeId};
use crate::core::shared::schedule::FusionPolicy;

/// Solid-style fusion policy (Luminal-like).
///
/// Aggressive: can inline multi-consumer nodes as long as they're not
/// at compilation boundaries. This enables more fusion opportunities
/// since we have whole-program visibility.
///
/// The policy is controlled by a set of "boundary" nodes that must
/// materialize (symbolic inputs, outputs, large intermediate values).
pub struct SolidFusionPolicy {
    /// Nodes at compilation boundary (must materialize)
    boundary: HashSet<NodeId>,
}

impl SolidFusionPolicy {
    /// Create a new Solid fusion policy.
    pub fn new() -> Self {
        Self {
            boundary: HashSet::new(),
        }
    }

    /// Mark a node as a boundary (must materialize).
    pub fn add_boundary(&mut self, node_id: NodeId) {
        self.boundary.insert(node_id);
    }

    /// Check if a node is a boundary.
    pub fn is_boundary(&self, node_id: NodeId) -> bool {
        self.boundary.contains(&node_id)
    }
}

impl Default for SolidFusionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionPolicy for SolidFusionPolicy {
    fn can_inline(
        &self,
        graph: &Graph,
        node_id: NodeId,
        consumer_id: NodeId,
        _consumer_counts: &HashMap<NodeId, usize>,
    ) -> bool {
        let node = graph.node(node_id);

        // Only inline elementwise operations
        if !node.op.is_elementwise() {
            return false;
        }

        // Must have matching numel for 1:1 element mapping
        if node.numel() != graph.node(consumer_id).numel() {
            return false;
        }

        // Aggressive: can inline multi-consumer nodes if not at boundary
        // This is safe for AOT because we have whole-program visibility
        !self.boundary.contains(&node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::shared::dtype::DType;
    use crate::core::shared::graph::{Node, Op};

    #[test]
    fn solid_policy_allows_multi_consumer() {
        let mut graph = Graph::new();

        // Create a graph with an elementwise node consumed multiple times
        let input = graph.add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        // a = input + input (elementwise, multi-consumer candidate)
        let a = graph.add_node(Node {
            op: Op::Add,
            inputs: vec![input, input],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        // b = a * a
        let b = graph.add_node(Node {
            op: Op::Mul,
            inputs: vec![a, a],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        // c = a + a
        let _c = graph.add_node(Node {
            op: Op::Add,
            inputs: vec![a, a],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        let policy = SolidFusionPolicy::new();
        let mut consumer_counts = HashMap::new();
        consumer_counts.insert(a, 2); // a has 2 consumers (b and c)

        // Solid policy should allow inlining multi-consumer elementwise node
        assert!(policy.can_inline(&graph, a, b, &consumer_counts));
    }

    #[test]
    fn solid_policy_respects_boundaries() {
        let mut graph = Graph::new();

        let a = graph.add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        let b = graph.add_node(Node {
            op: Op::Add,
            inputs: vec![a, a],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        let mut policy = SolidFusionPolicy::new();
        policy.add_boundary(a);

        let consumer_counts = HashMap::new();

        // Should not inline boundary nodes
        assert!(!policy.can_inline(&graph, a, b, &consumer_counts));
    }
}
