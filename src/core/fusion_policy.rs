//! Fusion policy for Solid mode.

use std::collections::{HashMap, HashSet};

use crate::core::graph::{Graph, NodeId};
use crate::core::schedule::FusionPolicy;

/// Solid-style fusion policy (Luminal-like).
///
/// ## Aggressive Fusion Strategy
///
/// This policy can inline **multi-consumer elementwise nodes** as long as they're not
/// at compilation boundaries. This enables maximum fusion with whole-program visibility.
///
/// ### Inlining Criteria
/// 1. Node must be elementwise (`Op::Add`, `Op::Mul`, etc.)
/// 2. Node and consumer must have matching numel (ensures 1:1 element mapping)
/// 3. Node must NOT be at a boundary (symbolic inputs, outputs, large intermediates)
/// 4. **Consumer count is ignored** - multi-consumer nodes CAN be inlined
///
/// ### Multi-Consumer Handling
/// Multi-consumer nodes are **inlined and duplicated** in each consumer's expression tree:
/// - **Pro**: No intermediate buffer allocation (reduced memory)
/// - **Pro**: Better instruction-level parallelism in generated code
/// - **Con**: Duplicate computation if not optimized by CSE or backend compiler
/// - **Mitigation**: CSE memoization in `build_expression` prevents redundant IR building
///
/// ### Scheduler-Codegen Contract
/// When a node passes `can_inline()`:
/// - Scheduler adds it to `inlined` set, NOT to `input_buffers`
/// - Codegen recursively builds its expression tree for each consumer
/// - CSE memoization ensures shared subexpressions are computed once per kernel
/// - Backend compiler (Cranelift) may further optimize redundant computation
///
/// ### When to Use
/// Use Solid policy when:
/// - Whole-program visibility is available (AOT compilation)
/// - Memory is constrained (minimizing intermediate buffers)
/// - Backend compiler can optimize duplicate computation
/// - Expression trees are small (duplication overhead is low)
pub(crate) struct SolidFusionPolicy {
    /// Nodes at compilation boundary (must materialize)
    boundary: HashSet<NodeId>,
}

impl SolidFusionPolicy {
    /// Create a new Solid fusion policy.
    pub(crate) fn new() -> Self {
        Self {
            boundary: HashSet::new(),
        }
    }

    /// Mark a node as a boundary (must materialize).
    pub(crate) fn add_boundary(&mut self, node_id: NodeId) {
        self.boundary.insert(node_id);
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
    use crate::core::dtype::DType;
    use crate::core::graph::{Node, Op};

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
