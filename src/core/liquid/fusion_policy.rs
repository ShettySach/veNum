use std::collections::HashMap;

use crate::core::shared::graph::{Graph, NodeId};
use crate::core::shared::schedule::FusionPolicy;

/// Liquid-style fusion policy (Tinygrad-like).
///
/// ## Conservative Fusion Strategy
///
/// This policy only inlines **single-consumer elementwise nodes** with matching numel.
/// This is the safe default for JIT compilation where we lack global graph visibility.
///
/// ### Inlining Criteria
/// 1. Node must be elementwise (`Op::Add`, `Op::Mul`, etc.)
/// 2. Node must have exactly 1 consumer (prevents duplicate computation)
/// 3. Node and consumer must have matching numel (ensures 1:1 element mapping)
///
/// ### Multi-Consumer Handling
/// Multi-consumer nodes are **materialized as kernel inputs** (buffers), ensuring:
/// - No duplicate computation (each value computed once, stored in buffer)
/// - Memory overhead for intermediate buffers
/// - Multiple loads from the same buffer (cache-friendly if accessed locally)
///
/// ### Scheduler-Codegen Contract
/// When a node fails `can_inline()`:
/// - Scheduler adds it to `input_buffers` in `FusedKernel`
/// - Codegen treats it as a materialized buffer load via `input_index`
/// - Expression tree traversal stops at these boundary nodes
///
/// ### Comparison with Solid Policy
/// - **Liquid**: Conservative, materializes multi-consumer nodes (no duplicate compute)
/// - **Solid**: Aggressive, inlines multi-consumer nodes (duplicates compute, no extra buffers)
/// - **Both**: Rely on CSE memoization in codegen to avoid redundant expression building
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
