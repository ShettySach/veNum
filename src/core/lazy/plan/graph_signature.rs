use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::core::lazy::graph::{Graph, NodeId, Op};

/// Structural identity of the reachable computation graph for a root node.
///
/// Two graphs with the same op topology, dtypes, shapes, and layout details
/// will produce the same `GraphSignature`, allowing their `ExecutionPlan`s to
/// be shared via caching.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphSignature(u64);

impl GraphSignature {
    /// Compute the structural signature by hashing the reachable subgraph.
    pub fn from_graph(graph: &Graph, root: NodeId) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut visited = HashSet::new();
        hash_subgraph(graph, root, &mut hasher, &mut visited);
        GraphSignature(hasher.finish())
    }
}

/// Recursively hash the subgraph rooted at `id`.
fn hash_subgraph(
    graph: &Graph,
    id: NodeId,
    hasher: &mut impl Hasher,
    visited: &mut HashSet<NodeId>,
) {
    // Use a tag to distinguish first-visit from back-ref.
    if !visited.insert(id) {
        // Already visited — hash a back-reference marker + the id index
        // so that DAG structure (shared nodes) is captured.
        0xFFu8.hash(hasher);
        id.0.hash(hasher);
        return;
    }

    let node = graph.node(id);

    // Hash op discriminant.
    std::mem::discriminant(&node.op).hash(hasher);

    // Hash op-specific payload.
    match &node.op {
        Op::Const(v) => v.hash(hasher),
        Op::Permute(axes) => axes.hash(hasher),
        Op::Transpose(d1, d2) => {
            d1.hash(hasher);
            d2.hash(hasher);
        }
        Op::Slice(ranges) => ranges.hash(hasher),
        Op::Flip(dims) => dims.hash(hasher),
        Op::Unsqueeze(rank) => rank.hash(hasher),
        Op::Pad(val, padding) => {
            val.hash(hasher);
            padding.hash(hasher);
        }
        Op::Sum(dims, kd) | Op::Prod(dims, kd) | Op::Max(dims, kd) | Op::Min(dims, kd) => {
            dims.hash(hasher);
            kd.hash(hasher);
        }
        // Load, Reshape, Expand, Squeeze, Add, Sub, Mul, Div, Exp, Ln, Sqrt, Neg
        // — no extra payload beyond discriminant.
        _ => {}
    }

    // Hash dtype and shape.
    node.dtype.hash(hasher);
    node.shape.hash(hasher);

    // Hash number of inputs (structural arity).
    node.inputs.len().hash(hasher);

    // Recurse into inputs.
    for &input_id in &node.inputs {
        hash_subgraph(graph, input_id, hasher, visited);
    }
}
