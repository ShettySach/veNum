//! Helper functions for liquid tensor operations.

use std::collections::{HashMap, HashSet};

use crate::core::shared::graph::{Graph, Node, NodeId, Op};

/// Clone a reachable subgraph starting from a root node.
pub(super) fn clone_reachable_subgraph(src: &Graph, root: NodeId) -> (Graph, NodeId) {
    let (dst, new_root, _) = clone_reachable_subgraph_with_map(src, root);
    (dst, new_root)
}

/// Clone a reachable subgraph and return the node ID mapping.
pub(super) fn clone_reachable_subgraph_with_map(
    src: &Graph,
    root: NodeId,
) -> (Graph, NodeId, HashMap<NodeId, NodeId>) {
    let mut dst = Graph::new();
    let mut id_map = HashMap::new();

    fn import_node(
        src_graph: &Graph,
        src_id: NodeId,
        dst_graph: &mut Graph,
        id_map: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if let Some(&mapped) = id_map.get(&src_id) {
            return mapped;
        }

        let node = src_graph.node(src_id);
        let new_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|&input_id| import_node(src_graph, input_id, dst_graph, id_map))
            .collect();

        let new_id = dst_graph.add_node(Node {
            op: node.op.clone(),
            inputs: new_inputs,
            shape: node.shape.clone(),
            dtype: node.dtype,
            buffer: node.buffer.clone(),
        });

        id_map.insert(src_id, new_id);
        new_id
    }

    let new_root = import_node(src, root, &mut dst, &mut id_map);
    (dst, new_root, id_map)
}

/// Check if it's safe to optimize the subgraph rooted at the given node.
///
/// Returns true only if:
/// - The root node is a float dtype (egglog rules use float constants)
/// - All reachable nodes use operations that are safe for egglog optimization
pub(super) fn is_optimize_safe(graph: &Graph, root: NodeId) -> bool {
    // Only optimize float dtypes - egglog rules use float constants.
    if !graph.node(root).dtype.is_float() {
        return false;
    }

    fn dfs(graph: &Graph, id: NodeId, seen: &mut HashSet<NodeId>) -> bool {
        if !seen.insert(id) {
            return true;
        }

        let node = graph.node(id);
        match node.op {
            Op::Load
            | Op::Const(_)
            | Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Exp
            | Op::Ln
            | Op::Sqrt
            | Op::Neg
            | Op::Reshape
            | Op::Permute(_)
            | Op::Transpose(_, _)
            | Op::Expand
            | Op::Squeeze
            | Op::Unsqueeze(_) => node.inputs.iter().all(|&inp| dfs(graph, inp, seen)),
            _ => false,
        }
    }

    let mut seen = HashSet::new();
    dfs(graph, root, &mut seen)
}
