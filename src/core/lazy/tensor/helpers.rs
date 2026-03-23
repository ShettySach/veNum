use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};

use crate::core::shared::graph::{Graph, Node, NodeId, Op};

pub(super) fn clone_reachable_subgraph(src: &Graph, root: NodeId) -> (Graph, NodeId) {
    let (dst, new_root, _) = clone_reachable_subgraph_with_map(src, root);
    (dst, new_root)
}

pub(super) fn clone_reachable_subgraph_with_map(
    src: &Graph,
    root: NodeId,
) -> (Graph, NodeId, HashMap<NodeId, NodeId>) {
    let mut dst = Graph::new();
    let mut id_map = HashMap::new();
    let new_root = import_node(src, root, &mut dst, &mut id_map);
    (dst, new_root, id_map)
}

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

pub(super) fn broadcast_batch(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        if da != db && da != 1 && db != 1 {
            bail!("batch dimensions not broadcastable: {:?} vs {:?}", a, b);
        }
        out[i] = da.max(db);
    }
    Ok(out)
}

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
