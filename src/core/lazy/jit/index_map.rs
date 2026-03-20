use super::super::graph::NodeId;

#[inline]
pub(super) fn id_to_index(id: NodeId) -> usize {
    id.0
}

pub(super) fn build_input_index_map(
    graph_nodes_len: usize,
    input_buffers: &[NodeId],
) -> Vec<Option<usize>> {
    let mut map = vec![None; graph_nodes_len];
    for (i, &buf_id) in input_buffers.iter().enumerate() {
        map[id_to_index(buf_id)] = Some(i);
    }
    map
}
