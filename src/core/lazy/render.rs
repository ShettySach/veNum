use std::collections::HashSet;

use super::graph::{Graph, NodeId, Op};
use super::schedule::{build_schedule, ScheduleItem};

fn op_label(op: &Op) -> &'static str {
    match op {
        Op::Const(_) => "Const",
        Op::Load => "Load",
        Op::Add => "Add",
        Op::Sub => "Sub",
        Op::Mul => "Mul",
        Op::Div => "Div",
        Op::Exp => "Exp",
        Op::Ln => "Ln",
        Op::Sqrt => "Sqrt",
        Op::Neg => "Neg",
        Op::Reshape(_) => "Reshape",
        Op::Permute(_) => "Permute",
        Op::Transpose(_, _) => "Transpose",
        Op::Expand(_) => "Expand",
        Op::Slice(_) => "Slice",
        Op::Flip(_) => "Flip",
        Op::Squeeze => "Squeeze",
        Op::Unsqueeze(_) => "Unsqueeze",
        Op::Pad(_, _) => "Pad",
        Op::Sum(_, _) => "Sum",
        Op::Prod(_, _) => "Prod",
        Op::Max(_, _) => "Max",
        Op::Min(_, _) => "Min",
    }
}

fn node_label(graph: &Graph, id: NodeId) -> String {
    let node = graph.node(id);
    let shape_str = format!("{:?}", node.shape);
    match &node.op {
        Op::Const(v) => format!("Const({v:?})\\n{shape_str}"),
        Op::Load => {
            let preview = node
                .buffer
                .as_ref()
                .map(|b| {
                    let len = b.len();
                    match b {
                        super::dtype::Buffer::F32(v) => {
                            if len <= 4 {
                                format!("{:?}", &**v)
                            } else {
                                format!("[{}, {}, ... {}]", v[0], v[1], v[len - 1])
                            }
                        }
                        super::dtype::Buffer::F64(v) => {
                            if len <= 4 {
                                format!("{:?}", &**v)
                            } else {
                                format!("[{}, {}, ... {}]", v[0], v[1], v[len - 1])
                            }
                        }
                        super::dtype::Buffer::I32(v) => {
                            if len <= 4 {
                                format!("{:?}", &**v)
                            } else {
                                format!("[{}, {}, ... {}]", v[0], v[1], v[len - 1])
                            }
                        }
                        super::dtype::Buffer::I64(v) => {
                            if len <= 4 {
                                format!("{:?}", &**v)
                            } else {
                                format!("[{}, {}, ... {}]", v[0], v[1], v[len - 1])
                            }
                        }
                    }
                })
                .unwrap_or_default();
            format!("Load {:?}\\n{preview}\\n{shape_str}", node.dtype)
        }
        other => format!("{}\\n{shape_str}", op_label(other)),
    }
}

/// Render the raw DAG (before fusion) rooted at `root` as Mermaid flowchart code.
pub fn render_dag(graph: &Graph, root: NodeId) -> String {
    let mut visited = HashSet::new();
    let mut lines = vec!["flowchart BT".to_string()];
    render_dag_node(graph, root, &mut visited, &mut lines);
    lines.join("\n")
}

fn render_dag_node(
    graph: &Graph,
    id: NodeId,
    visited: &mut HashSet<usize>,
    lines: &mut Vec<String>,
) {
    if !visited.insert(id.0) {
        return;
    }
    let node = graph.node(id);
    let label = node_label(graph, id);
    lines.push(format!("    N{}[\"{}\"]", id.0, label));

    for &input_id in &node.inputs {
        render_dag_node(graph, input_id, visited, lines);
        lines.push(format!("    N{} --> N{}", input_id.0, id.0));
    }
}

/// Render the DAG after scheduling, with fused kernels and shape-op barriers grouped.
pub fn render_fused_dag(graph: &Graph, root: NodeId) -> String {
    let schedule = build_schedule(graph, root);

    // Collect which nodes belong to which fused kernel.
    let mut node_to_kernel: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    // Collect shape/reduce barrier roots to render explicitly.
    let mut shape_barriers: Vec<(usize, usize)> = Vec::new(); // (node_id, schedule_index)
    let mut reduce_barriers: Vec<(usize, usize)> = Vec::new(); // (node_id, schedule_index)

    for (si, item) in schedule.iter().enumerate() {
        match item {
            ScheduleItem::Fused(kernel) => {
                node_to_kernel.insert(kernel.root.0, si);
                collect_fused_nodes(
                    graph,
                    kernel.root,
                    &kernel.input_buffers,
                    &mut node_to_kernel,
                    si,
                );
            }
            ScheduleItem::Shape(shape_item) => {
                shape_barriers.push((shape_item.root.0, si));
            }
            ScheduleItem::Reduce(reduce_item) => {
                reduce_barriers.push((reduce_item.root.0, si));
            }
        }
    }

    let mut lines = vec!["flowchart BT".to_string()];

    // Render scheduled subgraphs in schedule order.
    let mut fused_idx = 0usize;
    let mut shape_idx = 0usize;
    let mut reduce_idx = 0usize;

    for (si, item) in schedule.iter().enumerate() {
        match item {
            ScheduleItem::Fused(_) => {
                let members: Vec<usize> = node_to_kernel
                    .iter()
                    .filter(|(_, &k)| k == si)
                    .map(|(&n, _)| n)
                    .collect();

                if members.is_empty() {
                    continue;
                }

                lines.push(format!(
                    "    subgraph Kernel_{} [\"Fused Kernel {}\"]",
                    fused_idx, fused_idx
                ));
                fused_idx += 1;

                for &nid in &members {
                    let label = node_label(graph, NodeId(nid));
                    lines.push(format!("        N{}[\"{}\"]", nid, label));
                }
                lines.push("    end".to_string());
            }
            ScheduleItem::Shape(shape_item) => {
                lines.push(format!(
                    "    subgraph Shape_{} [\"Shape Op {}\"]",
                    shape_idx,
                    op_label(&shape_item.op)
                ));
                shape_idx += 1;

                let nid = shape_item.root.0;
                let label = node_label(graph, NodeId(nid));
                lines.push(format!("        N{}[\"{}\"]", nid, label));
                lines.push("    end".to_string());
            }
            ScheduleItem::Reduce(reduce_item) => {
                lines.push(format!(
                    "    subgraph Reduce_{} [\"Reduce Op {}\"]",
                    reduce_idx,
                    op_label(&reduce_item.op)
                ));
                reduce_idx += 1;

                let nid = reduce_item.root.0;
                let label = node_label(graph, NodeId(nid));
                lines.push(format!("        N{}[\"{}\"]", nid, label));
                lines.push("    end".to_string());
            }
        }
    }

    // Render non-scheduled leaf nodes.
    let mut scheduled_nodes: HashSet<usize> = node_to_kernel.keys().copied().collect();
    for (nid, _) in &shape_barriers {
        scheduled_nodes.insert(*nid);
    }
    for (nid, _) in &reduce_barriers {
        scheduled_nodes.insert(*nid);
    }

    let mut visited = HashSet::new();
    render_leaf_nodes(graph, root, &scheduled_nodes, &mut visited, &mut lines);

    // Render edges.
    let mut edge_visited = HashSet::new();
    render_edges(graph, root, &mut edge_visited, &mut lines);

    lines.join("\n")
}

fn collect_fused_nodes(
    graph: &Graph,
    id: NodeId,
    input_buffers: &[NodeId],
    node_to_kernel: &mut std::collections::HashMap<usize, usize>,
    ki: usize,
) {
    use std::collections::HashSet;

    let input_set: HashSet<NodeId> = input_buffers.iter().copied().collect();

    fn dfs(
        graph: &Graph,
        id: NodeId,
        input_set: &std::collections::HashSet<NodeId>,
        node_to_kernel: &mut std::collections::HashMap<usize, usize>,
        ki: usize,
    ) {
        let node = graph.node(id);
        for &input_id in &node.inputs {
            if input_set.contains(&input_id) {
                continue;
            }
            node_to_kernel.insert(input_id.0, ki);
            dfs(graph, input_id, input_set, node_to_kernel, ki);
        }
    }

    dfs(graph, id, &input_set, node_to_kernel, ki);
}

fn render_leaf_nodes(
    graph: &Graph,
    id: NodeId,
    scheduled_nodes: &HashSet<usize>,
    visited: &mut HashSet<usize>,
    lines: &mut Vec<String>,
) {
    if !visited.insert(id.0) {
        return;
    }
    if !scheduled_nodes.contains(&id.0) {
        let label = node_label(graph, id);
        lines.push(format!("    N{}[\"{}\"]", id.0, label));
    }
    let node = graph.node(id);
    for &input_id in &node.inputs {
        render_leaf_nodes(graph, input_id, scheduled_nodes, visited, lines);
    }
}

fn render_edges(graph: &Graph, id: NodeId, visited: &mut HashSet<usize>, lines: &mut Vec<String>) {
    if !visited.insert(id.0) {
        return;
    }
    let node = graph.node(id);
    for &input_id in &node.inputs {
        render_edges(graph, input_id, visited, lines);
        lines.push(format!("    N{} --> N{}", input_id.0, id.0));
    }
}
