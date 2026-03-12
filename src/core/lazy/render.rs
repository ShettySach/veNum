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
    }
}

fn node_label(graph: &Graph, id: NodeId) -> String {
    let node = graph.node(id);
    let shape_str = format!("{:?}", node.shape);
    match &node.op {
        Op::Const(v) => format!("Const({v})\\n{shape_str}"),
        Op::Load => {
            let preview = node
                .buffer
                .as_ref()
                .map(|b| {
                    let data = b.as_f32();
                    if data.len() <= 4 {
                        format!("{:?}", data)
                    } else {
                        format!("[{}, {}, ... {}]", data[0], data[1], data[data.len() - 1])
                    }
                })
                .unwrap_or_default();
            format!("Load\\n{preview}\\n{shape_str}")
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

/// Render the DAG after fusion, with fused kernels grouped in subgraphs.
pub fn render_fused_dag(graph: &Graph, root: NodeId) -> String {
    let schedule = build_schedule(graph, root);

    // Collect which nodes belong to which kernel.
    let mut node_to_kernel: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for (ki, item) in schedule.iter().enumerate() {
        match item {
            ScheduleItem::Fused(kernel) => {
                // The root is in this kernel.
                node_to_kernel.insert(kernel.root.0, ki);
                // Walk the fused tree to find all inlined nodes.
                collect_fused_nodes(
                    graph,
                    kernel.root,
                    &kernel.input_buffers,
                    &mut node_to_kernel,
                    ki,
                );
            }
        }
    }

    let mut lines = vec!["flowchart BT".to_string()];

    // Render kernel subgraphs (skip empty ones).
    let mut kernel_idx = 0;
    for (ki, item) in schedule.iter().enumerate() {
        match item {
            ScheduleItem::Fused(_) => {
                let members: Vec<usize> = node_to_kernel
                    .iter()
                    .filter(|(_, &k)| k == ki)
                    .map(|(&n, _)| n)
                    .collect();
                if members.is_empty() {
                    continue;
                }
                lines.push(format!(
                    "    subgraph Kernel_{} [\"Fused Kernel {}\"]",
                    kernel_idx, kernel_idx
                ));
                kernel_idx += 1;
                for &nid in &members {
                    let label = node_label(graph, NodeId(nid));
                    lines.push(format!("        N{}[\"{}\"]", nid, label));
                }
                lines.push("    end".to_string());
            }
        }
    }

    // Render non-kernel nodes (leaves).
    let mut visited = HashSet::new();
    render_leaf_nodes(graph, root, &node_to_kernel, &mut visited, &mut lines);

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
    let node = graph.node(id);
    for &input_id in &node.inputs {
        if input_buffers.contains(&input_id) {
            continue;
        }
        // This node is inlined into the kernel.
        node_to_kernel.insert(input_id.0, ki);
        collect_fused_nodes(graph, input_id, input_buffers, node_to_kernel, ki);
    }
}

fn render_leaf_nodes(
    graph: &Graph,
    id: NodeId,
    node_to_kernel: &std::collections::HashMap<usize, usize>,
    visited: &mut HashSet<usize>,
    lines: &mut Vec<String>,
) {
    if !visited.insert(id.0) {
        return;
    }
    if !node_to_kernel.contains_key(&id.0) {
        let label = node_label(graph, id);
        lines.push(format!("    N{}[\"{}\"]", id.0, label));
    }
    let node = graph.node(id);
    for &input_id in &node.inputs {
        render_leaf_nodes(graph, input_id, node_to_kernel, visited, lines);
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
