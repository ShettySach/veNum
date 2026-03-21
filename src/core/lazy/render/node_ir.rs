use std::collections::{HashMap, HashSet};

use super::super::graph::{Graph, NodeId, Op};
use super::super::schedule::{build_schedule, FusedKernel, ReduceKind, ScheduleItem};
use super::labels::{node_label, op_label};

/// Annotate an expression tree with the IR each node would emit.
///
/// Walks the same recursive structure as `build_expression` in the JIT
/// and records a human-readable IR description per node.
fn annotate_expr(
    graph: &Graph,
    id: NodeId,
    kernel: &FusedKernel,
    annotations: &mut HashMap<NodeId, String>,
    visited: &mut HashSet<NodeId>,
) {
    if !visited.insert(id) {
        return;
    }

    let node = graph.node(id);

    match &node.op {
        Op::Load => {
            let access = tracker_access_desc(id, kernel);
            annotations.insert(id, format!("load.{} {}", dtype_tag(graph, id), access));
        }

        Op::Const(v) => {
            annotations.insert(id, format!("{}const {:?}", dtype_tag(graph, id), v));
        }

        op if op.is_shape_op() => {
            let source = kernel
                .shape_source_map
                .get(id.0)
                .and_then(|s| *s);

            // If the source was inlined (elementwise op, not a buffer input),
            // recurse into its expression tree first.
            if let Some(src) = source {
                let src_node = graph.node(src);
                let is_inlined_source = src_node.op.is_elementwise()
                    && !kernel.input_buffers.contains(&src);
                if is_inlined_source {
                    annotate_expr(graph, src, kernel, annotations, visited);
                }
            }

            let source_desc = source
                .map(|src| format!("via N{}", src.0))
                .unwrap_or_else(|| "barrier".into());
            let access = tracker_access_desc(id, kernel);
            annotations.insert(
                id,
                format!("absorbed {} {} {}", op_label(op), source_desc, access),
            );
        }

        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            for &inp in &node.inputs {
                annotate_expr(graph, inp, kernel, annotations, visited);
            }
            let ir_op = match (&node.op, node.dtype.is_float()) {
                (Op::Add, true) => "fadd",
                (Op::Sub, true) => "fsub",
                (Op::Mul, true) => "fmul",
                (Op::Div, true) => "fdiv",
                (Op::Add, false) => "iadd",
                (Op::Sub, false) => "isub",
                (Op::Mul, false) => "imul",
                (Op::Div, false) => "sdiv",
                _ => unreachable!(),
            };
            annotations.insert(id, ir_op.to_string());
        }

        Op::Neg => {
            annotate_expr(graph, node.inputs[0], kernel, annotations, visited);
            let ir_op = if node.dtype.is_float() { "fneg" } else { "ineg" };
            annotations.insert(id, ir_op.to_string());
        }
        Op::Exp => {
            annotate_expr(graph, node.inputs[0], kernel, annotations, visited);
            annotations.insert(id, "call expf".to_string());
        }
        Op::Ln => {
            annotate_expr(graph, node.inputs[0], kernel, annotations, visited);
            annotations.insert(id, "call lnf".to_string());
        }
        Op::Sqrt => {
            annotate_expr(graph, node.inputs[0], kernel, annotations, visited);
            annotations.insert(id, "sqrt".to_string());
        }

        _ => {
            let access = tracker_access_desc(id, kernel);
            annotations.insert(id, format!("load.{} {}", dtype_tag(graph, id), access));
        }
    }
}

fn dtype_tag(graph: &Graph, id: NodeId) -> &'static str {
    match graph.node(id).dtype {
        super::super::dtype::DType::F32 => "f32",
        super::super::dtype::DType::F64 => "f64",
        super::super::dtype::DType::I32 => "i32",
        super::super::dtype::DType::I64 => "i64",
    }
}

fn tracker_access_desc(id: NodeId, kernel: &FusedKernel) -> &'static str {
    let resolved = kernel
        .shape_source_map
        .get(id.0)
        .and_then(|s| *s)
        .unwrap_or(id);

    if let Some(Some(tracker)) = kernel.input_trackers.get(resolved.0) {
        if tracker.is_contiguous() {
            "flat"
        } else {
            "tracked"
        }
    } else {
        "flat"
    }
}

/// Render a Mermaid flowchart showing nodes annotated with their emitted IR.
pub fn render_node_ir(graph: &Graph, root: NodeId) -> String {
    let schedule = build_schedule(graph, root);

    // Pass 1: collect per-kernel node membership and IR annotations.
    let mut node_to_kernel: HashMap<usize, usize> = HashMap::new();
    let mut ir_annotations: HashMap<NodeId, String> = HashMap::new();

    for (si, item) in schedule.iter().enumerate() {
        if let ScheduleItem::Fused(kernel) = item {
            node_to_kernel.insert(kernel.root.0, si);
            collect_fused_nodes(graph, kernel.root, &kernel.input_buffers, &mut node_to_kernel, si);

            let mut visited = HashSet::new();
            annotate_expr(graph, kernel.expr_root, kernel, &mut ir_annotations, &mut visited);

            // For reduce-fused kernels, annotate the reduce root too.
            if let Some(ref reduce) = kernel.reduce {
                let reduce_ir = match reduce.op {
                    ReduceKind::Sum => "reduce sum",
                    ReduceKind::Prod => "reduce prod",
                    ReduceKind::Max => "reduce max",
                    ReduceKind::Min => "reduce min",
                };
                ir_annotations
                    .entry(kernel.root)
                    .or_insert_with(|| reduce_ir.to_string());
            }

            // Annotate input buffers that weren't visited (barrier leaves).
            for &buf_id in &kernel.input_buffers {
                ir_annotations.entry(buf_id).or_insert_with(|| {
                    format!("load.{} flat", dtype_tag(graph, buf_id))
                });
            }
        }
    }

    // Pass 2: build Mermaid output.
    let mut lines = vec!["flowchart BT".to_string()];
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
                    let node_id = NodeId(nid);
                    let label = ir_node_label(graph, node_id, &ir_annotations);
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

    // Render non-scheduled leaf nodes (Loads, Consts) with IR annotations.
    let mut scheduled_nodes: HashSet<usize> = node_to_kernel.keys().copied().collect();
    for item in &schedule {
        match item {
            ScheduleItem::Shape(s) => { scheduled_nodes.insert(s.root.0); }
            ScheduleItem::Reduce(r) => { scheduled_nodes.insert(r.root.0); }
            _ => {}
        }
    }

    let mut visited = HashSet::new();
    render_leaf_nodes(graph, &schedule, &scheduled_nodes, &ir_annotations, &mut visited, &mut lines);

    // Render edges.
    let mut edge_visited = HashSet::new();
    for item in &schedule {
        match item {
            ScheduleItem::Fused(k) => render_edges(graph, k.root, &mut edge_visited, &mut lines),
            ScheduleItem::Shape(s) => render_edges(graph, s.root, &mut edge_visited, &mut lines),
            ScheduleItem::Reduce(r) => render_edges(graph, r.root, &mut edge_visited, &mut lines),
        }
    }

    lines.join("\n")
}

/// Build a node label that includes the graph op and the IR instruction.
fn ir_node_label(graph: &Graph, id: NodeId, annotations: &HashMap<NodeId, String>) -> String {
    let base = node_label(graph, id);

    if let Some(ir) = annotations.get(&id) {
        format!("{}\\n\\u21A7 {}", base, ir)
    } else {
        base
    }
}

fn collect_fused_nodes(
    graph: &Graph,
    id: NodeId,
    input_buffers: &[NodeId],
    node_to_kernel: &mut HashMap<usize, usize>,
    ki: usize,
) {
    let input_set: HashSet<NodeId> = input_buffers.iter().copied().collect();

    fn dfs(
        graph: &Graph,
        id: NodeId,
        input_set: &HashSet<NodeId>,
        node_to_kernel: &mut HashMap<usize, usize>,
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
    schedule: &[ScheduleItem],
    scheduled_nodes: &HashSet<usize>,
    annotations: &HashMap<NodeId, String>,
    visited: &mut HashSet<usize>,
    lines: &mut Vec<String>,
) {
    // Walk from all schedule roots to find unscheduled leaves.
    for item in schedule {
        let root = match item {
            ScheduleItem::Fused(k) => k.root,
            ScheduleItem::Shape(s) => s.root,
            ScheduleItem::Reduce(r) => r.root,
        };
        render_leaf_nodes_dfs(graph, root, scheduled_nodes, annotations, visited, lines);
    }
}

fn render_leaf_nodes_dfs(
    graph: &Graph,
    id: NodeId,
    scheduled_nodes: &HashSet<usize>,
    annotations: &HashMap<NodeId, String>,
    visited: &mut HashSet<usize>,
    lines: &mut Vec<String>,
) {
    if !visited.insert(id.0) {
        return;
    }
    if !scheduled_nodes.contains(&id.0) {
        let label = ir_node_label(graph, id, annotations);
        lines.push(format!("    N{}[\"{}\"]", id.0, label));
    }
    let node = graph.node(id);
    for &input_id in &node.inputs {
        render_leaf_nodes_dfs(graph, input_id, scheduled_nodes, annotations, visited, lines);
    }
}

fn render_edges(
    graph: &Graph,
    id: NodeId,
    visited: &mut HashSet<usize>,
    lines: &mut Vec<String>,
) {
    if !visited.insert(id.0) {
        return;
    }
    let node = graph.node(id);
    for &input_id in &node.inputs {
        render_edges(graph, input_id, visited, lines);
        lines.push(format!("    N{} --> N{}", input_id.0, id.0));
    }
}
