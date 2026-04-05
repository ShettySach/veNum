use std::fmt::Write;

use crate::core::hlir::{BufferId, HLIRGraph, NodeId, Op};

pub fn to_mermaid(graph: &HLIRGraph) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "graph TD");

    for (id, node) in graph.topo_iter() {
        let label = node_label(id, &node.op);
        let _ = writeln!(out, "  n{}[\"{}\"]", id.0, label);
    }

    for (id, node) in graph.topo_iter() {
        for input in node.op.inputs() {
            let _ = writeln!(out, "  n{} --> n{}", input.0, id.0);
        }
    }

    out
}

fn node_label(id: NodeId, op: &Op) -> String {
    match op {
        Op::Load { buffer } => format!("{}#{}\\nbuffer={}", op.name(), id.0, fmt_buffer(*buffer)),
        Op::Store { buffer, .. } => {
            format!("{}#{}\\nbuffer={}", op.name(), id.0, fmt_buffer(*buffer))
        }
        Op::Reduce {
            axes,
            op: reduce,
            keepdim,
            ..
        } => format!(
            "{}#{}\\n{:?} axes={:?} keepdim={}",
            op.name(),
            id.0,
            reduce,
            axes,
            keepdim
        ),
        Op::Cast { to, .. } => format!("{}#{}\\n{:?}", op.name(), id.0, to),
        Op::Cmp { op: cmp, .. } => format!("{}#{}\\n{:?}", op.name(), id.0, cmp),
        Op::Permute { axes, .. } => format!("{}#{}\\naxes={:?}", op.name(), id.0, axes),
        Op::Slice { ranges, .. } => format!("{}#{}\\nranges={}", op.name(), id.0, ranges.len()),
        Op::Concat { axis, .. } => format!("{}#{}\\naxis={}", op.name(), id.0, axis),
        _ => format!("{}#{}", op.name(), id.0),
    }
}

fn fmt_buffer(id: BufferId) -> usize {
    id.0
}
