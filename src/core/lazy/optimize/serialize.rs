use super::super::graph::{Graph, NodeId, Op};

/// Convert a graph node to an egglog s-expression string.
pub(super) fn node_to_egglog(graph: &Graph, id: NodeId) -> String {
    let node = graph.node(id);
    match &node.op {
        Op::Load => format!("(tLoad {})", id.0),
        Op::Const(v) => format!("(tConst {:.1})", v.to_f64()),
        Op::Add => format!(
            "(tAdd {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Sub => format!(
            "(tSub {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Mul => format!(
            "(tMul {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Div => format!(
            "(tDiv {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Exp => format!("(tExp {})", node_to_egglog(graph, node.inputs[0])),
        Op::Ln => format!("(tLn {})", node_to_egglog(graph, node.inputs[0])),
        Op::Sqrt => format!("(tSqrt {})", node_to_egglog(graph, node.inputs[0])),
        Op::Neg => format!("(tNeg {})", node_to_egglog(graph, node.inputs[0])),

        // Non-elementwise / shape ops are currently not modeled in egglog.
        // Keep optimizer safe by treating them as opaque leaves.
        Op::Reshape(_)
        | Op::Permute(_)
        | Op::Transpose(_, _)
        | Op::Expand(_)
        | Op::Slice(_)
        | Op::Flip(_)
        | Op::Squeeze
        | Op::Unsqueeze(_)
        | Op::Pad(_, _)
        | Op::Sum(_, _)
        | Op::Prod(_, _)
        | Op::Max(_, _)
        | Op::Min(_, _) => format!("(tLoad {})", id.0),
    }
}
