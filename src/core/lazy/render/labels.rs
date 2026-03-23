use crate::core::lazy::dtype::Buffer;
use crate::core::lazy::graph::{Graph, NodeId, Op};

pub(super) fn op_label(op: &Op) -> &'static str {
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
        Op::Reshape => "Reshape",
        Op::Permute(_) => "Permute",
        Op::Transpose(_, _) => "Transpose",
        Op::Expand => "Expand",
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

pub(super) fn node_label(graph: &Graph, id: NodeId) -> String {
    let node = graph.node(id);
    let shape_str = format!("{:?}", node.shape);
    match &node.op {
        Op::Const(v) => format!("Const({v:?})\\n{shape_str}"),
        Op::Load => {
            let preview = node.buffer.as_ref().map(buffer_preview).unwrap_or_default();
            format!("Load {:?}\\n{preview}\\n{shape_str}", node.dtype)
        }
        other => format!("{}\\n{shape_str}", op_label(other)),
    }
}

fn buffer_preview(buffer: &Buffer) -> String {
    match buffer {
        Buffer::F32(v) => fmt_preview_slice(v),
        Buffer::F64(v) => fmt_preview_slice(v),
        Buffer::I32(v) => fmt_preview_slice(v),
        Buffer::I64(v) => fmt_preview_slice(v),
    }
}

fn fmt_preview_slice<T: std::fmt::Debug>(slice: &[T]) -> String {
    let len = slice.len();
    if len <= 4 {
        format!("{:?}", slice)
    } else {
        format!("[{:?}, {:?}, ... {:?}]", slice[0], slice[1], slice[len - 1])
    }
}
