use smallvec::{SmallVec, smallvec};

use super::dim::Dim;
use super::types::{BufferId, DType, NodeId, Scalar};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Range {
    pub start: Dim,
    pub end: Dim,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Const {
        value: Scalar,
        shape: Vec<Dim>,
        dtype: DType,
    },
    Load {
        buffer: BufferId,
    },
    Store {
        buffer: BufferId,
        value: NodeId,
    },
    Neg(NodeId),
    Recip(NodeId),
    Exp(NodeId),
    Log(NodeId),
    Sqrt(NodeId),
    Sin(NodeId),
    Cast {
        input: NodeId,
        to: DType,
    },
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Max(NodeId, NodeId),
    Min(NodeId, NodeId),
    Cmp {
        op: CmpOp,
        lhs: NodeId,
        rhs: NodeId,
    },
    Where {
        cond: NodeId,
        then_val: NodeId,
        else_val: NodeId,
    },
    Reduce {
        input: NodeId,
        axes: Vec<usize>,
        op: ReduceOp,
        keepdim: bool,
    },
    Reshape {
        input: NodeId,
        shape: Vec<Dim>,
    },
    Permute {
        input: NodeId,
        axes: Vec<usize>,
    },
    Slice {
        input: NodeId,
        ranges: Vec<Range>,
    },
    Expand {
        input: NodeId,
        shape: Vec<Dim>,
    },
    Concat {
        inputs: Vec<NodeId>,
        axis: usize,
    },
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Op::Const { .. } => "Const",
            Op::Load { .. } => "Load",
            Op::Store { .. } => "Store",
            Op::Neg(_) => "Neg",
            Op::Recip(_) => "Recip",
            Op::Exp(_) => "Exp",
            Op::Log(_) => "Log",
            Op::Sqrt(_) => "Sqrt",
            Op::Sin(_) => "Sin",
            Op::Cast { .. } => "Cast",
            Op::Add(_, _) => "Add",
            Op::Mul(_, _) => "Mul",
            Op::Max(_, _) => "Max",
            Op::Min(_, _) => "Min",
            Op::Cmp { .. } => "Cmp",
            Op::Where { .. } => "Where",
            Op::Reduce { .. } => "Reduce",
            Op::Reshape { .. } => "Reshape",
            Op::Permute { .. } => "Permute",
            Op::Slice { .. } => "Slice",
            Op::Expand { .. } => "Expand",
            Op::Concat { .. } => "Concat",
        }
    }

    pub fn inputs(&self) -> SmallVec<[NodeId; 3]> {
        match self {
            Op::Const { .. } | Op::Load { .. } => smallvec![],
            Op::Store { value, .. } => smallvec![*value],
            Op::Neg(a) | Op::Recip(a) | Op::Exp(a) | Op::Log(a) | Op::Sqrt(a) | Op::Sin(a) => {
                smallvec![*a]
            }
            Op::Cast { input, .. }
            | Op::Reduce { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Slice { input, .. }
            | Op::Expand { input, .. } => smallvec![*input],
            Op::Add(a, b) | Op::Mul(a, b) | Op::Max(a, b) | Op::Min(a, b) => smallvec![*a, *b],
            Op::Cmp { lhs, rhs, .. } => smallvec![*lhs, *rhs],
            Op::Where {
                cond,
                then_val,
                else_val,
            } => smallvec![*cond, *then_val, *else_val],
            Op::Concat { inputs, .. } => inputs.iter().copied().collect(),
        }
    }
}
