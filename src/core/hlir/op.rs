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
    Cos(NodeId),
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
