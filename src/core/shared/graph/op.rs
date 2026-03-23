use crate::core::shared::dtype::Scalar;

#[derive(Clone, Debug)]
pub enum Op {
    // Leaf
    Const(Scalar),
    Load,

    // Binary elementwise
    Add,
    Sub,
    Mul,
    Div,

    // Unary elementwise
    Exp,
    Ln,
    Sqrt,
    Neg,

    // Shape ops (lazy graph nodes)
    Reshape,
    Permute(Vec<usize>),
    Transpose(usize, usize),
    Expand,
    Slice(Vec<(usize, usize)>),
    Flip(Vec<usize>),
    Squeeze,
    Unsqueeze(usize),
    Pad(Scalar, Vec<(usize, usize)>),

    // Reduce ops
    Sum(Vec<usize>, bool),
    Prod(Vec<usize>, bool),
    Max(Vec<usize>, bool),
    Min(Vec<usize>, bool),
}

impl Op {
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Exp | Op::Ln | Op::Sqrt | Op::Neg
        )
    }

    pub fn is_shape_op(&self) -> bool {
        matches!(
            self,
            Op::Reshape
                | Op::Permute(_)
                | Op::Transpose(_, _)
                | Op::Expand
                | Op::Slice(_)
                | Op::Flip(_)
                | Op::Squeeze
                | Op::Unsqueeze(_)
                | Op::Pad(_, _)
        )
    }

    pub fn is_reduce_op(&self) -> bool {
        matches!(
            self,
            Op::Sum(_, _) | Op::Prod(_, _) | Op::Max(_, _) | Op::Min(_, _)
        )
    }
}
