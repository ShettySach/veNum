use crate::core::hlir::{DType, ReduceOp, Scalar};

use super::loop_nest::Loop;
use super::memory::MemoryAccess;

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Assign {
        dst: MemoryAccess,
        src: Expr,
    },
    Accumulate {
        dst: MemoryAccess,
        op: ReduceOp,
        src: Expr,
    },
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    Loop(Loop, Vec<Stmt>),
    Barrier,
    Epilogue {
        main_loop_var: String,
        remainder_body: Vec<Stmt>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Literal(Scalar),
    Load(MemoryAccess),
    Unary {
        op: UnaryOp,
        arg: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Ternary {
        cond: Box<Expr>,
        then_val: Box<Expr>,
        else_val: Box<Expr>,
    },
    Cast {
        arg: Box<Expr>,
        to: DType,
    },
    AbstractVector(AbstractVectorOp),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Recip,
    Exp,
    Log,
    Sqrt,
    Sin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Mul,
    Max,
    Min,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AbstractVectorOp {
    Fma {
        acc: Box<Expr>,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        width: usize,
    },
    HorizontalReduce {
        op: ReduceOp,
        arg: Box<Expr>,
        width: usize,
    },
    Broadcast {
        scalar: Box<Expr>,
        width: usize,
    },
    Gather {
        base: crate::core::hlir::BufferId,
        indices: Box<Expr>,
        width: usize,
    },
    Scatter {
        base: crate::core::hlir::BufferId,
        indices: Box<Expr>,
        value: Box<Expr>,
        width: usize,
    },
    VecBinary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        width: usize,
    },
    VecCast {
        arg: Box<Expr>,
        from: DType,
        to: DType,
        width: usize,
    },
}
