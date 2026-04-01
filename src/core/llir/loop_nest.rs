use crate::core::hlir::{DType, ReduceOp, Scalar};

use super::affine::AffineExpr;

#[derive(Clone, Debug, PartialEq)]
pub struct LoopNest {
    pub loops: Vec<Loop>,
    pub body: Vec<crate::core::llir::Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Loop {
    pub var: String,
    pub lower: AffineExpr,
    pub upper: AffineExpr,
    pub step: i64,
    pub kind: LoopKind,
    pub annotations: LoopAnnotations,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LoopKind {
    Sequential,
    Parallel,
    Vectorized {
        width: usize,
    },
    Unrolled {
        factor: usize,
    },
    Reduce {
        accumulators: Vec<ReductionAccumulator>,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoopAnnotations {
    pub comment: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReductionAccumulator {
    pub var: String,
    pub op: ReduceOp,
    pub init: Scalar,
    pub dtype: DType,
}
