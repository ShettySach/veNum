//! Elementwise operations for Solid tensors.

use anyhow::Result;

use crate::core::shared::graph::Op;
use crate::core::solid::tensor::Tensor;

impl Tensor {
    /// Element-wise addition.
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(rhs, Op::Add)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(rhs, Op::Sub)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(rhs, Op::Mul)
    }

    /// Element-wise division.
    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(rhs, Op::Div)
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Tensor {
        self.unary_op(Op::Neg)
    }

    /// Element-wise exponential (e^x).
    pub fn exp(&self) -> Tensor {
        self.unary_op(Op::Exp)
    }

    /// Element-wise natural logarithm.
    pub fn ln(&self) -> Tensor {
        self.unary_op(Op::Ln)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Tensor {
        self.unary_op(Op::Sqrt)
    }
}
