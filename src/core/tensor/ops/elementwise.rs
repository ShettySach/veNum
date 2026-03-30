//! Elementwise operations for tensors.

use anyhow::{bail, Result};

use crate::core::graph::Op;

use crate::core::tensor::structure::Tensor;

impl Tensor {
    // ==================== Binary Operations ====================

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

    // ==================== Unary Operations ====================

    /// Element-wise negation.
    pub fn neg(&self) -> Tensor {
        self.unary_op(Op::Neg)
    }

    /// Element-wise exponential (e^x).
    pub fn exp(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("exp requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Exp))
    }

    /// Element-wise natural logarithm.
    pub fn ln(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("ln requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Ln))
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("sqrt requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Sqrt))
    }
}
