//! Elementwise operations for tensors.

use anyhow::{bail, Result};

use crate::core::shared::graph::Op;

use super::context::Context;
use super::structure::Tensor;

impl<C: Context> Tensor<C> {
    // ==================== Binary Operations ====================

    /// Element-wise addition.
    pub fn add(&self, rhs: &Tensor<C>) -> Result<Tensor<C>> {
        self.binary_op(rhs, Op::Add)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, rhs: &Tensor<C>) -> Result<Tensor<C>> {
        self.binary_op(rhs, Op::Sub)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, rhs: &Tensor<C>) -> Result<Tensor<C>> {
        self.binary_op(rhs, Op::Mul)
    }

    /// Element-wise division.
    pub fn div(&self, rhs: &Tensor<C>) -> Result<Tensor<C>> {
        self.binary_op(rhs, Op::Div)
    }

    // ==================== Unary Operations ====================

    /// Element-wise negation.
    pub fn neg(&self) -> Tensor<C> {
        self.unary_op(Op::Neg)
    }

    /// Element-wise exponential (e^x).
    ///
    /// Returns an error if the tensor dtype is not a float type.
    pub fn exp(&self) -> Result<Tensor<C>> {
        if !self.dtype.is_float() {
            bail!("exp requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Exp))
    }

    /// Element-wise natural logarithm.
    ///
    /// Returns an error if the tensor dtype is not a float type.
    pub fn ln(&self) -> Result<Tensor<C>> {
        if !self.dtype.is_float() {
            bail!("ln requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Ln))
    }

    /// Element-wise square root.
    ///
    /// Returns an error if the tensor dtype is not a float type.
    pub fn sqrt(&self) -> Result<Tensor<C>> {
        if !self.dtype.is_float() {
            bail!("sqrt requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Sqrt))
    }
}
