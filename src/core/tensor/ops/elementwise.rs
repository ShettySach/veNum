//! Elementwise operations for tensors.

use anyhow::{Result, bail};

use crate::core::hlir::{Op, decompose};

use crate::core::tensor::structure::Tensor;

impl Tensor {
    // ==================== Binary Operations ====================

    /// Element-wise addition.
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(rhs, Op::Add(self.id, rhs.id))
    }

    /// Element-wise subtraction.
    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        self.sub_decomposed(rhs)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(rhs, Op::Mul(self.id, rhs.id))
    }

    /// Element-wise division.
    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        self.div_decomposed(rhs)
    }

    // ==================== Unary Operations ====================

    /// Element-wise negation.
    pub fn neg(&self) -> Tensor {
        self.unary_op(Op::Neg(self.id))
    }

    /// Element-wise exponential (e^x).
    pub fn exp(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("exp requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Exp(self.id)))
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("log requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Log(self.id)))
    }

    pub fn ln(&self) -> Result<Tensor> {
        self.log()
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("sqrt requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Sqrt(self.id)))
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("sin requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Sin(self.id)))
    }

    /// Element-wise cosine. Decomposed as `sin(x + π/2)`.
    pub fn cos(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("cos requires float dtype, got {:?}", self.dtype);
        }
        let id = self.with_graph_mut(|g| decompose::cos(g, self.id));
        Ok(self.derived(id, self.shape.clone()))
    }
}
