use anyhow::{bail, Result};

use super::super::graph::Op;
use super::Tensor;

impl Tensor {
    pub fn exp(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("exp requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Exp))
    }

    pub fn ln(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("ln requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Ln))
    }

    pub fn sqrt(&self) -> Result<Tensor> {
        if !self.dtype.is_float() {
            bail!("sqrt requires float dtype, got {:?}", self.dtype);
        }
        Ok(self.unary_op(Op::Sqrt))
    }

    pub fn neg(&self) -> Tensor {
        self.unary_op(Op::Neg)
    }
}
