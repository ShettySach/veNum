//! Matrix multiplication for tensors.

use anyhow::{anyhow, bail, Result};

use crate::core::hlir::{decompose, Dim};
use crate::core::tensor::helpers::broadcast_batch;
use crate::core::tensor::structure::Tensor;

impl Tensor {
    /// Matrix multiplication.
    ///
    /// For 2D tensors: standard matrix multiply `[M, K] @ [K, N] -> [M, N]`
    /// For 3D+ tensors: batched matrix multiply with broadcasting
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.dtype != rhs.dtype {
            bail!(
                "matmul requires matching dtypes: {:?} vs {:?}",
                self.dtype,
                rhs.dtype
            );
        }

        let a_shape = &self.shape;
        let b_shape = &rhs.shape;

        if a_shape.len() < 2 || b_shape.len() < 2 {
            bail!(
                "matmul requires at least 2D tensors, got {:?} and {:?}",
                a_shape,
                b_shape
            );
        }

        let m = a_shape[a_shape.len() - 2]
            .as_const()
            .ok_or_else(|| anyhow!("matmul currently requires constant dimensions"))?;
        let k = a_shape[a_shape.len() - 1]
            .as_const()
            .ok_or_else(|| anyhow!("matmul currently requires constant dimensions"))?;
        let k2 = b_shape[b_shape.len() - 2]
            .as_const()
            .ok_or_else(|| anyhow!("matmul currently requires constant dimensions"))?;
        let n = b_shape[b_shape.len() - 1]
            .as_const()
            .ok_or_else(|| anyhow!("matmul currently requires constant dimensions"))?;

        if k != k2 {
            bail!(
                "matmul inner dimensions mismatch: {:?} vs {:?}",
                a_shape,
                b_shape
            );
        }

        let batch_a = &a_shape[..a_shape.len() - 2];
        let batch_b = &b_shape[..b_shape.len() - 2];
        let batch = broadcast_batch(batch_a, batch_b)?;
        let id = self.with_graph_mut(|g| decompose::matmul(g, self.id, rhs.id));
        let mut out_shape = batch;
        out_shape.push(Dim::constant(m));
        out_shape.push(Dim::constant(n));
        Ok(self.derived(id, out_shape))
    }
}
