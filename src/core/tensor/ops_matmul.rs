//! Matrix multiplication for tensors.

use anyhow::{bail, Result};

use super::helpers::broadcast_batch;
use super::structure::Tensor;

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

        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let k2 = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];

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
        let blen = batch.len();

        let mut a_rs = vec![1usize; blen - batch_a.len()];
        a_rs.extend_from_slice(batch_a);
        a_rs.extend_from_slice(&[m, k, 1]);
        let mut a_exp: Vec<usize> = batch.clone();
        a_exp.extend_from_slice(&[m, k, n]);
        let lhs = self.reshape(a_rs)?.expand(a_exp)?;

        let mut b_rs = vec![1usize; blen - batch_b.len()];
        b_rs.extend_from_slice(batch_b);
        b_rs.extend_from_slice(&[1, k, n]);
        let mut b_exp: Vec<usize> = batch.clone();
        b_exp.extend_from_slice(&[m, k, n]);
        let rhs = rhs.reshape(b_rs)?.expand(b_exp)?;

        lhs.mul(&rhs)?.sum(&[(blen + 1) as isize], false)
    }
}
