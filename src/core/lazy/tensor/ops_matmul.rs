use anyhow::{bail, Result};

use crate::core::lazy::tensor::helpers::broadcast_batch;
use crate::core::lazy::tensor::Tensor;

impl Tensor {
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
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

        // Pad batch dims with leading 1s to match broadcast rank, then add
        // the extra dim for the dot-product axis, then expand everything.
        //
        // a: [batch_a..., M, K] -> reshape [1..., batch_a..., M, K, 1]
        //                       -> expand  [batch...,         M, K, N]
        let mut a_rs = vec![1usize; blen - batch_a.len()];
        a_rs.extend_from_slice(batch_a);
        a_rs.extend_from_slice(&[m, k, 1]);
        let mut a_exp: Vec<usize> = batch.clone();
        a_exp.extend_from_slice(&[m, k, n]);
        let lhs = self.reshape(a_rs)?.expand(a_exp)?;

        // b: [batch_b..., K, N] -> reshape [1..., batch_b..., 1, K, N]
        //                       -> expand  [batch...,         M, K, N]
        let mut b_rs = vec![1usize; blen - batch_b.len()];
        b_rs.extend_from_slice(batch_b);
        b_rs.extend_from_slice(&[1, k, n]);
        let mut b_exp: Vec<usize> = batch.clone();
        b_exp.extend_from_slice(&[m, k, n]);
        let rhs = rhs.reshape(b_rs)?.expand(b_exp)?;

        // elementwise mul then sum-reduce over K
        (&lhs * &rhs)?.sum_dims(vec![blen + 1], false)
    }
}
