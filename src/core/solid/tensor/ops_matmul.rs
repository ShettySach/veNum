//! Matrix multiplication for Solid tensors.

use anyhow::{bail, Result};

use crate::core::solid::tensor::Tensor;

impl Tensor {
    /// Matrix multiplication.
    ///
    /// For 2D tensors: standard matrix multiply
    /// For 3D+ tensors: batched matrix multiply
    ///
    /// Implemented as reshape + expand + multiply + sum (same as Liquid mode).
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.dtype != rhs.dtype {
            bail!("matmul requires matching dtypes");
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
        lhs.mul(&rhs)?.sum(&[(blen + 1) as isize], false)
    }
}

// Helper function from lazy/tensor/helpers.rs
fn broadcast_batch(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_idx = if i < a.len() {
            a.len() - 1 - i
        } else {
            usize::MAX
        };
        let b_idx = if i < b.len() {
            b.len() - 1 - i
        } else {
            usize::MAX
        };

        let a_val = if a_idx < a.len() { a[a_idx] } else { 1 };
        let b_val = if b_idx < b.len() { b[b_idx] } else { 1 };

        if a_val != b_val && a_val != 1 && b_val != 1 {
            bail!("incompatible batch dimensions: {:?} vs {:?}", a, b);
        }

        result.push(a_val.max(b_val));
    }

    result.reverse();
    Ok(result)
}
