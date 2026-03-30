//! Convolution operations for tensors.

use anyhow::{Result, bail};

use super::structure::Tensor;

impl Tensor {
    /// 2D convolution (cross-correlation) as used in CNNs.
    ///
    /// - `self` (input):  `[N, C_in, H, W]`
    /// - `weight`:        `[C_out, C_in, kH, kW]`
    /// - Output:          `[N, C_out, oH, oW]`
    pub fn conv2d(&self, weight: &Tensor) -> Result<Tensor> {
        if self.dtype != weight.dtype {
            bail!(
                "conv2d requires matching dtypes: {:?} vs {:?}",
                self.dtype,
                weight.dtype
            );
        }
        if self.shape.len() != 4 {
            bail!(
                "conv2d: input must be 4D [N, C_in, H, W], got {:?}",
                self.shape
            );
        }
        if weight.shape.len() != 4 {
            bail!(
                "conv2d: weight must be 4D [C_out, C_in, kH, kW], got {:?}",
                weight.shape
            );
        }

        let n = self.shape[0];
        let c_in = self.shape[1];
        let h = self.shape[2];
        let w = self.shape[3];

        let c_out = weight.shape[0];
        let wc_in = weight.shape[1];
        let kh = weight.shape[2];
        let kw = weight.shape[3];

        if c_in != wc_in {
            bail!(
                "conv2d: input channels {} != weight channels {}",
                c_in,
                wc_in
            );
        }
        if kh > h || kw > w {
            bail!(
                "conv2d: kernel [{}, {}] larger than input [{}, {}]",
                kh,
                kw,
                h,
                w
            );
        }

        let oh = h - kh + 1;
        let ow = w - kw + 1;

        let mut acc: Option<Tensor> = None;

        for ki in 0..kh {
            for kj in 0..kw {
                let patch = self.slice(vec![(0, n), (0, c_in), (ki, ki + oh), (kj, kj + ow)])?;
                let wslice =
                    weight.slice(vec![(0, c_out), (0, c_in), (ki, ki + 1), (kj, kj + 1)])?;

                let patch = patch.reshape(vec![n, 1, c_in, oh, ow])?;
                let wslice = wslice.reshape(vec![1, c_out, c_in, 1, 1])?;

                let product = patch.mul(&wslice)?;
                let summed = product.sum(&[2], false)?;

                acc = Some(match acc {
                    Some(prev) => prev.add(&summed)?,
                    None => summed,
                });
            }
        }

        acc.ok_or_else(|| anyhow::anyhow!("conv2d: empty kernel"))
    }
}
