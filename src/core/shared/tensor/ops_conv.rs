//! Convolution operations for tensors.

use anyhow::{Result, bail};

use super::context::Context;
use super::structure::Tensor;

impl<C: Context> Tensor<C> {
    /// 2D convolution (cross-correlation) as used in CNNs.
    ///
    /// - `self` (input):  `[N, C_in, H, W]`
    /// - `weight`:        `[C_out, C_in, kH, kW]`
    /// - Output:          `[N, C_out, oH, oW]`
    ///
    /// where `oH = H - kH + 1`, `oW = W - kW + 1`.
    ///
    /// This is the standard valid-mode convolution with stride 1,
    /// used in CNNs (e.g. MNIST, ResNet). For other padding modes,
    /// call `pad()` on the input before `conv2d()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // MNIST-style: 1 input channel, 32 output channels, 3x3 kernel
    /// let input  = Tensor::from_slice(&cx, &data, vec![1, 1, 28, 28]);
    /// let weight = Tensor::from_slice(&cx, &w,    vec![32, 1, 3, 3]);
    /// let output = input.conv2d(&weight)?; // [1, 32, 26, 26]
    /// ```
    pub fn conv2d(&self, weight: &Tensor<C>) -> Result<Tensor<C>> {
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

        // Accumulate over kernel positions.
        //
        // For each (ki, kj):
        //   patch  = input[:, :, ki..ki+oH, kj..kj+oW]  -> [N, C_in, oH, oW]
        //   wslice = weight[:, :, ki, kj]                -> [C_out, C_in]
        //
        // Reshape to align for broadcast multiply:
        //   patch  -> [N, 1, C_in, oH, oW]
        //   wslice -> [1, C_out, C_in, 1, 1]
        //
        // Multiply -> [N, C_out, C_in, oH, oW]
        // Sum over C_in (dim 2) -> [N, C_out, oH, oW]

        let mut acc: Option<Tensor<C>> = None;

        for ki in 0..kh {
            for kj in 0..kw {
                // Slice input patch
                let patch = self.slice(vec![(0, n), (0, c_in), (ki, ki + oh), (kj, kj + ow)])?;

                // Slice single kernel position
                let wslice =
                    weight.slice(vec![(0, c_out), (0, c_in), (ki, ki + 1), (kj, kj + 1)])?;

                // Reshape for broadcasting: mul handles expand automatically
                let patch = patch.reshape(vec![n, 1, c_in, oh, ow])?;
                let wslice = wslice.reshape(vec![1, c_out, c_in, 1, 1])?;

                let product = patch.mul(&wslice)?;
                let summed = product.sum(&[2], false)?; // [N, C_out, oH, oW]

                acc = Some(match acc {
                    Some(prev) => prev.add(&summed)?,
                    None => summed,
                });
            }
        }

        acc.ok_or_else(|| anyhow::anyhow!("conv2d: empty kernel"))
    }
}
