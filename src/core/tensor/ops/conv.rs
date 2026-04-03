//! Convolution operations for tensors.

use crate::core::tensor::structure::Tensor;
use anyhow::{anyhow, bail, Result};

impl Tensor {
    /// 2D convolution (cross-correlation) as used in CNNs.
    ///
    /// - `self` (input):  `[batch_size, channels_in, input_height, input_width]`
    /// - `weight`:        `[channels_out, channels_in, kernel_height, kernel_width]`
    /// - Output:          `[batch_size, channels_out, output_height, output_width]`
    ///
    /// Uses the tinygrad-style `_pool` trick: instead of iterating over each kernel
    /// position, we use shape manipulation (expand with zero strides, permute, slice)
    /// to create all sliding windows in a single graph fragment.
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
                "conv2d: input must be 4D [batch_size, channels_in, input_height, input_width], got {:?}",
                self.shape
            );
        }
        if weight.shape.len() != 4 {
            bail!(
                "conv2d: weight must be 4D [channels_out, channels_in, kernel_height, kernel_width], got {:?}",
                weight.shape
            );
        }

        let batch_size = self.shape[0]
            .as_const()
            .ok_or_else(|| anyhow!("conv2d requires constant batch dimension"))?;
        let channels_in = self.shape[1]
            .as_const()
            .ok_or_else(|| anyhow!("conv2d requires constant channels_in"))?;
        let inp_height = self.shape[2]
            .as_const()
            .ok_or_else(|| anyhow!("conv2d requires constant input height"))?;
        let inp_width = self.shape[3]
            .as_const()
            .ok_or_else(|| anyhow!("conv2d requires constant input width"))?;

        let channels_out = weight.shape[0]
            .as_const()
            .ok_or_else(|| anyhow!("conv2d requires constant channels_out"))?;
        let kernel_channels_in = weight.shape[1]
            .as_const()
            .ok_or_else(|| anyhow!("conv2d requires constant kernel channels_in"))?;
        let kernel_height = weight.shape[2]
            .as_const()
            .ok_or_else(|| anyhow!("conv2d requires constant kernel height"))?;
        let kernel_width = weight.shape[3]
            .as_const()
            .ok_or_else(|| anyhow!("conv2d requires constant kernel width"))?;

        if channels_in != kernel_channels_in {
            bail!(
                "conv2d: input channels {} != weight channels {}",
                channels_in,
                kernel_channels_in
            );
        }

        if kernel_height > inp_height || kernel_width > inp_width {
            bail!(
                "conv2d: kernel [{}, {}] larger than input [{}, {}]",
                kernel_height,
                kernel_width,
                inp_height,
                inp_width
            );
        }

        let output_height = inp_height - kernel_height + 1;
        let output_width = inp_width - kernel_width + 1;

        // Use _pool to create sliding windows
        // Input: [batch, channels_in, height, width]
        // Pooled: [batch, channels_in, out_height, out_width, kernel_height, kernel_width]
        let pooled = self._pool2d(kernel_height, kernel_width, 1, 1)?;

        // Reshape weight for broadcasting:
        // [channels_out, channels_in, kernel_height, kernel_width]
        // -> [1, channels_out, channels_in, 1, 1, kernel_height, kernel_width]
        let weight_reshaped = weight.reshape(vec![
            1,
            channels_out,
            channels_in,
            1,
            1,
            kernel_height,
            kernel_width,
        ])?;

        // Reshape pooled for broadcasting:
        // [batch, channels_in, out_height, out_width, kernel_height, kernel_width]
        // -> [batch, 1, channels_in, out_height, out_width, kernel_height, kernel_width]
        let pooled_reshaped = pooled.reshape(vec![
            batch_size,
            1,
            channels_in,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
        ])?;

        // Expand both for multiplication
        let pooled_expanded = pooled_reshaped.expand(vec![
            batch_size,
            channels_out,
            channels_in,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
        ])?;
        let weight_expanded = weight_reshaped.expand(vec![
            batch_size,
            channels_out,
            channels_in,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
        ])?;

        // Multiply and reduce over channels_in, kernel_height, kernel_width (axes 2, 5, 6)
        let result = pooled_expanded
            .mul(&weight_expanded)?
            .sum(&[2, 5, 6], false)?;

        // Result shape: [batch, channels_out, out_height, out_width]
        Ok(result)
    }

    /// Create sliding windows over 2D spatial dimensions (pool operation).
    ///
    /// Input: `[batch, channels, height, width]`
    /// Output: `[batch, channels, out_height, out_width, kernel_height, kernel_width]`
    ///
    /// This uses the tinygrad trick: reshape to add kernel dimensions, then use
    /// expand with zero strides to create overlapping windows without data copying.
    fn _pool2d(
        &self,
        kernel_height: i64,
        kernel_width: i64,
        stride_h: i64,
        stride_w: i64,
    ) -> Result<Tensor> {
        let batch = self.shape[0]
            .as_const()
            .ok_or_else(|| anyhow!("_pool2d requires constant batch"))?;
        let channels = self.shape[1]
            .as_const()
            .ok_or_else(|| anyhow!("_pool2d requires constant channels"))?;
        let height = self.shape[2]
            .as_const()
            .ok_or_else(|| anyhow!("_pool2d requires constant height"))?;
        let width = self.shape[3]
            .as_const()
            .ok_or_else(|| anyhow!("_pool2d requires constant width"))?;

        let out_height = (height - kernel_height) / stride_h + 1;
        let out_width = (width - kernel_width) / stride_w + 1;

        if stride_h == 1 && stride_w == 1 {
            // Build all kernel-offset slices, then concatenate them into
            // [B, C, out_H, out_W, kH, kW].
            let mut patches: Vec<Tensor> =
                Vec::with_capacity((kernel_height * kernel_width) as usize);

            for ky in 0..kernel_height {
                for kx in 0..kernel_width {
                    let patch = self
                        .slice(vec![
                            (0, batch),
                            (0, channels),
                            (ky, ky + out_height),
                            (kx, kx + out_width),
                        ])?
                        .reshape(vec![batch, channels, out_height, out_width, 1, 1])?;
                    patches.push(patch);
                }
            }

            let mut rows: Vec<Tensor> = Vec::with_capacity(kernel_height as usize);
            for ky in 0..kernel_height {
                let start = (ky * kernel_width) as usize;
                let end = ((ky + 1) * kernel_width) as usize;
                let row_patches = &patches[start..end];

                let mut row = row_patches[0].clone();
                for patch in &row_patches[1..] {
                    row = row.concat(patch, 5)?;
                }
                rows.push(row);
            }

            let mut pooled = rows[0].clone();
            for row in &rows[1..] {
                pooled = pooled.concat(row, 4)?;
            }

            Ok(pooled)
        } else {
            // For stride > 1, we need to be more careful about which elements to pick
            // This is a TODO for strided convolutions
            bail!("_pool2d with stride > 1 not yet implemented")
        }
    }
}
