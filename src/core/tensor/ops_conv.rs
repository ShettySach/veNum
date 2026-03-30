//! Convolution operations for tensors.

use anyhow::{Result, anyhow, bail};

use super::structure::Tensor;

impl Tensor {
    /// 2D convolution (cross-correlation) as used in CNNs.
    ///
    /// - `self` (input):  `[batch_size, channels_in, input_height, input_width]`
    /// - `weight`:        `[channels_out, channels_in, kernel_height, kernel_width]`
    /// - Output:          `[batch_size, channels_out, output_height, output_width]`
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

        let batch_size = self.shape[0];
        let channels_in = self.shape[1];
        let inp_height = self.shape[2];
        let inp_width = self.shape[3];

        let channels_out = weight.shape[0];
        let kernel_channels_in = weight.shape[1];
        let kernel_height = weight.shape[2];
        let kernel_width = weight.shape[3];

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

        let mut accumulator: Option<Tensor> = None;

        for kernel_y in 0..kernel_height {
            for kernel_x in 0..kernel_width {
                let patch = self.slice(vec![
                    (0, batch_size),
                    (0, channels_in),
                    (kernel_y, kernel_y + output_height),
                    (kernel_x, kernel_x + output_width),
                ])?;
                let w_slice = weight.slice(vec![
                    (0, channels_out),
                    (0, channels_in),
                    (kernel_y, kernel_y + 1),
                    (kernel_x, kernel_x + 1),
                ])?;

                let patch = patch.reshape(vec![
                    batch_size,
                    1,
                    channels_in,
                    output_height,
                    output_width,
                ])?;
                let weight_slice = w_slice.reshape(vec![1, channels_out, channels_in, 1, 1])?;

                let prod_sum = patch.mul(&weight_slice)?.sum(&[2], false)?;

                accumulator = Some(match accumulator {
                    Some(prev) => prev.add(&prod_sum)?,
                    None => prod_sum,
                });
            }
        }

        accumulator.ok_or_else(|| anyhow!("conv2d: empty kernel"))
    }
}
