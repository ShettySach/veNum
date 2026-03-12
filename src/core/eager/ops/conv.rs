use anyhow::{Ok, Result};
use std::{iter::Sum, ops::Mul};

use crate::{core::eager::ETensor, core::iters::Strider, core::shape::Shape};

pub enum Mode {
    Valid,
    Full,
    Same,
}

impl<T> ETensor<T>
where
    T: Copy + Mul<Output = T> + Sum<T> + Default,
{
    pub fn correlate_1d(
        &self,
        kernel: &ETensor<T>,
        strides: &[usize; 1],
        mode: Mode,
    ) -> Result<ETensor<T>> {
        let i_zero = self.rank() - 1;
        let input_width = self.shape.sizes[i_zero];
        let input_conv_sizes = &[input_width];

        let k_zero = kernel.rank() - 1;
        let kernel_width = kernel.shape.sizes[k_zero];
        let kernel_conv_sizes = &[kernel_width];

        let range_fn = mode.range_fn(input_conv_sizes, kernel_conv_sizes)?;

        let output_sizes = mode.output_sizes(input_conv_sizes, kernel_conv_sizes, strides);
        let output_width = output_sizes[0];

        let sizes = [&self.shape.sizes[..i_zero], &output_sizes].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        for iter_index in (0..output_width).step_by(strides[0]) {
            let (input_ranges, kernel_ranges) =
                range_fn(input_conv_sizes, kernel_conv_sizes, &[iter_index]);

            let input_slice = match input_ranges {
                Some(input_ranges) => &self.slice_dims(&input_ranges, &[i_zero])?,
                None => self,
            };
            let kernel_slice = match kernel_ranges {
                Some(kernel_ranges) => &kernel.slice_dims(&kernel_ranges, &[k_zero])?,
                None => kernel,
            };
            let product_sum = (input_slice * kernel_slice)?;

            for (index, &value) in product_sum.data_contiguous().iter().enumerate() {
                let offset = index * output_width + index;
                data[offset] = value
            }
        }

        ETensor::init(data, &sizes)
    }

    pub fn correlate_2d(
        &self,
        kernel: &ETensor<T>,
        strides: &[usize; 2],
        mode: Mode,
    ) -> Result<ETensor<T>> {
        let n = self.rank();
        let input_dims = &[n - 2, n - 1];
        let input_sizes = &[
            self.shape.sizes[input_dims[0]],
            self.shape.sizes[input_dims[1]],
        ];

        let kn = kernel.rank();
        let kernel_dims = &[kn - 2, kn - 1];
        let kernel_sizes = &[
            kernel.shape.sizes[kernel_dims[0]],
            kernel.shape.sizes[kernel_dims[1]],
        ];

        let range_fn = mode.range_fn(input_sizes, kernel_sizes)?;

        let output_sizes = mode.output_sizes(input_sizes, kernel_sizes, strides);
        let output_product = output_sizes.iter().product::<usize>();
        let sizes = [&self.shape.sizes[..input_dims[0]], &output_sizes].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        for iter_index in Strider::new(&output_sizes, strides) {
            let (input_ranges, kernel_ranges) = range_fn(input_sizes, kernel_sizes, &iter_index);

            let input_slice = match input_ranges {
                Some(input_ranges) => &self.slice_dims(&input_ranges, input_dims)?,
                None => self,
            };
            let kernel_slice = match kernel_ranges {
                Some(kernel_ranges) => &kernel.slice_dims(&kernel_ranges, kernel_dims)?,
                None => kernel,
            };
            let product_sum = (input_slice * kernel_slice)?.sum_dims(input_dims, true)?;

            for (index, &value) in product_sum.data_contiguous().iter().enumerate() {
                let offset = index * output_product
                    + iter_index[0] / strides[0] * output_sizes[1]
                    + iter_index[1] / strides[1];

                data[offset] = value
            }
        }

        ETensor::init(data, &sizes)
    }

    pub fn convolve_1d(
        &self,
        kernel: &ETensor<T>,
        strides: &[usize; 1],
        mode: Mode,
    ) -> Result<ETensor<T>> {
        self.correlate_1d(&kernel.flip_all()?, strides, mode)
    }

    pub fn convolve_2d(
        &self,
        kernel: &ETensor<T>,
        strides: &[usize; 2],
        mode: Mode,
    ) -> Result<ETensor<T>> {
        self.correlate_2d(&kernel.flip_all()?, strides, mode)
    }
}

pub type Ranges = (Option<Vec<(usize, usize)>>, Option<Vec<(usize, usize)>>);
pub type RangeFn = fn(&[usize], &[usize], &[usize]) -> Ranges;

impl Mode {
    pub(crate) fn output_sizes(
        &self,
        input_sizes: &[usize],
        kernel_sizes: &[usize],
        strides: &[usize],
    ) -> Vec<usize> {
        input_sizes
            .iter()
            .zip(kernel_sizes)
            .zip(strides)
            .map(|((&i, &k), &s)| match self {
                Mode::Valid => i.abs_diff(k) / s + 1,
                Mode::Full => (i + k) / s - 1,
                Mode::Same => i / s,
            })
            .collect()
    }

    pub(crate) fn range_fn(
        &self,
        input_sizes: &[usize],
        kernel_sizes: &[usize],
    ) -> Result<RangeFn> {
        Ok(match self {
            Mode::Valid => {
                if Shape::larger_conv_input(input_sizes, kernel_sizes)? {
                    Mode::valid_ranges_i
                } else {
                    Mode::valid_ranges_k
                }
            }
            Mode::Full => Mode::full_ranges,
            Mode::Same => Mode::same_ranges,
        })
    }

    fn valid_ranges_i(_input_sizes: &[usize], kernel_sizes: &[usize], indices: &[usize]) -> Ranges {
        let input_ranges = Some(
            indices
                .iter()
                .zip(kernel_sizes)
                .map(|(&index, &kernel_size)| (index, index + kernel_size))
                .collect(),
        );

        (input_ranges, None)
    }

    fn valid_ranges_k(input_sizes: &[usize], kernel_sizes: &[usize], indices: &[usize]) -> Ranges {
        let kernel_ranges = Some(
            indices
                .iter()
                .zip(input_sizes)
                .zip(kernel_sizes)
                .map(|((index, input_size), kernel_size)| {
                    let start = kernel_size - input_size - index;
                    let end = start + input_size;
                    (start, end)
                })
                .collect(),
        );

        (None, kernel_ranges)
    }

    fn full_ranges(input_sizes: &[usize], kernel_sizes: &[usize], indices: &[usize]) -> Ranges {
        let input_ranges = Some(
            indices
                .iter()
                .zip(input_sizes)
                .zip(kernel_sizes)
                .map(|((&index, &input_size), &kernel_size)| {
                    let start = index.saturating_sub(kernel_size - 1);
                    let end = (index + 1).min(input_size);
                    (start, end)
                })
                .collect(),
        );

        let kernel_ranges = Some(
            indices
                .iter()
                .zip(input_sizes)
                .zip(kernel_sizes)
                .map(|((&index, &input_size), &kernel_size)| {
                    let start = (kernel_size - 1).saturating_sub(index);
                    let range = input_size - index.saturating_sub(kernel_size - 1);
                    let end = kernel_size.min(start + range);
                    (start, end)
                })
                .collect(),
        );

        (input_ranges, kernel_ranges)
    }

    fn same_ranges(input_sizes: &[usize], kernel_sizes: &[usize], indices: &[usize]) -> Ranges {
        let input_ranges = Some(
            indices
                .iter()
                .zip(input_sizes)
                .zip(kernel_sizes)
                .map(|((&index, &input_size), &kernel_size)| {
                    let start = index.saturating_sub(1);
                    let end = (index + kernel_size - 1).min(input_size);
                    (start, end)
                })
                .collect(),
        );

        let kernel_ranges = Some(
            indices
                .iter()
                .zip(input_sizes)
                .zip(kernel_sizes)
                .map(|((&index, &input_size), &kernel_size)| {
                    let start = 1_usize.saturating_sub(index);
                    let range = input_size - index.saturating_sub(1);
                    let end = kernel_size.min(start + range);
                    (start, end)
                })
                .collect(),
        );

        (input_ranges, kernel_ranges)
    }
}
