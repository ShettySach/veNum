use anyhow::{bail, Result};
use std::{
    cmp::{max, Ordering},
    collections::HashSet,
};

use crate::core::errors::*;

#[derive(Clone)]
pub(crate) struct Shape {
    pub sizes: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: usize,
}

impl Shape {
    pub fn new(sizes: &[usize]) -> Shape {
        let mut strides: Vec<isize> = sizes
            .iter()
            .rev()
            .scan(1, |current, size| {
                let value = *current as isize;
                *current *= size;
                Some(value)
            })
            .collect();
        strides.reverse();

        Shape {
            sizes: sizes.to_vec(),
            strides,
            offset: 0,
        }
    }

    pub(crate) fn scalar() -> Shape {
        Shape {
            sizes: vec![1],
            strides: vec![1],
            offset: 0,
        }
    }

    pub(crate) fn rank(&self) -> usize {
        self.sizes.len()
    }

    pub(crate) fn numel(&self) -> usize {
        self.sizes.iter().product()
    }

    // --- Shape operations ---

    pub(crate) fn view(&self, sizes: &[usize]) -> Result<Shape> {
        self.valid_contiguity()?;
        self.valid_reshape(sizes)?;

        let positive = self
            .strides
            .first()
            .ok_or(EmptyTensorError::View)?
            .is_positive();

        let mut strides: Vec<isize> = sizes
            .iter()
            .rev()
            .scan(1, |current, size| {
                let value = *current as isize;
                *current *= size;

                Some(if positive { value } else { -value })
            })
            .collect::<Vec<isize>>();
        strides.reverse();

        Ok(Shape {
            sizes: sizes.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    pub(crate) fn permute(&self, permutation: &[usize]) -> Result<Shape> {
        self.valid_rank(permutation.len())?;
        self.valid_dimensions(permutation)?;

        let (sizes, strides) = permutation
            .iter()
            .map(|i| (self.sizes[*i], self.strides[*i]))
            .collect();

        Ok(Shape {
            sizes,
            strides,
            offset: self.offset,
        })
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Result<Shape> {
        let rank = self.rank();
        if rank < 2 {
            bail!(TransposeError);
        }

        let mut permutation = Vec::from_iter(0..rank);
        permutation.swap(dim_1, dim_2);

        self.permute(&permutation)
    }

    pub(crate) fn flip(&self, flips: &[usize]) -> Result<Shape> {
        self.valid_dimensions(flips)?;

        let mut strides = self.strides.to_vec();
        flips.iter().for_each(|&f| strides[f] = -strides[f]);

        Ok(Shape {
            sizes: self.sizes.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    pub(crate) fn flip_all(&self) -> Result<Shape> {
        let strides = self.strides.iter().map(|s| -s).collect();

        Ok(Shape {
            sizes: self.sizes.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    pub(crate) fn expand(&self, expansions: &[usize]) -> Result<Shape> {
        self.valid_rank(expansions.len())?;

        let (sizes, strides) = self
            .sizes
            .iter()
            .zip(self.strides.iter())
            .zip(expansions)
            .map(|((&size, &stride), &expansion)| {
                if expansion == size {
                    Ok((size, stride))
                } else if size == 1 {
                    Ok((expansion, 0))
                } else {
                    Err(ExpansionError { size, expansion }.into())
                }
            })
            .collect::<Result<(Vec<usize>, Vec<isize>)>>()?;

        Ok(Shape {
            sizes,
            strides,
            offset: self.offset,
        })
    }

    pub(crate) fn squeeze(&self) -> Result<Shape> {
        if self.numel() == 1 {
            return Ok(Shape {
                sizes: vec![1],
                strides: vec![1],
                offset: self.offset,
            });
        }

        let (sizes, strides) = self
            .sizes
            .iter()
            .zip(&self.strides)
            .filter_map(|(&size, &stride)| (size != 1).then_some((size, stride)))
            .collect();

        Ok(Shape {
            sizes,
            strides,
            offset: self.offset,
        })
    }

    pub(crate) fn unsqueeze(&self, new_rank: usize) -> Result<Shape> {
        let current = self.rank();

        match new_rank.cmp(&current) {
            Ordering::Greater => {
                let ones_len = new_rank - current;
                let mut sizes = self.sizes.to_vec();
                sizes.splice(..0, std::iter::repeat_n(1, ones_len));

                Ok(Shape::new(&sizes))
            }
            Ordering::Equal => Ok(self.clone()),
            Ordering::Less => Err(UnsqueezeError { current, new_rank }.into()),
        }
    }

    // --- Index, Slice and Pad ---

    pub(crate) fn idx(&self, indices: &[usize]) -> usize {
        self.sizes
            .iter()
            .zip(self.strides.iter())
            .zip(indices)
            .map(|((&size, &stride), &index)| offset_fn(stride, index, size))
            .sum::<usize>()
            + self.offset
    }

    pub(crate) fn index(&self, indices: &[usize]) -> Result<usize> {
        self.valid_indices(indices)?;

        Ok(self.idx(indices))
    }

    pub(crate) fn index_dims(&self, indices: &[usize], dimensions: &[usize]) -> Result<usize> {
        self.valid_dimensions(dimensions)?;
        self.valid_indices_dims(indices, dimensions)?;

        Ok((0..self.rank())
            .map(|dimension| {
                let size = self.sizes[dimension];
                let stride = self.strides[dimension];

                if let Some(position) = dimensions.iter().position(|&d| d == dimension) {
                    offset_fn(stride, indices[position], size)
                } else {
                    offset_fn(stride, 0, size)
                }
            })
            .sum::<usize>()
            + self.offset)
    }

    pub(crate) fn slice(&self, ranges: &[(usize, usize)]) -> Result<Shape> {
        self.valid_contiguity()?;

        self.valid_ranges(ranges)?;

        let positive = self
            .strides
            .first()
            .ok_or(EmptyTensorError::Slice)?
            .is_positive();

        let mut offset = if positive {
            self.offset
        } else {
            self.numel() - 1 - self.offset
        };

        let sizes: Vec<usize> = self
            .sizes
            .iter()
            .zip(&self.strides)
            .zip(ranges)
            .map(|((&size, &stride), &(start, end))| {
                let end = if end == 0 { size } else { end };

                if positive {
                    offset += start * stride as usize;
                } else {
                    offset -= (end - 1) * -stride as usize;
                }

                end - start
            })
            .collect();

        Ok(Shape {
            sizes,
            strides: self.strides.to_vec(),
            offset,
        })
    }

    pub(crate) fn slice_dims(
        &self,
        ranges: &[(usize, usize)],
        dimensions: &[usize],
    ) -> Result<Shape> {
        self.valid_contiguity()?;
        self.valid_dimensions(dimensions)?;
        self.valid_ranges_dims(ranges, dimensions)?;

        let positive = self
            .strides
            .first()
            .ok_or(EmptyTensorError::Slice)?
            .is_positive();

        let mut offset = if positive {
            self.offset
        } else {
            self.numel() - 1 - self.offset
        };

        let sizes = (0..self.rank())
            .map(|dimension| {
                let stride = self.strides[dimension];

                if let Some(position) = dimensions.iter().position(|&d| d == dimension) {
                    let (start, end) = ranges[position];
                    let end = if end == 0 { self.sizes[dimension] } else { end };

                    if positive {
                        offset += start * stride as usize
                    } else {
                        offset -= (end - 1) * -stride as usize
                    };

                    end - start
                } else {
                    let size = self.sizes[dimension];

                    if !positive {
                        offset -= size * -stride as usize;
                    };

                    size
                }
            })
            .collect();

        Ok(Shape {
            sizes,
            strides: self.strides.to_vec(),
            offset,
        })
    }

    pub(crate) fn pad(&self, padding: &[(usize, usize)]) -> Result<Shape> {
        let mut padding = padding.to_vec();
        padding.resize(self.rank(), (0, 0));

        let sizes = self
            .sizes
            .iter()
            .zip(padding)
            .map(|(&size, (start, end))| start + size + end)
            .collect::<Vec<usize>>();

        Ok(Shape::new(&sizes))
    }

    pub(crate) fn pad_dims(
        &self,
        padding: &[(usize, usize)],
        dimensions: &[usize],
    ) -> Result<Shape> {
        self.valid_dimensions(dimensions)?;

        let sizes = (0..self.rank())
            .map(|dimension| {
                if let Some(position) = dimensions.iter().position(|&d| d == dimension) {
                    let (start, end) = padding[position];
                    start + self.sizes[dimension] + end
                } else {
                    self.sizes[dimension]
                }
            })
            .collect::<Vec<usize>>();

        Ok(Shape::new(&sizes))
    }

    pub(crate) fn slicer(&self, indices: &[Option<usize>]) -> Result<Shape> {
        self.valid_contiguity()?;

        let positive = self
            .strides
            .first()
            .ok_or(EmptyTensorError::Slice)?
            .is_positive();

        let mut offset = if positive {
            self.offset
        } else {
            self.numel() - 1 - self.offset
        };

        let sizes = self
            .sizes
            .iter()
            .zip(&self.strides)
            .zip(indices)
            .map(|((&size, &stride), i)| {
                if let Some(i) = i {
                    if positive {
                        offset += i * stride as usize
                    } else {
                        offset -= i * -stride as usize
                    }

                    1
                } else {
                    size
                }
            })
            .collect();

        Ok(Shape {
            sizes,
            strides: self.strides.to_vec(),
            offset,
        })
    }

    // --- Broadcast ---

    pub(crate) fn broadcast(lhs_sizes: &[usize], rhs_sizes: &[usize]) -> Result<Vec<usize>> {
        let mut lhs_iter = lhs_sizes.iter();
        let mut rhs_iter = rhs_sizes.iter();

        let max_len = max(lhs_sizes.len(), rhs_sizes.len());
        let mut result = Vec::with_capacity(max_len);

        loop {
            match (lhs_iter.next_back(), rhs_iter.next_back()) {
                (Some(&l), Some(&r)) => {
                    if l == r {
                        result.push(l);
                    } else if l == 1 {
                        result.push(r);
                    } else if r == 1 {
                        result.push(l);
                    } else {
                        bail!(BroadcastError {
                            lhs_sizes: lhs_sizes.to_vec(),
                            rhs_sizes: rhs_sizes.to_vec(),
                        });
                    }
                }
                (Some(&l), None) => result.push(l),
                (None, Some(&r)) => result.push(r),
                (None, None) => break,
            }
        }

        result.reverse();
        Ok(result)
    }

    // --- Validation ---

    pub(crate) fn is_contiguous(&self) -> bool {
        for i in 0..self.rank() - 1 {
            if self.strides[i] != self.strides[i + 1] * self.sizes[i + 1] as isize {
                return false;
            }
        }

        true
    }

    pub(crate) fn valid_contiguity(&self) -> Result<()> {
        if self.is_contiguous() {
            Ok(())
        } else {
            Err(NonContiguousError.into())
        }
    }

    pub(crate) fn valid_reshape(&self, sizes: &[usize]) -> Result<()> {
        if self.numel() != sizes.iter().product::<usize>() {
            bail!(ReshapeError {
                current_shape: self.sizes.to_vec(),
                new_shape: sizes.to_vec(),
            });
        }

        Ok(())
    }

    fn valid_indices(&self, indices: &[usize]) -> Result<()> {
        for (dimension, &index) in indices.iter().enumerate() {
            let size = self.sizes[dimension];

            if index >= size {
                bail!(IndexError::OutOfRange {
                    index,
                    dimension,
                    size,
                });
            }
        }

        Ok(())
    }

    fn valid_indices_dims(&self, indices: &[usize], dimensions: &[usize]) -> Result<()> {
        for (&dimension, &index) in dimensions.iter().zip(indices) {
            let size = self.sizes[dimension];

            if index >= size {
                bail!(IndexError::OutOfRange {
                    index,
                    dimension,
                    size,
                });
            }
        }

        Ok(())
    }

    fn valid_ranges(&self, ranges: &[(usize, usize)]) -> Result<()> {
        for (dimension, &range) in ranges.iter().enumerate() {
            let size = self.sizes[dimension];

            if range.0 > range.1 && range.1 > 0 {
                bail!(RangeError::GreaterStartRange(range.0, range.1));
            } else if range.0 > size || range.1 > size {
                bail!(RangeError::OutOfRange {
                    range,
                    dimension,
                    size,
                });
            }
        }

        Ok(())
    }

    fn valid_ranges_dims(&self, ranges: &[(usize, usize)], dimensions: &[usize]) -> Result<()> {
        for (&dimension, &range) in dimensions.iter().zip(ranges) {
            let size = self.sizes[dimension];

            if range.0 > range.1 && range.1 > 0 {
                bail!(RangeError::GreaterStartRange(range.0, range.1));
            } else if range.0 > size || range.1 > size {
                bail!(RangeError::OutOfRange {
                    range,
                    dimension,
                    size,
                });
            }
        }

        Ok(())
    }

    pub(crate) fn valid_dimensions(&self, dimensions: &[usize]) -> Result<()> {
        let dim_range = self.rank() - 1;
        let mut set = HashSet::with_capacity(dimensions.len());

        for &dimension in dimensions {
            if dim_range < dimension {
                bail!(DimensionError::OutOfRange {
                    dimension,
                    dim_range,
                });
            } else if set.contains(&dimension) {
                bail!(DimensionError::Repetition(dimension));
            } else {
                set.insert(dimension);
            }
        }

        Ok(())
    }

    fn valid_rank(&self, num_indices: usize) -> Result<()> {
        let num_dimensions = self.rank();

        if num_indices != num_dimensions {
            Err(IndexError::IndicesLength {
                num_indices,
                num_dimensions,
            }
            .into())
        } else {
            Ok(())
        }
    }

    pub(crate) fn valid_data_size(&self, data_size: usize) -> Result<()> {
        let tensor_size = self.numel();

        if data_size != tensor_size {
            bail!(InvalidDataSizeError {
                data_size,
                tensor_size,
            })
        }

        Ok(())
    }

    pub(crate) fn larger_conv_input(input_sizes: &[usize], kernel_sizes: &[usize]) -> Result<bool> {
        if input_sizes.iter().zip(kernel_sizes).all(|(&i, &k)| i >= k) {
            Ok(true)
        } else if input_sizes.iter().zip(kernel_sizes).all(|(&i, &k)| k >= i) {
            Ok(false)
        } else {
            Err((ValidConvShapeError {
                input_sizes: input_sizes.to_vec(),
                kernel_sizes: kernel_sizes.to_vec(),
            })
            .into())
        }
    }
}

impl PartialEq for Shape {
    fn eq(&self, rhs: &Shape) -> bool {
        self.sizes == rhs.sizes && self.strides == rhs.strides
    }
}

pub(crate) fn offset_fn(stride: isize, index: usize, size: usize) -> usize {
    if stride.is_positive() {
        index * stride as usize
    } else {
        (size - 1 - index) * -stride as usize
    }
}
