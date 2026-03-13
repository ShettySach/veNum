use anyhow::{bail, Result};
use num_traits::{FromPrimitive, NumOps, One, Zero};
use std::{borrow::Cow, cmp::Ordering, fmt::Debug, iter::successors, ops::Add, sync::Arc};

use crate::core::{
    errors::*,
    iters::{Indexer, Slicer},
    shape::Shape,
    utils::cast_to_usize,
};

pub struct NaiveTensor<T> {
    pub(crate) data: Arc<Vec<T>>,
    pub(crate) shape: Shape,
}

impl<T: Copy> NaiveTensor<T> {
    pub(crate) fn init(data: Vec<T>, sizes: &[usize]) -> Result<NaiveTensor<T>> {
        Ok(NaiveTensor {
            data: Arc::new(data),
            shape: Shape::new(sizes),
        })
    }

    pub fn new(data: &[T], sizes: &[usize]) -> Result<NaiveTensor<T>> {
        let data_size = data.len();
        let tensor_size = sizes.iter().product();

        if data_size != tensor_size {
            bail!(InvalidDataSizeError {
                data_size,
                tensor_size
            });
        }

        NaiveTensor::init(data.to_vec(), sizes)
    }

    pub fn new_1d(data: &[T]) -> Result<NaiveTensor<T>> {
        NaiveTensor::init(data.to_vec(), &[data.len()])
    }

    pub fn scalar(data: T) -> Result<NaiveTensor<T>> {
        Ok(NaiveTensor {
            data: Arc::new(vec![data]),
            shape: Shape::scalar(),
        })
    }

    pub fn same(element: T, size: usize) -> Result<NaiveTensor<T>> {
        NaiveTensor::init(vec![element; size], &[size])
    }

    pub fn zeroes(size: usize) -> Result<NaiveTensor<T>>
    where
        T: Zero,
    {
        NaiveTensor::same(T::zero(), size)
    }

    pub fn ones(size: usize) -> Result<NaiveTensor<T>>
    where
        T: One,
    {
        NaiveTensor::same(T::one(), size)
    }

    pub fn eye(size: usize) -> Result<NaiveTensor<T>>
    where
        T: Zero + One,
    {
        let diagonal = size + 1;
        let data = (0..size * size)
            .map(|elem| {
                if elem % diagonal == 0 {
                    T::one()
                } else {
                    T::zero()
                }
            })
            .collect();

        NaiveTensor::init(data, &[size, size])
    }

    pub fn arange(start: T, end: T, step: T) -> Result<NaiveTensor<T>>
    where
        T: Add<Output = T> + PartialOrd + Zero,
    {
        let ascending = match step
            .partial_cmp(&T::zero())
            .ok_or(ArangeError::Comparison)?
        {
            Ordering::Greater if end > start => Ok(true),
            Ordering::Less if start > end => Ok(false),
            Ordering::Greater => Err(ArangeError::Positive),
            Ordering::Less => Err(ArangeError::Negative),
            Ordering::Equal => Err(ArangeError::Zero),
        }?;

        let data: Vec<T> = successors(Some(start), |&prev| {
            let curr = prev + step;
            let cond = end > curr;
            (ascending == cond).then_some(curr)
        })
        .collect();

        NaiveTensor::new_1d(&data)
    }

    pub fn linspace(start: T, end: T, num: usize) -> Result<NaiveTensor<T>>
    where
        T: NumOps + FromPrimitive + Debug,
    {
        let num_casted = cast_to_usize::<T>(num - 1)?;
        let step = (end - start) / num_casted;

        let data = successors(Some(start), |&prev| Some(prev + step))
            .take(num)
            .collect();

        NaiveTensor::init(data, &[num])
    }

    // --- Data ---

    pub fn to_contiguous(&self) -> Result<NaiveTensor<T>> {
        Ok(NaiveTensor {
            data: Arc::new(self.data_non_contiguous()),
            shape: Shape::new(&self.shape.sizes),
        })
    }

    pub(crate) fn into_contiguous(self) -> Result<NaiveTensor<T>> {
        if self.is_contiguous() {
            Ok(self)
        } else {
            self.to_contiguous()
        }
    }

    pub fn data_contiguous_positive_strides(&self) -> &[T] {
        let start = self.offset();
        let end = start + self.numel();
        &self.data[start..end]
    }

    pub fn data_contiguous_negative_strides(&self) -> Vec<T> {
        let start = self.offset();
        let end = start + self.numel();

        let mut data = self.data[start..end].to_vec();
        data.reverse();
        data
    }

    pub fn data_contiguous(&'_ self) -> Cow<'_, [T]> {
        if self.shape.strides[0].is_positive() {
            Cow::Borrowed(self.data_contiguous_positive_strides())
        } else {
            Cow::Owned(self.data_contiguous_negative_strides())
        }
    }

    pub fn data_non_contiguous(&self) -> Vec<T> {
        Indexer::new(&self.shape.sizes)
            .map(|index| self.idx(&index))
            .collect()
    }

    pub fn data(&'_ self) -> Cow<'_, [T]> {
        if self.is_contiguous() {
            self.data_contiguous()
        } else {
            Cow::Owned(self.data_non_contiguous())
        }
    }

    pub(crate) fn idx(&self, indices: &[usize]) -> T {
        self.data[self.shape.idx(indices)]
    }

    pub fn index(&self, indices: &[usize]) -> Result<T> {
        Ok(self.data[self.shape.index(indices)?])
    }

    pub fn index_dims(&self, indices: &[usize], dimensions: &[usize]) -> Result<T> {
        Ok(self.data[self.shape.index_dims(indices, dimensions)?])
    }

    // --- New Data, New Shape ---

    pub fn reshape(&self, sizes: &[usize]) -> Result<NaiveTensor<T>> {
        self.shape.valid_reshape(sizes)?;

        NaiveTensor::init(self.data_non_contiguous(), sizes)
    }

    pub fn flatten(&self) -> Result<NaiveTensor<T>> {
        self.reshape(&[self.numel()])
    }

    pub fn view_else_reshape(&self, sizes: &[usize]) -> Result<NaiveTensor<T>> {
        self.view(sizes).or_else(|_| self.reshape(sizes))
    }

    pub fn pad(&self, constant: T, padding: &[(usize, usize)]) -> Result<NaiveTensor<T>> {
        let shape = self.shape.pad(padding)?;
        let data = Arc::new(vec![constant; shape.numel()]);
        let tensor = NaiveTensor { data, shape };

        let ranges = padding
            .iter()
            .enumerate()
            .map(|(dimension, &(start, _))| (start, self.shape.sizes[dimension] + start))
            .collect::<Vec<(usize, usize)>>();

        tensor.slice_zip(&self.data(), |_, new| new, &ranges)
    }

    pub fn pad_dims(
        &self,
        constant: T,
        padding: &[(usize, usize)],
        dimensions: &[usize],
    ) -> Result<NaiveTensor<T>> {
        let shape = self.shape.pad_dims(padding, dimensions)?;
        let data = Arc::new(vec![constant; shape.numel()]);
        let tensor = NaiveTensor { data, shape };

        let ranges = dimensions
            .iter()
            .zip(padding)
            .map(|(&dimension, &(start, _))| (start, self.shape.sizes[dimension] + start))
            .collect::<Vec<(usize, usize)>>();

        tensor.slice_zip_dims(&self.data(), |_, new| new, &ranges, dimensions)
    }

    // --- Maps, Zips and Reduce ---

    pub fn unary_map<R>(&self, f: impl Fn(T) -> R) -> Result<NaiveTensor<R>> {
        let contiguous = self.is_contiguous();

        let data = if contiguous {
            self.data_contiguous().iter().map(|&elem| f(elem)).collect()
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| {
                    let elem = self.idx(&index);
                    f(elem)
                })
                .collect()
        };

        let shape = if contiguous {
            Shape {
                sizes: self.sizes().to_vec(),
                strides: self.strides().to_vec(),
                offset: 0,
            }
        } else {
            Shape::new(self.sizes())
        };

        Ok(NaiveTensor {
            data: Arc::new(data),
            shape,
        })
    }

    pub fn binary_map<R>(&self, rhs: T, f: impl Fn(T, T) -> R) -> Result<NaiveTensor<R>> {
        let contiguous = self.is_contiguous();

        let data = if contiguous {
            self.data_contiguous()
                .iter()
                .map(|&elem| f(elem, rhs))
                .collect()
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| {
                    let lhs_elem = self.idx(&index);
                    f(lhs_elem, rhs)
                })
                .collect()
        };

        let shape = if contiguous {
            Shape {
                sizes: self.sizes().to_vec(),
                strides: self.strides().to_vec(),
                offset: 0,
            }
        } else {
            Shape::new(self.sizes())
        };

        Ok(NaiveTensor {
            data: Arc::new(data),
            shape,
        })
    }

    pub fn zip<R>(&self, rhs: &NaiveTensor<T>, f: impl Fn(T, T) -> R) -> Result<NaiveTensor<R>> {
        if self.shape == rhs.shape {
            self.equal_zip(rhs, f)
        } else {
            self.broadcast_zip(rhs, f)
        }
    }

    fn equal_zip<R>(&self, rhs: &NaiveTensor<T>, f: impl Fn(T, T) -> R) -> Result<NaiveTensor<R>> {
        let contiguous = self.is_contiguous() && rhs.is_contiguous();

        let data = if contiguous {
            self.data_contiguous()
                .iter()
                .zip(rhs.data_contiguous().iter())
                .map(|(lhs_elem, rhs_elem)| f(*lhs_elem, *rhs_elem))
                .collect()
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| {
                    let lhs_elem = self.idx(&index);
                    let rhs_elem = rhs.idx(&index);

                    f(lhs_elem, rhs_elem)
                })
                .collect()
        };

        let shape = if contiguous {
            Shape {
                sizes: self.sizes().to_vec(),
                strides: self.strides().to_vec(),
                offset: 0,
            }
        } else {
            Shape::new(self.sizes())
        };

        Ok(NaiveTensor {
            data: Arc::new(data),
            shape,
        })
    }

    fn broadcast_zip<R>(
        &self,
        rhs: &NaiveTensor<T>,
        f: impl Fn(T, T) -> R,
    ) -> Result<NaiveTensor<R>> {
        let sizes = Shape::broadcast(&self.shape.sizes, &rhs.shape.sizes)?;
        let shape = Shape::new(&sizes);
        let expansion = sizes.len();

        let lhs_broadcasted = if self.shape.sizes == sizes {
            self
        } else {
            &self.unsqueeze(expansion)?.expand(&sizes)?
        };
        let rhs_broadcasted = if rhs.shape.sizes == sizes {
            rhs
        } else {
            &rhs.unsqueeze(expansion)?.expand(&sizes)?
        };

        let data = Arc::new(
            Indexer::new(&shape.sizes)
                .map(|index| {
                    let lhs_elem = lhs_broadcasted.idx(&index);
                    let rhs_elem = rhs_broadcasted.idx(&index);

                    f(lhs_elem, rhs_elem)
                })
                .collect(),
        );

        Ok(NaiveTensor { data, shape })
    }

    pub fn zip_array<R>(&self, rhs: &[T], f: impl Fn(T, T) -> R) -> Result<NaiveTensor<R>> {
        self.shape.valid_data_size(rhs.len())?;

        let data = Indexer::new(&self.shape.sizes)
            .zip(rhs)
            .map(|(index, &rhs_elem)| {
                let offset = self.shape.index(&index)?;
                let lhs_elem = self.data[offset];

                Ok(f(lhs_elem, rhs_elem))
            })
            .collect::<Result<Vec<R>>>()?;

        Ok(NaiveTensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn reduce<R>(
        &self,
        f: impl Fn(&NaiveTensor<T>) -> Result<R>,
        dimensions: &[usize],
        keepdims: bool,
    ) -> Result<NaiveTensor<R>>
    where
        R: Copy,
    {
        self.shape.valid_dimensions(dimensions)?;

        let data = Slicer::new(&self.shape.sizes, dimensions, keepdims)
            .map(|index| f(&self.slicer(&index)?))
            .collect::<Result<Vec<R>>>()?;

        let sizes: Vec<usize> = self
            .shape
            .sizes
            .iter()
            .enumerate()
            .map(|(d, &size)| {
                if keepdims == dimensions.contains(&d) {
                    1
                } else {
                    size
                }
            })
            .collect();

        NaiveTensor::init(data, &sizes)
    }

    pub fn index_map(&self, f: impl Fn(T) -> T, index: &[usize]) -> Result<NaiveTensor<T>> {
        let mut data = self.data().to_vec();
        let offset = self.shape.index(index)?;
        data[offset] = f(data[offset]);

        Ok(NaiveTensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn index_map_dims(
        &self,
        f: impl Fn(T) -> T,
        index: &[usize],
        dimensions: &[usize],
    ) -> Result<NaiveTensor<T>> {
        let mut data = self.data().to_vec();
        let offset = self.shape.index_dims(index, dimensions)?;
        data[offset] = f(data[offset]);

        Ok(NaiveTensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn slice_map(
        &self,
        f: impl Fn(T) -> T,
        ranges: &[(usize, usize)],
    ) -> Result<NaiveTensor<T>> {
        let slice_shape = self.shape.slice(ranges)?;

        let mut data = self.data().to_vec();
        for index in Indexer::new(&slice_shape.sizes) {
            let offset = slice_shape.idx(&index);
            data[offset] = f(data[offset]);
        }

        Ok(NaiveTensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn slice_map_dims(
        &self,
        f: impl Fn(T) -> T,
        ranges: &[(usize, usize)],
        dimensions: &[usize],
    ) -> Result<NaiveTensor<T>> {
        let mut data = self.data().to_vec();
        let slice_shape = self.shape.slice_dims(ranges, dimensions)?;

        for index in Indexer::new(&slice_shape.sizes) {
            let offset = slice_shape.idx(&index);
            data[offset] = f(data[offset]);
        }

        Ok(NaiveTensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn slice_zip(
        &self,
        rhs: &[T],
        f: impl Fn(T, T) -> T,
        ranges: &[(usize, usize)],
    ) -> Result<NaiveTensor<T>> {
        let slice_shape = self.shape.slice(ranges)?;
        slice_shape.valid_data_size(rhs.len())?;

        let mut data = self.data().to_vec();
        for (index, &rhs_value) in Indexer::new(&slice_shape.sizes).zip(rhs) {
            let offset = slice_shape.idx(&index);
            data[offset] = f(data[offset], rhs_value);
        }

        Ok(NaiveTensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }

    pub fn slice_zip_dims(
        &self,
        rhs: &[T],
        f: impl Fn(T, T) -> T,
        ranges: &[(usize, usize)],
        dimensions: &[usize],
    ) -> Result<NaiveTensor<T>> {
        let slice_shape = self.shape.slice_dims(ranges, dimensions)?;
        slice_shape.valid_data_size(rhs.len())?;

        let mut data = self.data().to_vec();
        for (index, &rhs_value) in Indexer::new(&slice_shape.sizes).zip(rhs) {
            let offset = slice_shape.idx(&index);
            data[offset] = f(data[offset], rhs_value);
        }

        Ok(NaiveTensor {
            data: Arc::new(data),
            shape: self.shape.clone(),
        })
    }
}

impl<T> NaiveTensor<T> {
    // --- Same Data, Different Shape ---

    pub(crate) fn with_shape(&self, shape: Shape) -> Result<NaiveTensor<T>> {
        Ok(NaiveTensor {
            data: Arc::clone(&self.data),
            shape,
        })
    }

    pub fn view(&self, sizes: &[usize]) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.view(sizes)?)
    }

    pub fn ravel(&self) -> Result<NaiveTensor<T>> {
        self.view(&[self.numel()])
    }

    pub fn squeeze(&self) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.squeeze()?)
    }

    pub fn unsqueeze(&self, unsqueezed: usize) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.unsqueeze(unsqueezed)?)
    }

    pub fn permute(&self, permutation: &[usize]) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.permute(permutation)?)
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.transpose(dim_1, dim_2)?)
    }

    pub fn expand(&self, expansions: &[usize]) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.expand(expansions)?)
    }

    pub fn flip(&self, flips: &[usize]) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.flip(flips)?)
    }

    pub fn flip_all(&self) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.flip_all()?)
    }

    pub fn slice(&self, ranges: &[(usize, usize)]) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.slice(ranges)?)
    }

    pub fn slice_dims(
        &self,
        ranges: &[(usize, usize)],
        dimensions: &[usize],
    ) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.slice_dims(ranges, dimensions)?)
    }

    pub(crate) fn slicer(&self, indices: &[Option<usize>]) -> Result<NaiveTensor<T>> {
        self.with_shape(self.shape.slicer(indices)?)
    }

    // --- Shape Attributes ---

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn sizes(&self) -> &[usize] {
        &self.shape.sizes
    }

    pub fn strides(&self) -> &[isize] {
        &self.shape.strides
    }

    pub fn offset(&self) -> usize {
        self.shape.offset
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous()
    }
}

impl<T: Copy + PartialEq> PartialEq for NaiveTensor<T> {
    fn eq(&self, rhs: &NaiveTensor<T>) -> bool {
        self.data == rhs.data && self.shape == rhs.shape
    }
}
