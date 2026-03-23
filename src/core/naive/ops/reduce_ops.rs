use anyhow::Result;
use num_traits::FromPrimitive;
use std::{
    cmp::Ordering,
    iter::{Product, Sum},
    ops::Div,
};

use crate::{
    core::errors::EmptyTensorError, core::iters::Indexer, core::naive::NaiveTensor,
    core::utils::cast_to_usize,
};

impl<T> NaiveTensor<T>
where
    T: Copy,
{
    pub fn sum(&self) -> Result<T>
    where
        T: Sum<T>,
    {
        let sum = if self.is_contiguous() {
            self.data_contiguous().iter().copied().sum()
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| self.idx(&index))
                .sum()
        };

        Ok(sum)
    }

    pub fn mean(&self) -> Result<T>
    where
        T: Sum<T> + Div<T, Output = T> + FromPrimitive,
    {
        let numel = self.numel();
        let numel_casted = cast_to_usize(numel)?;

        Ok(self.sum()? / numel_casted)
    }

    pub fn product(&self) -> Result<T>
    where
        T: Product<T>,
    {
        let product = if self.is_contiguous() {
            self.data_contiguous().iter().copied().product()
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| self.idx(&index))
                .product()
        };

        Ok(product)
    }

    pub fn max(&self) -> Result<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            let data = self.data_contiguous();
            let mut iter = data.iter().copied();
            let mut current_max = iter.next().ok_or(EmptyTensorError::ReduceMax)?;
            for value in iter {
                if let Some(Ordering::Greater) = value.partial_cmp(&current_max) {
                    current_max = value;
                }
            }
            Ok(current_max)
        } else {
            let mut iter = Indexer::new(&self.shape.sizes).map(|index| self.idx(&index));
            let mut current_max = iter.next().ok_or(EmptyTensorError::ReduceMax)?;
            for value in iter {
                if let Some(Ordering::Greater) = value.partial_cmp(&current_max) {
                    current_max = value;
                }
            }
            Ok(current_max)
        }
    }

    pub fn min(&self) -> Result<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            let data = self.data_contiguous();
            let mut iter = data.iter().copied();
            let mut current_min = iter.next().ok_or(EmptyTensorError::ReduceMin)?;
            for value in iter {
                if let Some(Ordering::Less) = value.partial_cmp(&current_min) {
                    current_min = value;
                }
            }
            Ok(current_min)
        } else {
            let mut iter = Indexer::new(&self.shape.sizes).map(|index| self.idx(&index));
            let mut current_min = iter.next().ok_or(EmptyTensorError::ReduceMin)?;
            for value in iter {
                if let Some(Ordering::Less) = value.partial_cmp(&current_min) {
                    current_min = value;
                }
            }
            Ok(current_min)
        }
    }

    pub fn sum_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<NaiveTensor<T>>
    where
        T: Sum<T>,
    {
        self.reduce(NaiveTensor::sum, dimensions, keepdims)
    }

    pub fn mean_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<NaiveTensor<T>>
    where
        T: Sum<T> + Div<T, Output = T> + FromPrimitive,
    {
        self.reduce(NaiveTensor::mean, dimensions, keepdims)
    }

    pub fn product_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<NaiveTensor<T>>
    where
        T: Product<T>,
    {
        self.reduce(NaiveTensor::product, dimensions, keepdims)
    }

    pub fn max_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<NaiveTensor<T>>
    where
        T: PartialOrd,
    {
        self.reduce(NaiveTensor::max, dimensions, keepdims)
    }

    pub fn min_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<NaiveTensor<T>>
    where
        T: PartialOrd,
    {
        self.reduce(NaiveTensor::min, dimensions, keepdims)
    }
}
