use anyhow::Result;
use num_traits::FromPrimitive;
use std::{
    iter::{Product, Sum},
    ops::Div,
};

use crate::{
    core::eager::ETensor, core::errors::EmptyTensorError, core::iters::Indexer,
    core::utils::cast_to_usize,
};

impl<T> ETensor<T>
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
        let max = if self.is_contiguous() {
            self.data_contiguous()
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| self.idx(&index))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
        };

        max.ok_or(EmptyTensorError::ReduceMax.into())
    }

    pub fn min(&self) -> Result<T>
    where
        T: PartialOrd,
    {
        let min = if self.is_contiguous() {
            self.data_contiguous()
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        } else {
            Indexer::new(&self.shape.sizes)
                .map(|index| self.idx(&index))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        };

        min.ok_or(EmptyTensorError::ReduceMin.into())
    }

    pub fn sum_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<ETensor<T>>
    where
        T: Sum<T>,
    {
        self.reduce(ETensor::sum, dimensions, keepdims)
    }

    pub fn mean_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<ETensor<T>>
    where
        T: Sum<T> + Div<T, Output = T> + FromPrimitive,
    {
        self.reduce(ETensor::mean, dimensions, keepdims)
    }

    pub fn product_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<ETensor<T>>
    where
        T: Product<T>,
    {
        self.reduce(ETensor::product, dimensions, keepdims)
    }

    pub fn max_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<ETensor<T>>
    where
        T: PartialOrd,
    {
        self.reduce(ETensor::max, dimensions, keepdims)
    }

    pub fn min_dims(&self, dimensions: &[usize], keepdims: bool) -> Result<ETensor<T>>
    where
        T: PartialOrd,
    {
        self.reduce(ETensor::min, dimensions, keepdims)
    }
}
