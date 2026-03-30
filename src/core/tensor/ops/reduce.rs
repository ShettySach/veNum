//! Reduce operations for tensors.

use anyhow::Result;

use crate::core::tensor::helpers::{compute_reduced_shape, normalize_axes};
use crate::core::tensor::structure::Tensor;

impl Tensor {
    /// Reduce sum along specified dimensions.
    pub fn sum(&self, dims: &[isize], keepdims: bool) -> Result<Tensor> {
        let dims_usize = normalize_axes(dims, self.shape.len())?;
        let shape = compute_reduced_shape(&self.shape, &dims_usize, keepdims);
        let id = self.with_graph_mut(|g| g.sum(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Reduce product along specified dimensions.
    pub fn prod(&self, dims: &[isize], keepdims: bool) -> Result<Tensor> {
        let dims_usize = normalize_axes(dims, self.shape.len())?;
        let shape = compute_reduced_shape(&self.shape, &dims_usize, keepdims);
        let id = self.with_graph_mut(|g| g.prod(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Reduce max along specified dimensions.
    pub fn max(&self, dims: &[isize], keepdims: bool) -> Result<Tensor> {
        let dims_usize = normalize_axes(dims, self.shape.len())?;
        let shape = compute_reduced_shape(&self.shape, &dims_usize, keepdims);
        let id = self.with_graph_mut(|g| g.max(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Reduce min along specified dimensions.
    pub fn min(&self, dims: &[isize], keepdims: bool) -> Result<Tensor> {
        let dims_usize = normalize_axes(dims, self.shape.len())?;
        let shape = compute_reduced_shape(&self.shape, &dims_usize, keepdims);
        let id = self.with_graph_mut(|g| g.min(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Reduce sum over all dimensions.
    pub fn sum_all(&self) -> Result<Tensor> {
        let dims: Vec<isize> = (0..self.shape.len() as isize).collect();
        self.sum(&dims, true)
    }

    /// Reduce product over all dimensions.
    pub fn prod_all(&self) -> Result<Tensor> {
        let dims: Vec<isize> = (0..self.shape.len() as isize).collect();
        self.prod(&dims, true)
    }

    /// Reduce max over all dimensions.
    pub fn max_all(&self) -> Result<Tensor> {
        let dims: Vec<isize> = (0..self.shape.len() as isize).collect();
        self.max(&dims, true)
    }

    /// Reduce min over all dimensions.
    pub fn min_all(&self) -> Result<Tensor> {
        let dims: Vec<isize> = (0..self.shape.len() as isize).collect();
        self.min(&dims, true)
    }
}
