//! Reduce operations for tensors.

use anyhow::Result;

use super::context::Context;
use super::helpers::{compute_reduced_shape, normalize_axes};
use super::structure::Tensor;

impl<C: Context> Tensor<C> {
    /// Reduce sum along specified dimensions.
    ///
    /// Dimensions can be negative (counting from the end).
    ///
    /// # Arguments
    /// * `dims` - Dimensions to reduce over (supports negative indexing)
    /// * `keepdims` - If true, reduced dimensions are kept with size 1
    pub fn sum(&self, dims: &[isize], keepdims: bool) -> Result<Tensor<C>> {
        let dims_usize = normalize_axes(dims, self.shape.len())?;
        let shape = compute_reduced_shape(&self.shape, &dims_usize, keepdims);
        let id = self.with_graph_mut(|g| g.sum(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Reduce product along specified dimensions.
    ///
    /// Dimensions can be negative (counting from the end).
    pub fn prod(&self, dims: &[isize], keepdims: bool) -> Result<Tensor<C>> {
        let dims_usize = normalize_axes(dims, self.shape.len())?;
        let shape = compute_reduced_shape(&self.shape, &dims_usize, keepdims);
        let id = self.with_graph_mut(|g| g.prod(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Reduce max along specified dimensions.
    ///
    /// Dimensions can be negative (counting from the end).
    pub fn max(&self, dims: &[isize], keepdims: bool) -> Result<Tensor<C>> {
        let dims_usize = normalize_axes(dims, self.shape.len())?;
        let shape = compute_reduced_shape(&self.shape, &dims_usize, keepdims);
        let id = self.with_graph_mut(|g| g.max(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Reduce min along specified dimensions.
    ///
    /// Dimensions can be negative (counting from the end).
    pub fn min(&self, dims: &[isize], keepdims: bool) -> Result<Tensor<C>> {
        let dims_usize = normalize_axes(dims, self.shape.len())?;
        let shape = compute_reduced_shape(&self.shape, &dims_usize, keepdims);
        let id = self.with_graph_mut(|g| g.min(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    // ==================== Convenience Methods ====================

    /// Reduce sum over all dimensions.
    pub fn sum_all(&self) -> Result<Tensor<C>> {
        let dims: Vec<isize> = (0..self.shape.len() as isize).collect();
        self.sum(&dims, true)
    }

    /// Reduce product over all dimensions.
    pub fn prod_all(&self) -> Result<Tensor<C>> {
        let dims: Vec<isize> = (0..self.shape.len() as isize).collect();
        self.prod(&dims, true)
    }

    /// Reduce max over all dimensions.
    pub fn max_all(&self) -> Result<Tensor<C>> {
        let dims: Vec<isize> = (0..self.shape.len() as isize).collect();
        self.max(&dims, true)
    }

    /// Reduce min over all dimensions.
    pub fn min_all(&self) -> Result<Tensor<C>> {
        let dims: Vec<isize> = (0..self.shape.len() as isize).collect();
        self.min(&dims, true)
    }
}
