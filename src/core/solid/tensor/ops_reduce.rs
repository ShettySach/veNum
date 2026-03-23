//! Reduce operations for Solid tensors.

use anyhow::{bail, Result};

use crate::core::solid::tensor::Tensor;

impl Tensor {
    /// Reduce sum along specified dimensions.
    pub fn sum(&self, dims: &[isize], keepdims: bool) -> Result<Tensor> {
        let shape = compute_reduced_shape(&self.shape, dims, keepdims)?;
        let dims_usize: Vec<usize> = normalize_axes_usize(dims, self.shape.len())?;
        let id = self.with_graph_mut(|g| g.sum(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Reduce product along specified dimensions.
    pub fn prod(&self, dims: &[isize], keepdims: bool) -> Result<Tensor> {
        let shape = compute_reduced_shape(&self.shape, dims, keepdims)?;
        let dims_usize: Vec<usize> = normalize_axes_usize(dims, self.shape.len())?;
        let id = self.with_graph_mut(|g| g.prod(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Reduce max along specified dimensions.
    pub fn max(&self, dims: &[isize], keepdims: bool) -> Result<Tensor> {
        let shape = compute_reduced_shape(&self.shape, dims, keepdims)?;
        let dims_usize: Vec<usize> = normalize_axes_usize(dims, self.shape.len())?;
        let id = self.with_graph_mut(|g| g.max(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Reduce min along specified dimensions.
    pub fn min(&self, dims: &[isize], keepdims: bool) -> Result<Tensor> {
        let shape = compute_reduced_shape(&self.shape, dims, keepdims)?;
        let dims_usize: Vec<usize> = normalize_axes_usize(dims, self.shape.len())?;
        let id = self.with_graph_mut(|g| g.min(self.id, dims_usize, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }
}

// Helper functions (copied from lazy/tensor/ops_reduce.rs)

fn normalize_axes_usize(dims: &[isize], ndim: usize) -> Result<Vec<usize>> {
    let ndim_i = ndim as isize;
    dims.iter()
        .map(|&d| {
            let normalized = if d < 0 { ndim_i + d } else { d };
            if normalized < 0 || normalized >= ndim_i {
                bail!("axis {} out of bounds for tensor of rank {}", d, ndim);
            }
            Ok(normalized as usize)
        })
        .collect()
}

fn compute_reduced_shape(shape: &[usize], dims: &[isize], keepdims: bool) -> Result<Vec<usize>> {
    let ndim = shape.len();
    let dims_usize = normalize_axes_usize(dims, ndim)?;
    let mut new_shape = Vec::with_capacity(ndim);
    for (i, &s) in shape.iter().enumerate() {
        if dims_usize.contains(&i) {
            if keepdims {
                new_shape.push(1);
            }
        } else {
            new_shape.push(s);
        }
    }
    if new_shape.is_empty() {
        new_shape.push(1);
    }
    Ok(new_shape)
}
