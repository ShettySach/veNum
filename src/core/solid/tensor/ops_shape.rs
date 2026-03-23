//! Shape operations for Solid tensors.

use anyhow::{bail, Result};

use crate::core::solid::tensor::Tensor;

impl Tensor {
    /// Reshape the tensor to a new shape.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            bail!(
                "reshape: new shape {:?} has {} elements, expected {}",
                new_shape,
                new_numel,
                self.numel()
            );
        }
        let id = self.with_graph_mut(|g| g.reshape(self.id, new_shape.clone()));
        Ok(self.derived(id, new_shape))
    }

    /// Permute (transpose) the dimensions.
    pub fn permute(&self, axes: Vec<usize>) -> Result<Tensor> {
        if axes.len() != self.shape.len() {
            bail!("permute: axes length must match tensor rank");
        }
        let new_shape: Vec<usize> = axes.iter().map(|&i| self.shape[i]).collect();
        let id = self.with_graph_mut(|g| g.permute(self.id, axes, new_shape.clone()));
        Ok(self.derived(id, new_shape))
    }

    /// Expand dimensions by broadcasting.
    pub fn expand(&self, new_shape: Vec<usize>) -> Result<Tensor> {
        if new_shape.len() != self.shape.len() {
            bail!("expand: shape length must match");
        }
        for (i, (&old, &new)) in self.shape.iter().zip(&new_shape).enumerate() {
            if old != 1 && old != new {
                bail!(
                    "expand: dimension {} cannot be broadcast from {} to {}",
                    i,
                    old,
                    new
                );
            }
        }
        let id = self.with_graph_mut(|g| g.expand(self.id, new_shape.clone()));
        Ok(self.derived(id, new_shape))
    }
}
