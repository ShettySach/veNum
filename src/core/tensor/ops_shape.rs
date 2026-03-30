//! Shape operations for tensors.

use anyhow::{bail, Result};

use crate::core::dtype::Scalar;

use super::structure::Tensor;

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

    /// Permute (reorder) the dimensions.
    pub fn permute(&self, axes: Vec<usize>) -> Result<Tensor> {
        if axes.len() != self.shape.len() {
            bail!(
                "permute: axes length {} must match tensor rank {}",
                axes.len(),
                self.shape.len()
            );
        }

        let mut seen = vec![false; axes.len()];
        for &p in &axes {
            if p >= axes.len() || seen[p] {
                bail!(
                    "permute: invalid permutation {:?} for rank {}",
                    axes,
                    self.shape.len()
                );
            }
            seen[p] = true;
        }

        let new_shape: Vec<usize> = axes.iter().map(|&i| self.shape[i]).collect();
        let id = self.with_graph_mut(|g| g.permute(self.id, axes, new_shape.clone()));
        Ok(self.derived(id, new_shape))
    }

    /// Transpose two dimensions.
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if rank < 2 || dim1 >= rank || dim2 >= rank {
            bail!(
                "transpose: dimensions {} and {} out of bounds for rank {}",
                dim1,
                dim2,
                rank
            );
        }

        let mut new_shape = self.shape.clone();
        new_shape.swap(dim1, dim2);

        let id = self.with_graph_mut(|g| g.transpose(self.id, dim1, dim2, new_shape.clone()));
        Ok(self.derived(id, new_shape))
    }

    /// Expand dimensions by broadcasting.
    pub fn expand(&self, new_shape: Vec<usize>) -> Result<Tensor> {
        if new_shape.len() != self.shape.len() {
            bail!(
                "expand: shape length {} must match tensor rank {}",
                new_shape.len(),
                self.shape.len()
            );
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

    /// Slice the tensor along each dimension.
    pub fn slice(&self, ranges: Vec<(usize, usize)>) -> Result<Tensor> {
        if ranges.len() != self.shape.len() {
            bail!(
                "slice: ranges length {} must match tensor rank {}",
                ranges.len(),
                self.shape.len()
            );
        }

        let mut new_shape = Vec::with_capacity(self.shape.len());
        for (dim, &(start, end_raw)) in ranges.iter().enumerate() {
            let size = self.shape[dim];
            let end = if end_raw == 0 { size } else { end_raw };
            if start > end || end > size {
                bail!(
                    "slice: range {:?} out of bounds for dim {} with size {}",
                    (start, end),
                    dim,
                    size
                );
            }
            new_shape.push(end - start);
        }

        let id = self.with_graph_mut(|g| g.slice(self.id, ranges, new_shape.clone()));
        Ok(self.derived(id, new_shape))
    }

    /// Flip the tensor along specified dimensions.
    pub fn flip(&self, dims: Vec<usize>) -> Result<Tensor> {
        for &d in &dims {
            if d >= self.shape.len() {
                bail!(
                    "flip: dimension {} out of bounds for rank {}",
                    d,
                    self.shape.len()
                );
            }
        }

        let shape = self.shape.clone();
        let id = self.with_graph_mut(|g| g.flip(self.id, dims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    /// Remove all dimensions of size 1.
    pub fn squeeze(&self) -> Result<Tensor> {
        let mut new_shape: Vec<usize> = self.shape.iter().copied().filter(|&s| s != 1).collect();
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let id = self.with_graph_mut(|g| g.squeeze(self.id, new_shape.clone()));
        Ok(self.derived(id, new_shape))
    }

    /// Add leading dimensions of size 1 to reach a target rank.
    pub fn unsqueeze(&self, new_rank: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if new_rank < rank {
            bail!(
                "unsqueeze: new rank {} must be >= current rank {}",
                new_rank,
                rank
            );
        }
        if new_rank == rank {
            return Ok(self.clone());
        }

        let mut new_shape = vec![1; new_rank - rank];
        new_shape.extend_from_slice(&self.shape);

        let id = self.with_graph_mut(|g| g.unsqueeze(self.id, new_rank, new_shape.clone()));
        Ok(self.derived(id, new_shape))
    }

    /// Pad the tensor with a constant value.
    pub fn pad(&self, constant: f32, padding: Vec<(usize, usize)>) -> Result<Tensor> {
        self.pad_scalar(Scalar::F32(constant), padding)
    }

    /// Pad the tensor with a scalar value.
    pub fn pad_scalar(&self, constant: Scalar, padding: Vec<(usize, usize)>) -> Result<Tensor> {
        let mut pad = padding;
        pad.resize(self.shape.len(), (0, 0));

        let mut new_shape = Vec::with_capacity(self.shape.len());
        for (s, (l, r)) in self.shape.iter().copied().zip(pad.iter().copied()) {
            new_shape.push(l + s + r);
        }

        let id = self.with_graph_mut(|g| g.pad(self.id, constant, pad, new_shape.clone()));
        Ok(self.derived(id, new_shape))
    }
}
