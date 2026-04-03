//! Shape operations for tensors.

use anyhow::{Result, anyhow, bail};

use crate::core::hlir::{Dim, Range};

use crate::core::tensor::structure::Tensor;

impl Tensor {
    /// Reshape the tensor to a new shape.
    pub fn reshape(&self, new_shape: Vec<i64>) -> Result<Tensor> {
        let new_shape: Vec<Dim> = new_shape.into_iter().map(Dim::constant).collect();
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

        let new_shape: Vec<Dim> = axes.iter().map(|&i| self.shape[i].clone()).collect();
        let id = self.with_graph_mut(|g| g.permute(self.id, axes));
        Ok(self.derived(id, new_shape))
    }

    /// Expand dimensions by broadcasting.
    pub fn expand(&self, new_shape: Vec<i64>) -> Result<Tensor> {
        let new_shape: Vec<Dim> = new_shape.into_iter().map(Dim::constant).collect();
        if new_shape.len() != self.shape.len() {
            bail!(
                "expand: shape length {} must match tensor rank {}",
                new_shape.len(),
                self.shape.len()
            );
        }

        for (i, (old, new)) in self.shape.iter().zip(&new_shape).enumerate() {
            let old = old
                .as_const()
                .ok_or_else(|| anyhow!("expand currently requires constant dims"))?;
            let new = new
                .as_const()
                .ok_or_else(|| anyhow!("expand currently requires constant dims"))?;
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
    pub fn slice(&self, ranges: Vec<(i64, i64)>) -> Result<Tensor> {
        if ranges.len() != self.shape.len() {
            bail!(
                "slice: ranges length {} must match tensor rank {}",
                ranges.len(),
                self.shape.len()
            );
        }

        let mut new_shape = Vec::with_capacity(self.shape.len());
        let mut ir_ranges = Vec::with_capacity(self.shape.len());
        for (dim, &(start, end_raw)) in ranges.iter().enumerate() {
            let size = self.shape[dim]
                .as_const()
                .ok_or_else(|| anyhow!("slice currently requires constant dims"))?;
            let end = if end_raw == 0 { size } else { end_raw };
            if start > end || end > size {
                bail!(
                    "slice: range {:?} out of bounds for dim {} with size {}",
                    (start, end),
                    dim,
                    size
                );
            }
            new_shape.push(Dim::constant(end - start));
            ir_ranges.push(Range {
                start: Dim::constant(start),
                end: Dim::constant(end),
            });
        }

        let id = self.with_graph_mut(|g| g.slice(self.id, ir_ranges));
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

        let mut new_shape = vec![Dim::constant(1); new_rank - rank];
        new_shape.extend_from_slice(&self.shape);

        let id = self.with_graph_mut(|g| g.reshape(self.id, new_shape.clone()));
        Ok(self.derived(id, new_shape))
    }

    /// Concatenate tensors along a dimension.
    pub fn concat(&self, other: &Tensor, axis: usize) -> Result<Tensor> {
        if !self.cx.same_graph(&other.cx) {
            bail!("concat requires tensors from the same context");
        }
        if self.shape.len() != other.shape.len() {
            bail!(
                "concat requires same rank: {} vs {}",
                self.shape.len(),
                other.shape.len()
            );
        }
        for (i, (a, b)) in self.shape.iter().zip(&other.shape).enumerate() {
            if i != axis && a != b {
                bail!(
                    "concat dimension mismatch at axis {}: {:?} vs {:?}",
                    axis,
                    a,
                    b
                );
            }
        }

        let inputs = vec![self.id, other.id];
        let new_shape = self.shape.clone();
        let new_shape_axis = self.shape[axis].clone() + other.shape[axis].clone();
        let mut out_shape = new_shape.clone();
        out_shape[axis] = new_shape_axis;

        let id = self.with_graph_mut(|g| g.concat(inputs, axis));
        Ok(self.derived(id, out_shape))
    }
}
