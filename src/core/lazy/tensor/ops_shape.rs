use anyhow::{bail, Result};

use crate::core::errors::{ExpansionError, ReshapeError, TransposeError, UnsqueezeError};
use crate::core::lazy::dtype::Scalar;
use crate::core::lazy::tensor::Tensor;

impl Tensor {
    pub fn reshape(&self, sizes: Vec<usize>) -> Result<Tensor> {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = sizes.iter().product();
        if old_numel != new_numel {
            bail!(ReshapeError {
                current_shape: self.shape.clone(),
                new_shape: sizes
            });
        }

        let id = self.with_graph_mut(|g| g.reshape(self.id, sizes.clone()));
        Ok(self.derived(id, sizes))
    }

    pub fn permute(&self, permutation: Vec<usize>) -> Result<Tensor> {
        if permutation.len() != self.shape.len() {
            bail!(ReshapeError {
                current_shape: self.shape.clone(),
                new_shape: self.shape.clone()
            });
        }

        let mut seen = vec![false; permutation.len()];
        for &p in &permutation {
            if p >= permutation.len() || seen[p] {
                bail!(ReshapeError {
                    current_shape: self.shape.clone(),
                    new_shape: self.shape.clone()
                });
            }
            seen[p] = true;
        }

        let shape: Vec<_> = permutation.iter().map(|&i| self.shape[i]).collect();
        let id = self.with_graph_mut(|g| g.permute(self.id, permutation, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if rank < 2 || dim_1 >= rank || dim_2 >= rank {
            bail!(TransposeError);
        }

        let mut shape = self.shape.clone();
        shape.swap(dim_1, dim_2);

        let id = self.with_graph_mut(|g| g.transpose(self.id, dim_1, dim_2, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn expand(&self, expansions: Vec<usize>) -> Result<Tensor> {
        if expansions.len() != self.shape.len() {
            bail!(ExpansionError {
                size: self.shape.len(),
                expansion: expansions.len()
            });
        }

        for (&size, &exp) in self.shape.iter().zip(expansions.iter()) {
            if !(exp == size || size == 1) {
                bail!(ExpansionError {
                    size,
                    expansion: exp
                });
            }
        }

        let id = self.with_graph_mut(|g| g.expand(self.id, expansions.clone()));
        Ok(self.derived(id, expansions))
    }

    pub fn slice(&self, ranges: Vec<(usize, usize)>) -> Result<Tensor> {
        if ranges.len() != self.shape.len() {
            bail!(
                "slice ranges rank mismatch: got {}, expected {}",
                ranges.len(),
                self.shape.len()
            );
        }

        let mut out = Vec::with_capacity(self.shape.len());
        for (dim, &(start, end_raw)) in ranges.iter().enumerate() {
            let size = self.shape[dim];
            let end = if end_raw == 0 { size } else { end_raw };
            if start > end || end > size {
                bail!(
                    "slice range {:?} out of bounds for dim {} with size {}",
                    (start, end),
                    dim,
                    size
                );
            }
            out.push(end - start);
        }

        let id = self.with_graph_mut(|g| g.slice(self.id, ranges, out.clone()));
        Ok(self.derived(id, out))
    }

    pub fn flip(&self, flips: Vec<usize>) -> Result<Tensor> {
        for &d in &flips {
            if d >= self.shape.len() {
                bail!(
                    "flip dimension {} out of bounds for rank {}",
                    d,
                    self.shape.len()
                );
            }
        }

        let shape = self.shape.clone();
        let id = self.with_graph_mut(|g| g.flip(self.id, flips, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn squeeze(&self) -> Result<Tensor> {
        let mut shape: Vec<usize> = self.shape.iter().copied().filter(|&s| s != 1).collect();
        if shape.is_empty() {
            shape.push(1);
        }

        let id = self.with_graph_mut(|g| g.squeeze(self.id, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn unsqueeze(&self, new_rank: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if new_rank < rank {
            bail!(UnsqueezeError {
                current: rank,
                new_rank
            });
        }
        if new_rank == rank {
            return Ok(self.clone());
        }

        let mut shape = vec![1; new_rank - rank];
        shape.extend_from_slice(&self.shape);

        let id = self.with_graph_mut(|g| g.unsqueeze(self.id, new_rank, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn pad(&self, constant: f32, padding: Vec<(usize, usize)>) -> Result<Tensor> {
        self.pad_scalar(Scalar::F32(constant), padding)
    }

    pub fn pad_scalar(&self, constant: Scalar, padding: Vec<(usize, usize)>) -> Result<Tensor> {
        let mut pad = padding;
        pad.resize(self.shape.len(), (0, 0));

        let mut shape = Vec::with_capacity(self.shape.len());
        for (s, (l, r)) in self.shape.iter().copied().zip(pad.iter().copied()) {
            shape.push(l + s + r);
        }

        let id = self.with_graph_mut(|g| g.pad(self.id, constant, pad, shape.clone()));
        Ok(self.derived(id, shape))
    }
}
