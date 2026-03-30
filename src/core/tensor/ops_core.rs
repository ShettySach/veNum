//! Core tensor operations: binary_op and unary_op.

use anyhow::{Result, bail};

use crate::core::graph::Op;
use crate::core::shape_tracker::broadcast;

use super::helpers::unsqueeze_shape;
use super::structure::Tensor;

impl Tensor {
    /// Apply a binary operation with broadcasting.
    pub(crate) fn binary_op(&self, rhs: &Tensor, op: Op) -> Result<Tensor> {
        if !self.cx.same_graph(&rhs.cx) {
            bail!("binary_op requires tensors from the same context");
        }
        if self.dtype != rhs.dtype {
            bail!(
                "binary_op requires matching dtypes: {:?} vs {:?}",
                self.dtype,
                rhs.dtype
            );
        }

        // Compute the broadcast shape
        let broadcast_shape = broadcast(&self.shape, &rhs.shape)?;

        // Broadcast lhs if needed
        let lhs_id = if self.shape == broadcast_shape {
            self.id
        } else {
            self.with_graph_mut(|g| {
                let unsqueezed = g.unsqueeze(
                    self.id,
                    broadcast_shape.len(),
                    unsqueeze_shape(&self.shape, broadcast_shape.len()),
                );
                g.expand(unsqueezed, broadcast_shape.clone())
            })
        };

        // Broadcast rhs if needed
        let rhs_id = if rhs.shape == broadcast_shape {
            rhs.id
        } else {
            self.with_graph_mut(|g| {
                let unsqueezed = g.unsqueeze(
                    rhs.id,
                    broadcast_shape.len(),
                    unsqueeze_shape(&rhs.shape, broadcast_shape.len()),
                );
                g.expand(unsqueezed, broadcast_shape.clone())
            })
        };

        let id = self.with_graph_mut(|g| g.binary(op, lhs_id, rhs_id, broadcast_shape.clone()));
        Ok(self.derived(id, broadcast_shape))
    }

    /// Apply a unary operation.
    pub(crate) fn unary_op(&self, op: Op) -> Tensor {
        let id = self.with_graph_mut(|g| g.unary(op, self.id));
        self.derived(id, self.shape.clone())
    }
}
