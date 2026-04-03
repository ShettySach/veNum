//! Core tensor operations: binary_op and unary_op.

use anyhow::{anyhow, bail, Result};

use crate::core::hlir::decompose;
use crate::core::hlir::{Dim, NodeId, Op};

use crate::core::tensor::helpers::unsqueeze_shape;
use crate::core::tensor::structure::Tensor;

impl Tensor {
    fn broadcast_shape(lhs: &[Dim], rhs: &[Dim]) -> Result<Vec<Dim>> {
        let rank = lhs.len().max(rhs.len());
        let mut out = vec![Dim::constant(1); rank];

        for i in 0..rank {
            let l = if i < rank - lhs.len() {
                1
            } else {
                lhs[i - (rank - lhs.len())]
                    .as_const()
                    .ok_or_else(|| anyhow!("broadcast currently requires constant dims"))?
            };
            let r = if i < rank - rhs.len() {
                1
            } else {
                rhs[i - (rank - rhs.len())]
                    .as_const()
                    .ok_or_else(|| anyhow!("broadcast currently requires constant dims"))?
            };
            if l != r && l != 1 && r != 1 {
                bail!("cannot broadcast {:?} and {:?}", lhs, rhs);
            }
            out[i] = Dim::constant(l.max(r));
        }

        Ok(out)
    }

    /// Broadcast a tensor's node to match the target shape, inserting
    /// reshape+expand ops as needed. Returns the (possibly new) node id.
    fn broadcast_to_shape(
        &self,
        node_id: NodeId,
        node_shape: &[Dim],
        target: &[Dim],
    ) -> NodeId {
        if node_shape == target {
            return node_id;
        }
        self.with_graph_mut(|g| {
            let unsqueezed = g.reshape(node_id, unsqueeze_shape(node_shape, target.len()));
            g.expand(unsqueezed, target.to_vec())
        })
    }

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

        let broadcast_shape = Self::broadcast_shape(&self.shape, &rhs.shape)?;
        let lhs_id = self.broadcast_to_shape(self.id, &self.shape, &broadcast_shape);
        let rhs_id = self.broadcast_to_shape(rhs.id, &rhs.shape, &broadcast_shape);

        let id = self.with_graph_mut(|g| match op {
            Op::Add(_, _) => g.binary(lhs_id, rhs_id, Op::Add),
            Op::Mul(_, _) => g.binary(lhs_id, rhs_id, Op::Mul),
            Op::Max(_, _) => g.binary(lhs_id, rhs_id, Op::Max),
            Op::Min(_, _) => g.binary(lhs_id, rhs_id, Op::Min),
            _ => unreachable!("binary_op only supports binary primitives"),
        });
        Ok(self.derived(id, broadcast_shape))
    }

    pub(crate) fn sub_decomposed(&self, rhs: &Tensor) -> Result<Tensor> {
        if !self.cx.same_graph(&rhs.cx) {
            bail!("sub requires tensors from the same context");
        }
        if self.dtype != rhs.dtype {
            bail!(
                "sub requires matching dtypes: {:?} vs {:?}",
                self.dtype,
                rhs.dtype
            );
        }
        let broadcast_shape = Self::broadcast_shape(&self.shape, &rhs.shape)?;
        let lhs_id = self.broadcast_to_shape(self.id, &self.shape, &broadcast_shape);
        let rhs_id = self.broadcast_to_shape(rhs.id, &rhs.shape, &broadcast_shape);

        let id = self.with_graph_mut(|g| decompose::sub(g, lhs_id, rhs_id));
        Ok(self.derived(id, broadcast_shape))
    }

    pub(crate) fn div_decomposed(&self, rhs: &Tensor) -> Result<Tensor> {
        if !self.cx.same_graph(&rhs.cx) {
            bail!("div requires tensors from the same context");
        }
        if self.dtype != rhs.dtype {
            bail!(
                "div requires matching dtypes: {:?} vs {:?}",
                self.dtype,
                rhs.dtype
            );
        }
        let broadcast_shape = Self::broadcast_shape(&self.shape, &rhs.shape)?;
        let lhs_id = self.broadcast_to_shape(self.id, &self.shape, &broadcast_shape);
        let rhs_id = self.broadcast_to_shape(rhs.id, &rhs.shape, &broadcast_shape);

        let id = self.with_graph_mut(|g| decompose::div(g, lhs_id, rhs_id));
        Ok(self.derived(id, broadcast_shape))
    }

    /// Apply a unary operation.
    pub(crate) fn unary_op(&self, op: Op) -> Tensor {
        let id = self.with_graph_mut(|g| match op {
            Op::Neg(_) => g.unary(self.id, Op::Neg),
            Op::Recip(_) => g.unary(self.id, Op::Recip),
            Op::Exp(_) => g.unary(self.id, Op::Exp),
            Op::Log(_) => g.unary(self.id, Op::Log),
            Op::Sqrt(_) => g.unary(self.id, Op::Sqrt),
            Op::Sin(_) => g.unary(self.id, Op::Sin),
            _ => unreachable!("unary_op only supports unary primitives"),
        });
        self.derived(id, self.shape.clone())
    }
}
