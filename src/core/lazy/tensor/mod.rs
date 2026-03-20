use anyhow::{bail, Result};
use std::sync::Arc;

use super::context::{KernelCache, PlanCache, SharedBufferPool};
use super::dtype::DType;
use super::graph::{Graph, NodeId, Op};

mod accessors;
mod constructors;
mod helpers;
mod ops_elementwise;
mod ops_matmul;
mod ops_reduce;
mod ops_shape;
mod overloads;
mod realize;
mod visualize;

#[derive(Clone)]
pub struct Tensor {
    pub(super) graph: Arc<std::sync::Mutex<Graph>>,
    pub(super) kernel_cache: KernelCache,
    pub(super) plan_cache: PlanCache,
    pub(super) buffer_pool: SharedBufferPool,
    pub(super) id: NodeId,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: DType,
}

impl Tensor {
    pub(super) fn with_graph_mut<R>(&self, f: impl FnOnce(&mut Graph) -> R) -> R {
        f(&mut self.graph.lock().unwrap())
    }

    pub(super) fn derived(&self, id: NodeId, shape: Vec<usize>) -> Self {
        Self {
            graph: Arc::clone(&self.graph),
            kernel_cache: Arc::clone(&self.kernel_cache),
            plan_cache: Arc::clone(&self.plan_cache),
            buffer_pool: Arc::clone(&self.buffer_pool),
            id,
            shape,
            dtype: self.dtype,
        }
    }

    pub(super) fn binary_op(&self, rhs: &Tensor, op: Op) -> Result<Tensor> {
        if !Arc::ptr_eq(&self.graph, &rhs.graph) {
            bail!("binary_op requires both tensors to share the same Context");
        }
        if self.dtype != rhs.dtype {
            bail!(
                "binary_op requires matching dtypes, got {:?} and {:?}",
                self.dtype,
                rhs.dtype
            );
        }
        let id = self.with_graph_mut(|g| g.binary(op, self.id, rhs.id));
        Ok(self.derived(id, self.shape.clone()))
    }

    pub(super) fn unary_op(&self, op: Op) -> Tensor {
        let id = self.with_graph_mut(|g| g.unary(op, self.id));
        self.derived(id, self.shape.clone())
    }
}
