use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::graph::Graph;
use super::jit::{CompiledKernel, KernelSignature};

/// Cache of JIT-compiled kernels keyed by structural signature.
pub(crate) type KernelCache = Arc<Mutex<HashMap<KernelSignature, Arc<CompiledKernel>>>>;

/// Shared lazy execution context.
///
/// A `Context` owns a single computation graph so tensors created within the
/// same context naturally share nodes and avoid cross-graph imports.
/// It also maintains a cache of compiled JIT kernels so identical kernel
/// structures are only compiled once.
#[derive(Clone)]
pub struct Context {
    graph: Arc<Mutex<Graph>>,
    kernel_cache: KernelCache,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    /// Create a new lazy context with an empty graph.
    pub fn new() -> Self {
        Self {
            graph: Arc::new(Mutex::new(Graph::new())),
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get a clone of the shared graph handle.
    pub(crate) fn graph(&self) -> Arc<Mutex<Graph>> {
        Arc::clone(&self.graph)
    }

    /// Get a clone of the shared kernel cache handle.
    pub(crate) fn kernel_cache(&self) -> KernelCache {
        Arc::clone(&self.kernel_cache)
    }
}
