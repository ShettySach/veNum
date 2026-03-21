use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::backend::{Backend, CpuBackend};
use super::graph::Graph;
use super::jit::KernelSignature;
use super::kernel::ExecutableKernel;
use super::plan::{BufferPool, ExecutionPlan, GraphSignature};

/// Cache of JIT-compiled kernels keyed by structural signature.
pub(crate) type KernelCache = Arc<Mutex<HashMap<KernelSignature, Arc<dyn ExecutableKernel>>>>;

/// Cache of execution plans keyed by graph signature.
pub(crate) type PlanCache = Arc<Mutex<HashMap<GraphSignature, Arc<ExecutionPlan>>>>;

/// Shared pool of reusable byte buffers.
pub(crate) type SharedBufferPool = Arc<Mutex<BufferPool>>;

/// Shared lazy execution context.
///
/// A `Context` owns a single computation graph so tensors created within the
/// same context naturally share nodes and avoid cross-graph imports.
/// It also maintains a cache of compiled JIT kernels so identical kernel
/// structures are only compiled once, and a cache of execution plans so
/// repeated `realize()` calls skip schedule building and kernel compilation.
#[derive(Clone)]
pub struct Context {
    graph: Arc<Mutex<Graph>>,
    kernel_cache: KernelCache,
    plan_cache: PlanCache,
    buffer_pool: SharedBufferPool,
    backend: Arc<dyn Backend>,
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
            plan_cache: Arc::new(Mutex::new(HashMap::new())),
            buffer_pool: Arc::new(Mutex::new(BufferPool::new())),
            backend: Arc::new(CpuBackend::default()),
        }
    }

    /// Create a new lazy context with a specific backend.
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self {
            graph: Arc::new(Mutex::new(Graph::new())),
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
            plan_cache: Arc::new(Mutex::new(HashMap::new())),
            buffer_pool: Arc::new(Mutex::new(BufferPool::new())),
            backend,
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

    /// Get a clone of the shared plan cache handle.
    pub(crate) fn plan_cache(&self) -> PlanCache {
        Arc::clone(&self.plan_cache)
    }

    /// Get a clone of the shared buffer pool handle.
    pub(crate) fn buffer_pool(&self) -> SharedBufferPool {
        Arc::clone(&self.buffer_pool)
    }

    pub(crate) fn backend(&self) -> Arc<dyn Backend> {
        Arc::clone(&self.backend)
    }
}
