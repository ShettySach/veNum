use std::sync::{Arc, Mutex};

use crate::core::lazy::backend::{Backend, CpuBackend};
use crate::core::lazy::graph::Graph;
use crate::core::lazy::jit::KernelSignature;
use crate::core::lazy::kernel::ExecutableKernel;
use crate::core::lazy::lru_cache::LruCache;
use crate::core::lazy::plan::{BufferPool, ExecutionPlan, GraphSignature};

/// Cache of JIT-compiled kernels keyed by structural signature.
pub(crate) type KernelCache = Arc<Mutex<LruCache<KernelSignature, Arc<dyn ExecutableKernel>>>>;

/// Cache of execution plans keyed by graph signature.
pub(crate) type PlanCache = Arc<Mutex<LruCache<GraphSignature, Arc<ExecutionPlan>>>>;

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
    const DEFAULT_KERNEL_CACHE_CAPACITY: usize = 1000;
    const DEFAULT_PLAN_CACHE_CAPACITY: usize = 500;

    /// Create a new lazy context with an empty graph.
    pub fn new() -> Self {
        Self::with_cache_sizes(
            Self::DEFAULT_KERNEL_CACHE_CAPACITY,
            Self::DEFAULT_PLAN_CACHE_CAPACITY,
        )
    }

    pub fn with_cache_sizes(kernel_cache_capacity: usize, plan_cache_capacity: usize) -> Self {
        Self {
            graph: Arc::new(Mutex::new(Graph::new())),
            kernel_cache: Arc::new(Mutex::new(LruCache::new(kernel_cache_capacity))),
            plan_cache: Arc::new(Mutex::new(LruCache::new(plan_cache_capacity))),
            buffer_pool: Arc::new(Mutex::new(BufferPool::new())),
            backend: Arc::new(CpuBackend),
        }
    }

    /// Create a new lazy context with a specific backend.
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self {
            graph: Arc::new(Mutex::new(Graph::new())),
            kernel_cache: Arc::new(Mutex::new(LruCache::new(
                Self::DEFAULT_KERNEL_CACHE_CAPACITY,
            ))),
            plan_cache: Arc::new(Mutex::new(LruCache::new(Self::DEFAULT_PLAN_CACHE_CAPACITY))),
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
