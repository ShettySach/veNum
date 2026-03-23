use anyhow::Result;
use std::sync::Arc;

use crate::core::lazy::jit::compile_kernel;
use crate::core::lazy::kernel::ExecutableKernel;
use crate::core::lazy::schedule::FusedKernel;
use crate::core::shared::graph::Graph;

/// Backend abstraction for compiling and executing fused kernels.
///
/// `ExecutionPlan` stores kernels behind `ExecutableKernel`, allowing plan
/// replay to target different compilation/execution strategies.
pub trait Backend: Send + Sync {
    /// Compile a fused kernel for this backend.
    fn compile(
        &self,
        graph: &Graph,
        kernel: &FusedKernel,
        capture_ir: bool,
    ) -> Result<Arc<dyn ExecutableKernel>>;
}

/// Default CPU backend using Cranelift.
#[derive(Clone)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn compile(
        &self,
        graph: &Graph,
        kernel: &FusedKernel,
        capture_ir: bool,
    ) -> Result<Arc<dyn ExecutableKernel>> {
        Ok(Arc::new(compile_kernel(graph, kernel, capture_ir)?))
    }
}
