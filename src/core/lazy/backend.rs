use anyhow::Result;
use std::sync::Arc;

use super::graph::Graph;
use super::jit::compile_kernel;
use super::kernel::ExecutableKernel;
use super::schedule::FusedKernel;

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

    /// Execute a compiled kernel.
    fn execute(
        &self,
        kernel: &dyn ExecutableKernel,
        inputs: &[*const u8],
        output: *mut u8,
        numel: usize,
    ) {
        kernel.execute(inputs, output, numel)
    }
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
