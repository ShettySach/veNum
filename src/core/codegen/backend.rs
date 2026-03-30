//! Backend trait definition.

use anyhow::Result;
use std::sync::Arc;

use crate::core::graph::Graph;
use crate::core::kernel::ExecutableKernel;
use crate::core::schedule::FusedKernel;

/// Backend trait: compiles fused kernels for AOT programs.
///
/// Compiles all kernels needed by a program and packages
/// them for repeated execution.
pub(crate) trait Backend: Send + Sync {
    /// Compile a fused kernel to executable form.
    fn compile_kernel(
        &self,
        graph: &Graph,
        kernel: &FusedKernel,
        capture_ir: bool,
    ) -> Result<Arc<dyn ExecutableKernel>>;
}
