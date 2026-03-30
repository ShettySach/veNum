//! Code generator trait and generated kernel representation.

use anyhow::Result;
use cranelift_codegen::ir::Function;
use cranelift_jit::JITModule;

use crate::core::graph::Graph;

pub use crate::core::schedule::FusedKernel;

/// Intermediate representation of a compiled kernel before finalization.
///
/// Output of `CodeGenerator::generate_kernel`. Finalized into executable
/// kernels and linked into a compiled program.
pub struct GeneratedKernel {
    /// Cranelift function (not yet compiled to machine code).
    pub function: Function,
    /// Number of input buffer pointers.
    pub num_inputs: usize,
    /// Debug IR if requested.
    pub debug_ir: Option<String>,
}

/// Core code generation trait.
///
/// Defines the ability to generate Cranelift IR for a fused kernel.
/// Finalization (compilation to machine code) is backend-specific.
///
/// IMPORTANT: The module must be set up with math symbols registered before
/// calling `generate_kernel`. Use `register_math_symbols` from the math module.
pub trait CodeGenerator: Send + Sync {
    /// Generate Cranelift IR for a single fused kernel.
    ///
    /// # Arguments
    /// * `module` - The JIT module to declare functions in (must outlive the returned Function)
    /// * `graph` - The computation graph
    /// * `kernel` - The fused kernel to generate code for
    /// * `capture_ir` - Whether to capture the Cranelift IR text for debugging
    ///
    /// # Returns
    /// A `GeneratedKernel` containing the Cranelift function and debug info.
    ///
    /// # Note
    /// The returned `Function` contains references (`FuncRef`) to functions declared
    /// in the provided `module`. The module must remain alive for as long as the
    /// Function is used.
    fn generate_kernel(
        &self,
        module: &mut JITModule,
        graph: &Graph,
        kernel: &FusedKernel,
        capture_ir: bool,
    ) -> Result<GeneratedKernel>;
}
