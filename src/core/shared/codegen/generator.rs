//! Code generator trait and generated kernel representation.

use anyhow::Result;
use cranelift_codegen::ir::Function;
use cranelift_jit::JITModule;

use crate::core::shared::graph::Graph;

/// Forward declaration: FusedKernel is currently in lazy/schedule but will
/// eventually be moved to shared/schedule in a future phase.
/// For now we re-export from lazy to make it accessible.
pub use crate::core::liquid::schedule::FusedKernel;

/// Intermediate representation of a compiled kernel before finalization.
///
/// This is the output of the `CodeGenerator::generate_kernel` method.
/// Liquid backends finalize this to an executable kernel (JIT).
/// Solid backends collect multiple and link them into a program (AOT).
pub struct GeneratedKernel {
    /// Cranelift function (not yet compiled to machine code).
    pub function: Function,
    /// Number of input buffer pointers.
    pub num_inputs: usize,
    /// Debug IR if requested.
    pub debug_ir: Option<String>,
}

/// Core code generation capability shared by Liquid and Solid backends.
///
/// This trait defines the ability to generate Cranelift IR for a fused kernel.
/// It does NOT include finalization (compilation to machine code) - that's
/// backend-specific and defined in `LiquidBackend` and `SolidBackend`.
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
