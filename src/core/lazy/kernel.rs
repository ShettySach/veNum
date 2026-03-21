/// A backend-agnostic executable kernel.
///
/// Backends may compile kernels to machine code (CPU JIT), device code (GPU),
/// or any other representation. `ExecutionPlan` stores kernels behind this trait.
pub trait ExecutableKernel: Send + Sync {
    /// Number of input buffer pointers required by this kernel.
    fn num_inputs(&self) -> usize;

    /// Optional backend-specific debug IR/disassembly.
    fn debug_ir(&self) -> Option<String> {
        None
    }

    /// Execute the kernel.
    ///
    /// # Safety
    /// Inputs and output must point to valid buffers of sufficient size.
    unsafe fn execute(&self, inputs: &[*const u8], output: *mut u8, numel: usize);
}
