/// A compiled kernel ready to execute.
pub struct CompiledKernel {
    /// Number of input buffer pointers.
    pub num_inputs: usize,
    /// The JIT module that owns the compiled code.
    pub(crate) _module: cranelift_jit::JITModule,
    /// Raw function pointer to the compiled kernel.
    pub(crate) fn_ptr: *const u8,
    /// Cranelift IR (CLIF) text, captured only when requested.
    pub clif_ir: Option<String>,
}

// Safety: The compiled code is immutable once created and the function pointer
// is valid for the lifetime of _module.
unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

impl CompiledKernel {
    /// Execute the kernel with the given input buffer pointers and output pointer.
    ///
    /// # Safety
    /// - `inputs_ptr` must point to at least `self.num_inputs` valid pointers.
    /// - All input pointers must be valid and point to buffers of sufficient size.
    /// - `output` must be valid and point to a buffer of sufficient size.
    pub unsafe fn execute(&self, inputs_ptr: *const *const u8, output: *mut u8, numel: usize) {
        // ABI: fn(inputs: *const *const u8, out: *mut u8, n: u64)
        let f: extern "C" fn(*const *const u8, *mut u8, u64) = std::mem::transmute(self.fn_ptr);
        f(inputs_ptr, output, numel as u64);
    }
}

impl crate::core::liquid::kernel::ExecutableKernel for CompiledKernel {
    fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    fn debug_ir(&self) -> Option<String> {
        self.clif_ir.clone()
    }

    fn execute(&self, inputs: &[*const u8], output: *mut u8, numel: usize) {
        debug_assert_eq!(
            inputs.len(),
            self.num_inputs,
            "Kernel expected {} inputs, got {}",
            self.num_inputs,
            inputs.len()
        );
        // Safety: callers must provide valid pointers; we concentrate the
        // JIT call unsafety inside the CPU kernel implementation.
        unsafe { CompiledKernel::execute(self, inputs.as_ptr(), output, numel) }
    }
}
