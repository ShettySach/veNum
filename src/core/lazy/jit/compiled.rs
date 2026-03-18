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
    /// - All pointers must be valid and point to buffers of sufficient size.
    /// - `inputs` must have exactly `self.num_inputs` elements.
    pub unsafe fn execute(&self, inputs: &[*const u8], output: *mut u8, numel: usize) {
        // ABI: fn(in0: *const u8, in1: *const u8, ..., out: *mut u8, n: u64)
        // All pointer types have the same ABI representation.
        match self.num_inputs {
            0 => {
                let f: extern "C" fn(*mut u8, u64) = std::mem::transmute(self.fn_ptr);
                f(output, numel as u64);
            }
            1 => {
                let f: extern "C" fn(*const u8, *mut u8, u64) = std::mem::transmute(self.fn_ptr);
                f(inputs[0], output, numel as u64);
            }
            2 => {
                let f: extern "C" fn(*const u8, *const u8, *mut u8, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(inputs[0], inputs[1], output, numel as u64);
            }
            3 => {
                let f: extern "C" fn(*const u8, *const u8, *const u8, *mut u8, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(inputs[0], inputs[1], inputs[2], output, numel as u64);
            }
            4 => {
                let f: extern "C" fn(*const u8, *const u8, *const u8, *const u8, *mut u8, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    output,
                    numel as u64,
                );
            }
            5 => {
                let f: extern "C" fn(
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *mut u8,
                    u64,
                ) = std::mem::transmute(self.fn_ptr);
                f(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    output,
                    numel as u64,
                );
            }
            6 => {
                let f: extern "C" fn(
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *const u8,
                    *mut u8,
                    u64,
                ) = std::mem::transmute(self.fn_ptr);
                f(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    output,
                    numel as u64,
                );
            }
            _ => {
                panic!(
                    "CompiledKernel: too many inputs ({}), max 6 supported in dispatch",
                    self.num_inputs
                );
            }
        }
    }
}
