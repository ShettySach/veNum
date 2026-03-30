//! Solid backend trait and CPU implementation.

use anyhow::Result;
use std::sync::Arc;

use crate::core::codegen::cranelift_setup::create_native_isa;
use crate::core::codegen::{CodeGenerator, CpuCodeGenerator};
use crate::core::graph::Graph;
use crate::core::kernel::ExecutableKernel;
use crate::core::schedule::FusedKernel;

use cranelift_jit::JITModule;
use cranelift_module::Module;

/// Backend trait: compiles fused kernels for AOT programs.
///
/// Compiles all kernels needed by a program and packages
/// them for repeated execution.
pub trait SolidBackend: Send + Sync {
    /// Compile a fused kernel to executable form.
    fn compile_kernel(
        &self,
        graph: &Graph,
        kernel: &FusedKernel,
        capture_ir: bool,
    ) -> Result<Arc<dyn ExecutableKernel>>;
}

/// CPU Solid backend using Cranelift JIT.
///
/// Uses the shared `CpuCodeGenerator` for IR generation, then finalizes
/// each kernel into a standalone JIT-compiled function.
pub struct CpuSolidBackend {
    generator: CpuCodeGenerator,
}

impl CpuSolidBackend {
    /// Create a new CPU Solid backend.
    pub fn new() -> Self {
        Self {
            generator: CpuCodeGenerator {},
        }
    }
}

impl Default for CpuSolidBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SolidBackend for CpuSolidBackend {
    fn compile_kernel(
        &self,
        graph: &Graph,
        kernel: &FusedKernel,
        capture_ir: bool,
    ) -> Result<Arc<dyn ExecutableKernel>> {
        let isa = create_native_isa()?;

        let mut builder =
            cranelift_jit::JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        crate::core::codegen::math::register_math_symbols(&mut builder);

        let mut module = JITModule::new(builder);
        let generated = self
            .generator
            .generate_kernel(&mut module, graph, kernel, capture_ir)?;

        let func_id = module.declare_function(
            "kernel",
            cranelift_module::Linkage::Local,
            &generated.function.signature,
        )?;

        let mut ctx = cranelift_codegen::Context::for_function(generated.function);
        module.define_function(func_id, &mut ctx)?;
        module.finalize_definitions()?;

        let fn_ptr = module.get_finalized_function(func_id);

        Ok(Arc::new(SolidCompiledKernel {
            num_inputs: generated.num_inputs,
            _module: module,
            fn_ptr,
            debug_ir: generated.debug_ir,
        }))
    }
}

/// A compiled kernel owned by a Solid program.
struct SolidCompiledKernel {
    num_inputs: usize,
    _module: JITModule,
    fn_ptr: *const u8,
    debug_ir: Option<String>,
}

// Safety: The compiled code is immutable once created and the function pointer
// is valid for the lifetime of _module.
unsafe impl Send for SolidCompiledKernel {}
unsafe impl Sync for SolidCompiledKernel {}

impl ExecutableKernel for SolidCompiledKernel {
    fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    fn debug_ir(&self) -> Option<String> {
        self.debug_ir.clone()
    }

    fn execute(&self, inputs: &[*const u8], output: *mut u8, numel: usize) {
        debug_assert_eq!(inputs.len(), self.num_inputs);
        unsafe {
            let f: extern "C" fn(*const *const u8, *mut u8, u64) = std::mem::transmute(self.fn_ptr);
            f(inputs.as_ptr(), output, numel as u64);
        }
    }
}
