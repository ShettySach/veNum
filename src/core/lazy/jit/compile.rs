use anyhow::Result;
use cranelift::prelude::Configurable;
use cranelift_jit::JITModule;
use cranelift_module::Module;

use crate::core::lazy::jit::compiled::CompiledKernel;
use crate::core::lazy::schedule::FusedKernel;
use crate::core::shared::codegen::CodeGenerator;
use crate::core::shared::codegen::CpuCodeGenerator;
use crate::core::shared::graph::Graph;

/// Compile a fused elementwise kernel into native code via Cranelift.
///
/// When `capture_ir` is true, the Cranelift IR text is stored in the returned
/// `CompiledKernel` for debug/visualization purposes.
pub fn compile_kernel(
    graph: &Graph,
    kernel: &FusedKernel,
    capture_ir: bool,
) -> Result<CompiledKernel> {
    // Set up Cranelift ISA and JIT module
    let mut flag_builder = cranelift::prelude::settings::builder();
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let isa_builder = cranelift_native::builder().map_err(|e| anyhow::anyhow!("{}", e))?;
    let isa = isa_builder
        .finish(cranelift::prelude::settings::Flags::new(flag_builder))
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut builder =
        cranelift_jit::JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    // Register math symbols
    crate::core::shared::codegen::math::register_math_symbols(&mut builder);

    let mut module = JITModule::new(builder);

    // Use the shared CPU code generator to generate Cranelift IR
    // IMPORTANT: Pass the module by mutable reference so math function
    // references remain valid throughout the function's lifetime
    let generator = CpuCodeGenerator::new();
    let generated = generator.generate_kernel(&mut module, graph, kernel, capture_ir)?;

    // Now finalize: compile the Cranelift function to machine code
    // Declare the kernel function
    let func_id = module.declare_function(
        "kernel",
        cranelift_module::Linkage::Local,
        &generated.function.signature,
    )?;

    // Compile the function
    let mut ctx = cranelift_codegen::Context::for_function(generated.function);
    module.define_function(func_id, &mut ctx)?;
    module.finalize_definitions()?;

    let fn_ptr = module.get_finalized_function(func_id);

    Ok(CompiledKernel {
        num_inputs: generated.num_inputs,
        _module: module,
        fn_ptr,
        clif_ir: generated.debug_ir,
    })
}
