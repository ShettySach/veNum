//! Cranelift ISA and settings setup.

use anyhow::Result;
use cranelift::prelude::{settings, types, Configurable};
use cranelift_codegen::isa::TargetIsa;

use crate::core::dtype::DType;

/// Create the Cranelift ISA for the current native platform.
pub fn create_native_isa() -> Result<std::sync::Arc<dyn TargetIsa>> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let isa_builder = cranelift_native::builder().map_err(|e| anyhow::anyhow!("{}", e))?;
    isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| anyhow::anyhow!("{}", e))
}

/// Map a DType to the corresponding Cranelift IR type.
pub fn dtype_to_cl_type(dtype: DType) -> types::Type {
    match dtype {
        DType::F32 => types::F32,
        DType::F64 => types::F64,
        DType::I32 => types::I32,
        DType::I64 => types::I64,
    }
}
