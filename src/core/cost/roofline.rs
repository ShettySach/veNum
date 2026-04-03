use crate::core::schedule::{CostEstimate, KernelContext};

use super::hardware::CpuHardwareModel;

pub fn roofline_estimate(ctx: &KernelContext, hw: &CpuHardwareModel) -> CostEstimate {
    let trip_count = ctx
        .loop_bounds
        .iter()
        .copied()
        .map(|v| v.max(1) as f64)
        .product::<f64>();

    let reduce_penalty = if ctx.reduce_axes.is_empty() {
        1.0
    } else {
        1.0 + (ctx.reduce_axes.len() as f64 * 0.25)
    };

    let bytes_per_elem = match ctx.dtype {
        crate::core::hlir::DType::F64
        | crate::core::hlir::DType::I64
        | crate::core::hlir::DType::U64 => 8.0,
        crate::core::hlir::DType::F16
        | crate::core::hlir::DType::BF16
        | crate::core::hlir::DType::I16
        | crate::core::hlir::DType::U16 => 2.0,
        crate::core::hlir::DType::I8
        | crate::core::hlir::DType::U8
        | crate::core::hlir::DType::Bool => 1.0,
        _ => 4.0,
    };

    let flops = trip_count * 2.0 * reduce_penalty;
    let working_set_bytes = (trip_count * bytes_per_elem * 3.0) as usize;
    let bandwidth_bytes_per_cycle = (hw.mem_bandwidth_gb_s * 1e9) / 3.0e9;

    let compute_cycles = flops / ((hw.simd_width * hw.cores) as f64).max(1.0);
    let memory_cycles = (working_set_bytes as f64) / bandwidth_bytes_per_cycle.max(1.0);
    let total_cycles = compute_cycles.max(memory_cycles);

    CostEstimate {
        compute_cycles,
        memory_cycles,
        total_cycles,
        working_set_bytes,
        arithmetic_intensity: if working_set_bytes == 0 {
            0.0
        } else {
            flops / (working_set_bytes as f64)
        },
    }
}
