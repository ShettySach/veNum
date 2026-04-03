use crate::core::schedule::{BackendClass, CostEstimate, HardwareModel, KernelContext, Opt};

use super::roofline::roofline_estimate;

#[derive(Clone, Debug)]
pub struct CpuHardwareModel {
    pub simd_width: usize,
    pub cores: usize,
    pub l1_bytes: usize,
    pub l2_bytes: usize,
    pub mem_bandwidth_gb_s: f64,
}

impl Default for CpuHardwareModel {
    fn default() -> Self {
        Self {
            simd_width: 8,
            cores: 8,
            l1_bytes: 32 * 1024,
            l2_bytes: 512 * 1024,
            mem_bandwidth_gb_s: 50.0,
        }
    }
}

impl HardwareModel for CpuHardwareModel {
    fn estimate_cost(&self, ctx: &KernelContext) -> CostEstimate {
        roofline_estimate(ctx, self)
    }

    fn opt_candidates(&self, ctx: &KernelContext) -> Vec<Opt> {
        if !matches!(ctx.backend, BackendClass::Cpu) {
            return Vec::new();
        }
        crate::core::schedule::candidates::opt_candidates(ctx)
    }
}
