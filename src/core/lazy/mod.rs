mod backend;
mod context;
mod fusion_policy;
mod jit;
mod kernel;
mod lru_cache;
mod plan;
mod render;
pub(crate) mod schedule;
mod tensor;
mod tests;

// Re-export shared types for public API (used by lib.rs)
pub use crate::core::shared::dtype::{DType, RealizedTensor, Scalar};

// Local exports
pub use context::Context;
#[allow(unused_imports)]
pub use fusion_policy::LiquidFusionPolicy;
pub use tensor::Tensor;
