mod backend;
mod context;
mod fusion_policy;
mod jit;
pub(crate) mod kernel;
mod lru_cache;
mod plan;
pub(crate) mod render;
pub(crate) mod schedule;
mod tensor;
mod tests;

// Re-export shared types for public API (used by lib.rs)
pub use crate::core::shared::dtype::{DType, RealizedTensor, Scalar};

// Local exports
pub use context::LiquidContext;
