//! Liquid tensor module.
//!
//! Re-exports the shared `Tensor<LiquidContext>` type and provides
//! Liquid-specific extensions like `realize()` and `arange()`.

mod liquid_constructors;
mod realize;
mod visualize;
