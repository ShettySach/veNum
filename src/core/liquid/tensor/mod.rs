//! Liquid tensor module.
//!
//! Re-exports the shared `Tensor<LiquidContext>` type and provides
//! Liquid-specific extensions like `realize()` and `arange()`.

mod constructors;
mod helpers;
mod realize;
mod visualize;
