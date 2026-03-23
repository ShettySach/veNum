//! Compiler passes for Solid mode.
//!
//! The pass infrastructure provides a pipeline of graph transformations
//! that convert a raw computation graph into an optimized, executable program.
//!
//! # Pass Pipeline
//!
//! 1. **OptimizationPass** - Egglog equality saturation (shared with Liquid)
//! 2. **FusionPass** - Global kernel fusion using `SolidFusionPolicy`
//! 3. **MemoryPlanningPass** - Liveness analysis and static buffer allocation

pub mod fusion;
pub mod manager;
pub mod memory;
pub mod optimize;

pub use fusion::FusionPass;
pub use manager::{GraphPass, PassManager};
pub use memory::MemoryPlanningPass;
pub use optimize::OptimizationPass;
