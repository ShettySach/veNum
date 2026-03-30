//! Compiler passes.
//!
//! The pass infrastructure provides a pipeline of graph transformations
//! that convert a raw computation graph into an optimized, executable program.
//!
//! # Pass Pipeline
//!
//! 1. **OptimizationPass** - Egglog equality saturation
//! 2. **FusionPass** - Global kernel fusion using `DefaultFusionPolicy`
//! 3. **MemoryPlanningPass** - Liveness analysis and static buffer allocation

pub mod fusion;
pub mod manager;
pub mod memory;
pub mod optimize;
