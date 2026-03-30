//! Code generation infrastructure for Cranelift-based backends.

pub mod backend;
pub mod cpu;
pub mod cranelift_setup;
pub mod emit;
pub mod expr;
pub mod generator;
pub mod math;
pub mod reduce;
pub mod tracker;

// Re-exports for convenient access
pub use generator::CodeGenerator;
