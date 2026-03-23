//! Shared code generation infrastructure for Cranelift-based backends.
//!
//! This module provides the common code generation components used by both
//! Liquid (JIT) and Solid (AOT) execution modes.

pub mod cpu;
pub mod cranelift_setup;
pub mod emit;
pub mod expr;
pub mod generator;
pub mod math;
pub mod reduce;
pub mod tracker;

// Re-exports for convenient access
pub use cpu::CpuCodeGenerator;
pub use generator::CodeGenerator;
