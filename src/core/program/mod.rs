//! Compiled program representation and execution.

pub mod buffer_plan;
pub mod compiled_program;
mod labels;
pub mod spec;

pub use compiled_program::{CompiledProgram, Output};
