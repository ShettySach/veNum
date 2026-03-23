//! Compiled program representation and execution.

pub mod buffer_plan;
pub mod compiled_program;
pub mod spec;

pub use buffer_plan::StaticBufferPlan;
pub use compiled_program::CompiledProgram;
pub use spec::TensorSpec;
