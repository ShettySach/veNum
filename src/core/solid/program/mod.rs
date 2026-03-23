//! Compiled program representation and execution.

pub mod buffer_plan;
pub mod program;
pub mod spec;

pub use buffer_plan::StaticBufferPlan;
pub use program::CompiledProgram;
pub use spec::TensorSpec;
