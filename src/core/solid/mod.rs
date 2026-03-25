//! Solid execution mode: Luminal-style AOT whole-program compilation.
//!
//! # Overview
//!
//! Solid mode compiles entire computation graphs ahead-of-time using:
//! - Global optimization via egglog equality saturation
//! - Aggressive kernel fusion across the whole program
//! - Static buffer allocation with memory reuse
//! - AOT compilation to native code
//!
//! # Example
//!
//! ```rust,ignore
//! use venum::solid::{SolidContext, Tensor, compile};
//! use venum::DType;
//!
//! // Create context
//! let cx = SolidContext::new();
//!
//! // Create symbolic inputs
//! let input = Tensor::placeholder(&cx, DType::F32, vec![128, 768]);
//! let weights = Tensor::from_slice(&cx, &weight_data, vec![768, 768]);
//!
//! // Build computation graph
//! let hidden = input.matmul(&weights)?;
//! let output = hidden.exp();
//!
//! // Compile entire graph (AOT)
//! let program = compile(&cx, &[input.id()], &[output.id()])?;
//!
//! // Execute with runtime inputs
//! let results = program.execute(&[&input_buffer])?;
//! ```

pub mod backend;
pub mod compile;
pub mod context;
pub mod fusion_policy;
pub mod pass;
pub mod program;
pub mod tensor;
mod tests;

pub use backend::{CpuSolidBackend, SolidBackend};
pub use compile::compile;
pub use context::SolidContext;
pub use fusion_policy::SolidFusionPolicy;
pub use pass::{FusionPass, GraphPass, MemoryPlanningPass, OptimizationPass, PassManager};
pub use program::{CompiledProgram, StaticBufferPlan, TensorSpec};
