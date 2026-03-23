pub mod codegen;
pub mod dtype;
pub mod exec;
pub mod graph;
pub mod optimize;
pub mod schedule;
pub mod shape_tracker;
pub mod tensor;

// Note: Types are accessed directly via submodules (e.g., crate::core::shared::dtype::DType)
// rather than re-exported here to avoid unused import warnings.
// External API re-exports are in lazy/mod.rs and lib.rs.
