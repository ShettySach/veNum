//! Tensor operations for Solid mode.

pub mod constructors;
pub mod ops_elementwise;
pub mod ops_matmul;
pub mod ops_reduce;
pub mod ops_shape;
pub mod structure;

pub use structure::Tensor;
