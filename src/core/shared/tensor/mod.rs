//! Unified tensor infrastructure for Liquid and Solid modes.
//!
//! This module provides a generic `Tensor<C: Context>` type that works with
//! both JIT (Liquid) and AOT (Solid) execution modes.
//!
//! # Usage
//!
//! ```ignore
//! // Liquid mode
//! use venum::liquid::{LiquidContext, LiquidTensor};
//! let cx = LiquidContext::new();
//! let a = LiquidTensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
//!
//! // Solid mode
//! use venum::solid::{SolidContext, Tensor};
//! let cx = SolidContext::new();
//! let a = Tensor::placeholder(&cx, DType::F32, vec![3]);
//! ```

mod constructors;
mod context;
mod helpers;
mod ops_conv;
mod ops_core;
mod ops_elementwise;
mod ops_matmul;
mod ops_reduce;
mod ops_shape;
mod overloads;
mod structure;

pub use context::Context;
pub use structure::Tensor;
