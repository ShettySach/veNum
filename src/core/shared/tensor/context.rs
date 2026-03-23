//! Context trait for unified tensor operations.
//!
//! This trait enables a single generic `Tensor<C: Context>` type
//! to work with both JIT (Liquid) and AOT (Solid) execution modes.

use std::sync::{Arc, Mutex};

use crate::core::shared::graph::Graph;

/// Execution context for tensors.
///
/// Implemented by both `LiquidContext` (JIT) and `SolidContext` (AOT).
/// The context owns the computation graph and determines execution semantics.
pub trait Context: Clone + Send + Sync {
    /// Access the shared computation graph.
    fn graph(&self) -> &Arc<Mutex<Graph>>;

    /// Check if two contexts share the same graph.
    ///
    /// Used by binary operations to ensure tensors are from the same context.
    fn same_graph(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.graph(), other.graph())
    }
}
