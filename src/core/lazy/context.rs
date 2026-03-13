use std::sync::{Arc, Mutex};

use super::graph::Graph;

/// Shared lazy execution context.
///
/// A `Context` owns a single computation graph so tensors created within the
/// same context naturally share nodes and avoid cross-graph imports.
#[derive(Clone)]
pub struct Context {
    graph: Arc<Mutex<Graph>>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    /// Create a new lazy context with an empty graph.
    pub fn new() -> Self {
        Self {
            graph: Arc::new(Mutex::new(Graph::new())),
        }
    }

    /// Get a clone of the shared graph handle.
    pub(crate) fn graph(&self) -> Arc<Mutex<Graph>> {
        Arc::clone(&self.graph)
    }
}
