//! Tensor execution context.

use std::sync::{Arc, Mutex};

use crate::core::graph::{Graph, NodeId};

/// Execution context for tensors.
///
/// Owns a computation graph that records operations for AOT compilation.
#[derive(Clone)]
pub struct Context {
    /// Computation graph (shared across all tensors)
    graph: Arc<Mutex<Graph>>,

    /// Tracked symbolic inputs (placeholders)
    inputs: Arc<Mutex<Vec<NodeId>>>,
}

impl Context {
    /// Create a new context.
    pub fn new() -> Self {
        Self {
            graph: Arc::new(Mutex::new(Graph::new())),
            inputs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Access the shared computation graph.
    pub fn graph(&self) -> &Arc<Mutex<Graph>> {
        &self.graph
    }

    /// Check if two contexts share the same graph.
    pub fn same_graph(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.graph, &other.graph)
    }

    /// Register a node as a symbolic input (placeholder).
    pub(crate) fn register_input(&self, id: NodeId) {
        self.inputs
            .lock()
            .expect("Inputs mutex should not be poisoned")
            .push(id);
    }

    /// Get all registered symbolic inputs.
    pub fn inputs(&self) -> Vec<NodeId> {
        self.inputs
            .lock()
            .expect("Inputs mutex should not be poisoned")
            .clone()
    }

    /// Get the number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.graph
            .lock()
            .expect("Graph mutex should not be poisoned")
            .nodes
            .len()
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
