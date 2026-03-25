//! Solid execution context for AOT compilation.

use std::sync::{Arc, Mutex};

use crate::core::shared::graph::{Graph, NodeId};
use crate::core::shared::tensor::Context;

/// Context for Solid (AOT) execution mode.
///
/// Unlike LiquidContext which immediately executes operations, SolidContext:
/// - Records the full computation graph
/// - Tracks symbolic inputs (placeholders)
/// - Defers compilation until `compile()` is called
/// - Performs global optimization across the entire graph
///
/// # Example
///
/// ```rust
/// use venum::solid::{SolidContext, Tensor};
/// use venum::DType;
///
/// let cx = SolidContext::new();
///
/// // Create symbolic input
/// let input = Tensor::placeholder(&cx, DType::F32, vec![128, 768]);
///
/// // Build computation graph (no execution yet)
/// let output = input.exp().sum(&[0], false)?;
///
/// // Compile entire graph
/// let program = compile(&cx, &[input.id()], &[output.id()])?;
///
/// // Execute with concrete data
/// let results = program.execute(&[&input_buffer])?;
/// ```
#[derive(Clone)]
pub struct SolidContext {
    /// Computation graph (shared across all tensors)
    graph: Arc<Mutex<Graph>>,

    /// Tracked symbolic inputs (placeholders)
    inputs: Arc<Mutex<Vec<NodeId>>>,
}

impl SolidContext {
    /// Create a new Solid context.
    pub fn new() -> Self {
        Self {
            graph: Arc::new(Mutex::new(Graph::new())),
            inputs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get a reference to the computation graph.
    pub fn graph(&self) -> &Arc<Mutex<Graph>> {
        &self.graph
    }

    /// Register a node as a symbolic input (placeholder).
    ///
    /// This is called internally by `Tensor::placeholder()`.
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

impl Default for SolidContext {
    fn default() -> Self {
        Self::new()
    }
}

// Implement the shared Context trait for unified tensor operations
impl Context for SolidContext {
    fn graph(&self) -> &Arc<Mutex<Graph>> {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::shared::dtype::DType;
    use crate::core::shared::graph::{Node, Op};

    #[test]
    fn create_context() {
        let cx = SolidContext::new();
        assert_eq!(cx.num_nodes(), 0);
        assert_eq!(cx.inputs().len(), 0);
    }

    #[test]
    fn register_inputs() {
        let cx = SolidContext::new();

        // Add some placeholder nodes
        let graph = cx.graph();
        let id1 = graph.lock().unwrap().add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: vec![2, 3],
            dtype: DType::F32,
            buffer: None,
        });

        let id2 = graph.lock().unwrap().add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: vec![3, 4],
            dtype: DType::F32,
            buffer: None,
        });

        cx.register_input(id1);
        cx.register_input(id2);

        let inputs = cx.inputs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], id1);
        assert_eq!(inputs[1], id2);
    }
}
