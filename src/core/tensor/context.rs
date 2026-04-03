//! Tensor execution context.

use std::sync::{Arc, Mutex};

use crate::core::hlir::{BufferId, HLIRGraph, NodeId, Op, canonicalize_with_roots};

/// Execution context for tensors.
///
/// Owns a computation graph that records operations for AOT compilation.
#[derive(Clone)]
pub struct Context {
    /// Computation graph (shared across all tensors)
    graph: Arc<Mutex<HLIRGraph>>,

    /// Tracked symbolic inputs (placeholders)
    inputs: Arc<Mutex<Vec<NodeId>>>,

    next_buffer_id: Arc<Mutex<usize>>,
}

impl Context {
    /// Create a new context.
    pub fn new() -> Self {
        Self {
            graph: Arc::new(Mutex::new(HLIRGraph::new())),
            inputs: Arc::new(Mutex::new(Vec::new())),
            next_buffer_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Access the shared computation graph.
    pub fn graph(&self) -> &Arc<Mutex<HLIRGraph>> {
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

    pub(crate) fn alloc_buffer_id(&self) -> BufferId {
        let mut guard = self
            .next_buffer_id
            .lock()
            .expect("Buffer id mutex should not be poisoned");
        let id = *guard;
        *guard += 1;
        BufferId(id)
    }

    /// Get all registered symbolic inputs.
    pub fn inputs(&self) -> Vec<NodeId> {
        self.inputs
            .lock()
            .expect("Inputs mutex should not be poisoned")
            .clone()
    }

    pub fn input_buffers(&self) -> Vec<BufferId> {
        let input_ids = self.inputs();
        let graph = self
            .graph
            .lock()
            .expect("Graph mutex should not be poisoned");
        input_ids
            .into_iter()
            .filter_map(|id| match &graph.node(id).op {
                Op::Load { buffer } => Some(*buffer),
                _ => None,
            })
            .collect()
    }

    /// Get the number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.graph
            .lock()
            .expect("Graph mutex should not be poisoned")
            .len()
    }

    pub fn graph_mermaid(&self) -> String {
        let graph = self
            .graph
            .lock()
            .expect("Graph mutex should not be poisoned");
        crate::print::to_mermaid(&graph)
    }

    pub fn optimized_graph_mermaid(&self, roots: &[NodeId]) -> String {
        let graph = self
            .graph
            .lock()
            .expect("Graph mutex should not be poisoned")
            .clone();

        let optimized = canonicalize_with_roots(&graph, roots);
        crate::print::to_mermaid(&optimized)
    }

    pub fn schedule_mermaid(
        &self,
        roots: &[NodeId],
        config: &crate::core::compile::SearchConfig,
    ) -> String {
        use crate::core::schedule::ScheduleSearcher;

        let graph = self
            .graph
            .lock()
            .expect("Graph mutex should not be poisoned")
            .clone();

        let optimized = if config.enable_canonicalization {
            canonicalize_with_roots(&graph, roots)
        } else {
            graph
        };

        let mut searcher = ScheduleSearcher::new(crate::core::cost::CpuHardwareModel::default());
        searcher.beam_width = config.beam_width;
        searcher.max_iterations = config.max_iterations;

        let decision = searcher.search_best(&optimized).map(|(d, _)| d);
        match decision {
            Ok(d) => crate::print::to_schedule_mermaid(&d),
            Err(_) => "flowchart LR\n  e[\"search produced no decision\"]".to_owned(),
        }
    }

    pub fn llir_mermaid(
        &self,
        roots: &[NodeId],
        config: &crate::core::compile::SearchConfig,
    ) -> String {
        use crate::core::dep::NoOpDependenceAnalyzer;
        use crate::core::lower::lower;
        use crate::core::schedule::ScheduleSearcher;

        let graph = self
            .graph
            .lock()
            .expect("Graph mutex should not be poisoned")
            .clone();

        let optimized = if config.enable_canonicalization {
            canonicalize_with_roots(&graph, roots)
        } else {
            graph
        };

        let mut searcher = ScheduleSearcher::new(crate::core::cost::CpuHardwareModel::default());
        searcher.beam_width = config.beam_width;
        searcher.max_iterations = config.max_iterations;

        let decision = match searcher.search_best(&optimized) {
            Ok((d, _)) => d,
            Err(_) => return "flowchart TD\n  e[\"search produced no decision\"]".to_owned(),
        };

        match lower(&optimized, &decision, &NoOpDependenceAnalyzer) {
            Ok(program) => crate::print::to_llir_mermaid(&program),
            Err(e) => format!("flowchart TD\n  e[\"lowering failed: {}\"]", e),
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
