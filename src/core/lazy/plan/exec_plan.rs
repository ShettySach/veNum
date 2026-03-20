use std::sync::Arc;

use super::super::dtype::{DType, Scalar};
use super::super::graph::{Graph, NodeId, Op};
use super::super::jit::CompiledKernel;

/// A single step in a cached execution plan.
pub enum ExecItem {
    /// Execute a JIT-compiled fused elementwise kernel.
    Kernel {
        compiled: Arc<CompiledKernel>,
        /// Node IDs of input buffers (in the execution graph).
        inputs: Vec<NodeId>,
        /// Node ID of the output buffer.
        output: NodeId,
        numel: usize,
        dtype: DType,
    },
    /// Execute an interpreted shape operation.
    Shape {
        op: Op,
        input: NodeId,
        output: NodeId,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
    /// Execute an interpreted reduce operation.
    Reduce {
        op: Op,
        input: NodeId,
        output: NodeId,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
    /// Fill a buffer with a constant scalar value.
    ConstFill {
        value: Scalar,
        output: NodeId,
        numel: usize,
    },
}

/// A buffer that must be supplied at runtime to execute the plan.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PlanInput {
    /// Node ID in the execution graph whose buffer must be provided.
    pub node_id: NodeId,
    /// Corresponding NodeId in the original source graph.
    pub src_node_id: NodeId,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// A cached, replayable execution plan.
///
/// Built once from a computation graph + schedule, then replayed
/// on each `realize()` call by binding runtime input buffers and
/// executing each `ExecItem` in order.
#[allow(dead_code)]
pub struct ExecutionPlan {
    /// The self-contained execution graph (optimized/cloned subgraph).
    pub exec_graph: Graph,
    /// Ordered list of execution steps.
    pub items: Vec<ExecItem>,
    /// Input buffers that must be supplied at runtime.
    pub inputs: Vec<PlanInput>,
    /// Node ID of the final output in the execution graph.
    pub output: NodeId,
    /// Shape of the output tensor.
    pub output_shape: Vec<usize>,
}
