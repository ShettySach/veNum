use std::sync::Arc;

use crate::core::lazy::dtype::Scalar;
use crate::core::lazy::graph::{Graph, NodeId, Op};
use crate::core::lazy::kernel::ExecutableKernel;

/// A single step in a cached execution plan.
pub enum ExecItem {
    /// Execute a JIT-compiled fused elementwise kernel.
    Kernel {
        kernel: Arc<dyn ExecutableKernel>,
        /// Node IDs of input buffers (in the execution graph).
        inputs: Vec<NodeId>,
        /// Node ID of the output buffer.
        output: NodeId,
    },
    /// Execute an interpreted shape operation.
    Shape {
        op: Op,
        input: NodeId,
        output: NodeId,
    },
    /// Execute an interpreted reduce operation.
    Reduce {
        op: Op,
        input: NodeId,
        output: NodeId,
    },
    /// Fill a buffer with a constant scalar value.
    ConstFill {
        value: Scalar,
        output: NodeId,
        numel: usize,
    },
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
    /// Node ID of the final output in the execution graph.
    pub output: NodeId,
    /// Shape of the output tensor.
    pub output_shape: Vec<usize>,
}
