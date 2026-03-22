use std::collections::{HashMap, HashSet};

use crate::core::lazy::graph::{Graph, NodeId, Op};
use crate::core::lazy::shape_tracker::ShapeTracker;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Sum,
    Prod,
    Max,
    Min,
}

#[derive(Clone, Debug)]
pub struct ReduceSpec {
    pub op: ReduceKind,
    pub dims: Vec<usize>,
    pub keepdims: bool,
}

/// A fused elementwise kernel: a tree of ops over input buffers producing one output.
#[derive(Debug)]
pub struct FusedKernel {
    pub root: NodeId,
    /// The node whose expression tree is evaluated per iteration element.
    /// For pure elementwise kernels this equals `root`; for reduce kernels
    /// this is the reduce input subtree root.
    pub expr_root: NodeId,
    /// Leaf node ids that must be realized before this kernel runs.
    pub input_buffers: Vec<NodeId>,
    /// Number of elements in the output.
    pub numel: usize,
    /// Output shape for multi-dim index decomposition in the JIT.
    pub output_shape: Vec<usize>,
    /// Logical iteration shape for expression evaluation.
    /// For pure elementwise kernels this equals `output_shape`;
    /// for reduce kernels this is the pre-reduce input shape.
    pub iter_shape: Vec<usize>,
    /// Tracker for inputs reached through absorbed shape ops, keyed by source NodeId.
    pub input_trackers: HashMap<NodeId, ShapeTracker>,
    /// Maps absorbed shape-op NodeId to its ultimate source buffer NodeId.
    /// Used by the JIT to resolve graph references that point at absorbed shape ops.
    pub shape_source_map: HashMap<NodeId, NodeId>,
    /// Maps source NodeId to its index in `input_buffers`.
    pub input_index_map: HashMap<NodeId, usize>,
    /// True if at least one tracked input has a non-contiguous layout.
    pub has_noncontiguous_trackers: bool,
    /// Optional tracker that maps kernel iteration indices to final output
    /// layout when downstream shape ops are absorbed (forward fusion).
    pub output_tracker: Option<ShapeTracker>,
    /// Present for reduce-fused kernels.
    pub reduce: Option<ReduceSpec>,
}

/// Shape-op barrier item.
/// These run as standalone schedule steps when not absorbed into a fused kernel.
#[derive(Debug, Clone)]
pub struct ShapeOpItem {
    pub root: NodeId,
    pub op: Op,
    pub input: NodeId,
    pub shape: Vec<usize>,
}

/// Reduce-op barrier item.
/// Reductions are scheduled separately from elementwise fusion.
#[derive(Debug, Clone)]
pub struct ReduceOpItem {
    pub root: NodeId,
    pub op: Op,
    pub input: NodeId,
    pub shape: Vec<usize>,
}

pub(super) fn collect_kernel_inputs(
    graph: &Graph,
    id: NodeId,
    consumer_counts: &HashMap<NodeId, usize>,
    inlined: &mut HashSet<NodeId>,
    inputs: &mut Vec<NodeId>,
    trackers: &mut HashMap<NodeId, ShapeTracker>,
    source_map: &mut HashMap<NodeId, NodeId>,
    output_shape: &[usize],
) {
    let node = graph.node(id);

    for &input_id in &node.inputs {
        let input_node = graph.node(input_id);

        // Const nodes are always inlined (they emit consts in JIT).
        if matches!(input_node.op, Op::Const(_)) {
            inlined.insert(input_id);
            continue;
        }

        // Try to absorb a shape op chain.
        if input_node.op.is_shape_op() {
            if let Some((source, tracker, chain)) =
                try_build_tracker(graph, input_id, consumer_counts, output_shape)
            {
                // Mark all shape ops in the chain as inlined and record source mapping.
                for &shape_id in &chain {
                    inlined.insert(shape_id);
                    source_map.insert(shape_id, source);
                }

                // If the tracker is contiguous, the source's elements map 1:1
                // to the kernel's flat iteration. If the source is also a
                // single-consumer elementwise op with matching numel, we can
                // inline its expression tree instead of materializing a buffer.
                let source_node = graph.node(source);
                let can_inline_source = tracker.is_contiguous()
                    && source_node.op.is_elementwise()
                    && *consumer_counts.get(&source).unwrap_or(&0) == 1
                    && source_node.numel() == node.numel();

                if can_inline_source {
                    inlined.insert(source);
                    collect_kernel_inputs(
                        graph,
                        source,
                        consumer_counts,
                        inlined,
                        inputs,
                        trackers,
                        source_map,
                        output_shape,
                    );
                } else {
                    // The source becomes a kernel input with a tracker.
                    if !inputs.contains(&source) {
                        inputs.push(source);
                    }
                    trackers.insert(source, tracker);
                }
                continue;
            }
        }

        let can_inline = input_node.op.is_elementwise()
            && *consumer_counts.get(&input_id).unwrap_or(&0) == 1
            && input_node.numel() == node.numel();

        if can_inline {
            inlined.insert(input_id);
            collect_kernel_inputs(
                graph,
                input_id,
                consumer_counts,
                inlined,
                inputs,
                trackers,
                source_map,
                output_shape,
            );
        } else {
            // This is a barrier leaf input for the fused kernel.
            if !inputs.contains(&input_id) {
                inputs.push(input_id);
            }
        }
    }
}

/// Walk back through a chain of shape ops, composing a ShapeTracker.
/// Returns Some((source_node_id, tracker, chain)) if the chain can be absorbed.
pub(super) fn try_build_tracker(
    graph: &Graph,
    shape_node_id: NodeId,
    consumer_counts: &HashMap<NodeId, usize>,
    _output_shape: &[usize],
) -> Option<(NodeId, ShapeTracker, Vec<NodeId>)> {
    let mut chain = Vec::new();
    let mut current = shape_node_id;

    loop {
        let node = graph.node(current);
        if !node.op.is_shape_op() {
            break;
        }
        // Only absorb if single consumer.
        if *consumer_counts.get(&current).unwrap_or(&0) > 1 {
            break;
        }
        chain.push(current);
        current = node.inputs[0];
    }

    // `current` is the source node (Load or a realized barrier).
    let source = current;
    let source_shape = &graph.node(source).shape;

    // Build tracker starting from source shape, composing each shape op forward.
    let mut tracker = ShapeTracker::contiguous(source_shape);

    for &shape_id in chain.iter().rev() {
        let node = graph.node(shape_id);
        tracker = match &node.op {
            Op::Reshape => tracker.reshape(&node.shape),
            Op::Expand => tracker.expand(&node.shape),
            Op::Permute(axes) => tracker.permute(axes),
            Op::Transpose(d1, d2) => tracker.transpose(*d1, *d2),
            Op::Squeeze => Some(tracker.squeeze()),
            Op::Unsqueeze(new_rank) => tracker.unsqueeze(*new_rank),
            Op::Flip(dims) => Some(tracker.flip(dims)),
            _ => return None,
        }?;
    }

    Some((source, tracker, chain))
}
