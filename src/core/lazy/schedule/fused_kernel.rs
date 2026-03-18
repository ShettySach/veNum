use std::collections::{HashMap, HashSet};

use super::super::graph::{Graph, NodeId, Op};
use super::super::shape_tracker::ShapeTracker;

/// A fused elementwise kernel: a tree of ops over input buffers producing one output.
#[derive(Debug)]
pub struct FusedKernel {
    pub root: NodeId,
    /// Leaf node ids that must be realized before this kernel runs.
    pub input_buffers: Vec<NodeId>,
    /// Number of elements in the output.
    pub numel: usize,
    /// Output shape for multi-dim index decomposition in the JIT.
    pub output_shape: Vec<usize>,
    /// ShapeTrackers for input buffers that were reached through shape ops.
    /// Key = NodeId of the input buffer, value = tracker mapping output indices -> buffer offsets.
    pub input_trackers: HashMap<NodeId, ShapeTracker>,
    /// Maps inlined shape op NodeIds to their ultimate source buffer NodeId.
    /// Used by the JIT to resolve graph references that point at absorbed shape ops.
    pub shape_source_map: HashMap<NodeId, NodeId>,
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

                // The source becomes a kernel input with a tracker.
                if !inputs.contains(&source) {
                    inputs.push(source);
                }
                trackers.insert(source, tracker);
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
fn try_build_tracker(
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
            Op::Reshape(new_shape) => tracker.reshape(new_shape),
            Op::Expand(expansions) => tracker.expand(expansions),
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
