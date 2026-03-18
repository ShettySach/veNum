use std::collections::{HashMap, HashSet};

use super::graph::{Graph, NodeId, Op};
use super::shape_tracker::ShapeTracker;

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
    /// Key = NodeId of the input buffer, value = tracker mapping output indices → buffer offsets.
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

/// An item in the execution schedule.
#[derive(Debug)]
pub enum ScheduleItem {
    Fused(FusedKernel),
    Shape(ShapeOpItem),
    Reduce(ReduceOpItem),
}

/// Build a linear execution schedule from the graph, rooted at `root`.
pub fn build_schedule(graph: &Graph, root: NodeId) -> Vec<ScheduleItem> {
    let topo = topo_sort(graph, root);
    let consumer_counts = compute_consumer_counts(graph, &topo);

    // Nodes that are inlined into fused elementwise kernels.
    let mut inlined: HashSet<NodeId> = HashSet::new();

    let mut schedule = Vec::new();

    for &id in &topo {
        if inlined.contains(&id) {
            continue;
        }

        let node = graph.node(id);

        // Leaves don't need a schedule item.
        if matches!(node.op, Op::Load | Op::Const(_)) {
            continue;
        }

        // Shape ops: check if they are consumed only by fused kernels.
        // If so, they'll be absorbed when the consumer is processed.
        // If they feed into a reduce or have multiple consumers, emit a barrier.
        if node.op.is_shape_op() {
            // Check if this shape op will be absorbed by its consumer(s).
            // A shape op is absorbed if all its consumers are elementwise ops
            // (or other shape ops that eventually feed into elementwise ops).
            // For simplicity, we only absorb shape ops that are reachable from
            // an elementwise op during collect_kernel_inputs.
            // Here we emit it as a barrier; it may get removed if absorbed.
            //
            // Actually, let's just defer: if it hasn't been inlined by the time
            // we see it, emit a barrier.
            let input = node.inputs[0];
            schedule.push(ScheduleItem::Shape(ShapeOpItem {
                root: id,
                op: node.op.clone(),
                input,
                shape: node.shape.clone(),
            }));
            continue;
        }

        // Reduce ops are explicit barriers.
        if node.op.is_reduce_op() {
            let input = node.inputs[0];
            schedule.push(ScheduleItem::Reduce(ReduceOpItem {
                root: id,
                op: node.op.clone(),
                input,
                shape: node.shape.clone(),
            }));
            continue;
        }

        // Elementwise ops can be fused, absorbing shape ops along the way.
        if node.op.is_elementwise() {
            let mut input_buffers = Vec::new();
            let mut input_trackers = HashMap::new();
            let mut shape_source_map = HashMap::new();
            let output_shape = node.shape.clone();
            collect_kernel_inputs(
                graph,
                id,
                &consumer_counts,
                &mut inlined,
                &mut input_buffers,
                &mut input_trackers,
                &mut shape_source_map,
                &output_shape,
            );

            // Remove any previously emitted Shape items that were absorbed.
            schedule.retain(|item| {
                if let ScheduleItem::Shape(s) = item {
                    !inlined.contains(&s.root)
                } else {
                    true
                }
            });

            schedule.push(ScheduleItem::Fused(FusedKernel {
                root: id,
                input_buffers,
                numel: node.numel(),
                output_shape,
                input_trackers,
                shape_source_map,
            }));
            continue;
        }
    }

    schedule
}

/// Walk back through a chain of shape ops, composing a ShapeTracker.
/// Returns Some((source_node_id, tracker)) if the chain can be absorbed.
/// Returns None if any shape op in the chain can't be composed.
fn try_build_tracker(
    graph: &Graph,
    shape_node_id: NodeId,
    consumer_counts: &HashMap<NodeId, usize>,
    _output_shape: &[usize],
) -> Option<(NodeId, ShapeTracker, Vec<NodeId>)> {
    // Start with a contiguous tracker for the output shape (the elementwise op's shape).
    // We'll walk backwards through shape ops and apply each one to build the tracker
    // that maps output indices → source buffer offsets.
    //
    // Actually, we need to work differently: start from the source and compose forward.
    // First, collect the chain of shape ops, then compose them.

    let mut chain = Vec::new(); // (node_id, &Op)
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

/// Recursively collect the leaf inputs of a fused kernel rooted at `id`.
/// Mark intermediate fusible nodes as inlined.
/// When encountering shape ops, try to absorb them via ShapeTracker.
fn collect_kernel_inputs(
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

        // Const nodes are always inlined (they emit f32const in JIT).
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

/// Topological sort via reverse post-order DFS, rooted at `root`.
fn topo_sort(graph: &Graph, root: NodeId) -> Vec<NodeId> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    topo_dfs(graph, root, &mut visited, &mut order);
    order
}

fn topo_dfs(graph: &Graph, id: NodeId, visited: &mut HashSet<NodeId>, order: &mut Vec<NodeId>) {
    if !visited.insert(id) {
        return;
    }
    let node = graph.node(id);
    for &input in &node.inputs {
        topo_dfs(graph, input, visited, order);
    }
    order.push(id);
}

/// Count how many times each node in `topo` is referenced as an input.
fn compute_consumer_counts(graph: &Graph, topo: &[NodeId]) -> HashMap<NodeId, usize> {
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    let topo_set: HashSet<NodeId> = topo.iter().copied().collect();

    for &id in topo {
        let node = graph.node(id);
        for &input_id in &node.inputs {
            if topo_set.contains(&input_id) {
                *counts.entry(input_id).or_insert(0) += 1;
            }
        }
    }

    counts
}
