use std::collections::{HashMap, HashSet};

use super::graph::{Graph, NodeId, Op};

/// A fused elementwise kernel: a tree of ops over input buffers producing one output.
#[derive(Debug)]
pub struct FusedKernel {
    pub root: NodeId,
    /// Leaf node ids that must be realized before this kernel runs.
    pub input_buffers: Vec<NodeId>,
    /// Number of elements in the output.
    pub numel: usize,
}

/// An item in the execution schedule.
#[derive(Debug)]
pub enum ScheduleItem {
    Fused(FusedKernel),
}

/// Build a linear execution schedule from the graph, rooted at `root`.
///
/// Algorithm:
/// 1. Topological sort from root (reverse post-order DFS).
/// 2. Walk in topo order, fusing chains of elementwise ops.
///    - A node is "fusible into its consumer" if:
///      a) It is elementwise.
///      b) It has exactly one consumer (the node being fused into).
///      c) It has the same numel as the consumer.
///    - Otherwise it becomes a barrier and must be realized as a separate kernel / buffer.
pub fn build_schedule(graph: &Graph, root: NodeId) -> Vec<ScheduleItem> {
    let topo = topo_sort(graph, root);

    // For each node, how many other nodes consume it?
    let consumer_counts = compute_consumer_counts(graph, &topo);

    // Nodes that are "inlined" into a consumer's kernel and don't need their own output buffer.
    let mut inlined: HashSet<NodeId> = HashSet::new();

    // Walk topo order and decide what gets its own kernel.
    let mut schedule = Vec::new();

    for &id in &topo {
        if inlined.contains(&id) {
            continue;
        }

        let node = graph.node(id);

        // Leaf nodes don't need kernels.
        if matches!(node.op, Op::Load | Op::Const(_)) {
            continue;
        }

        // If it's not elementwise (future: reductions, etc.), skip for now.
        if !node.op.is_elementwise() {
            continue;
        }

        // This node is the root of a fused kernel. Collect all input buffers
        // by walking the expression tree (nodes that are inlined into this kernel).
        let mut input_buffers = Vec::new();
        collect_kernel_inputs(
            graph,
            id,
            &consumer_counts,
            &mut inlined,
            &mut input_buffers,
        );

        schedule.push(ScheduleItem::Fused(FusedKernel {
            root: id,
            input_buffers,
            numel: node.numel(),
        }));
    }

    schedule
}

/// Recursively collect the leaf inputs of a fused kernel rooted at `id`.
/// Mark intermediate fusible nodes as inlined.
fn collect_kernel_inputs(
    graph: &Graph,
    id: NodeId,
    consumer_counts: &HashMap<NodeId, usize>,
    inlined: &mut HashSet<NodeId>,
    inputs: &mut Vec<NodeId>,
) {
    let node = graph.node(id);

    for &input_id in &node.inputs {
        let input_node = graph.node(input_id);

        // Const nodes are always inlined (they emit f32const in JIT, no buffer needed).
        if matches!(input_node.op, Op::Const(_)) {
            inlined.insert(input_id);
            continue;
        }

        let can_inline = input_node.op.is_elementwise()
            && *consumer_counts.get(&input_id).unwrap_or(&0) == 1
            && input_node.numel() == node.numel();

        if can_inline {
            inlined.insert(input_id);
            collect_kernel_inputs(graph, input_id, consumer_counts, inlined, inputs);
        } else {
            // This is a barrier — it's a leaf input to our kernel.
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
