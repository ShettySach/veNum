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

/// Shape-op barrier item.
/// These run as standalone schedule steps (not fused into elementwise kernels yet).
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
///
/// Algorithm:
/// 1. Topological sort from root (reverse post-order DFS).
/// 2. Walk topo order:
///    - Leaf nodes (`Load`, `Const`) are skipped.
///    - Shape ops become `ScheduleItem::Shape` barriers.
///    - Reduce ops become `ScheduleItem::Reduce` barriers.
///    - Elementwise ops become `ScheduleItem::Fused`.
/// 3. Fusing rule for elementwise kernels:
///    - Input can be inlined iff:
///      a) input is elementwise
///      b) input has exactly one consumer
///      c) input has same `numel` as consumer
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

        // Shape ops are explicit barriers.
        if node.op.is_shape_op() {
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

        // Elementwise ops can be fused.
        if node.op.is_elementwise() {
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
            continue;
        }

        // Future non-elementwise ops (matmul/conv/etc.) are intentionally skipped here.
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

        // Const nodes are always inlined (they emit f32const in JIT).
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
