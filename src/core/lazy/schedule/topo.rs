use std::collections::{HashMap, HashSet};

use super::super::graph::{Graph, NodeId, Op};

use super::fused_kernel::{collect_kernel_inputs, FusedKernel, ReduceOpItem, ShapeOpItem};
use super::schedule_item::ScheduleItem;

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

        // Shape ops are barriers unless absorbed by an elementwise consumer.
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
        }
    }

    schedule
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
