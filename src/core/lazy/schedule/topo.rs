use std::collections::{HashMap, HashSet};

use super::super::graph::{Graph, NodeId, Op};

use super::fused_kernel::{
    collect_kernel_inputs, try_build_tracker, FusedKernel, ReduceKind, ReduceOpItem, ReduceSpec,
    ShapeOpItem,
};
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

        // Reduce ops: try to fuse upstream elementwise/shape ops into a
        // single JIT kernel. Fall back to interpreted if fusion isn't possible.
        if node.op.is_reduce_op() {
            let expr_input = node.inputs[0];

            if let Some(reduce_spec) = reduce_spec_from_op(&node.op) {
                let expr_node = graph.node(expr_input);
                let can_fuse = expr_node.op.is_elementwise()
                    || expr_node.op.is_shape_op()
                    || matches!(expr_node.op, Op::Load | Op::Const(_));

                if can_fuse {
                    let iter_shape = expr_node.shape.clone();
                    let mut input_buffers = Vec::new();
                    let mut input_trackers = vec![None; graph.nodes.len()];
                    let mut shape_source_map = vec![None; graph.nodes.len()];
                    let mut num_tracked_inputs = 0usize;
                    let mut num_absorbed_shape_ops = 0usize;

                    if matches!(expr_node.op, Op::Load | Op::Const(_)) {
                        // Direct leaf — add as input buffer.
                        if matches!(expr_node.op, Op::Load) {
                            input_buffers.push(expr_input);
                        }
                    } else if expr_node.op.is_shape_op() {
                        // Shape op chain: try to absorb via tracker, with the
                        // source Load becoming the input buffer.
                        if let Some((source, tracker, chain)) = try_build_tracker(
                            graph,
                            expr_input,
                            &consumer_counts,
                            &iter_shape,
                        ) {
                            for &shape_id in &chain {
                                inlined.insert(shape_id);
                                if shape_source_map[shape_id.0].is_none() {
                                    num_absorbed_shape_ops += 1;
                                }
                                shape_source_map[shape_id.0] = Some(source);
                            }
                            if !input_buffers.contains(&source) {
                                input_buffers.push(source);
                            }
                            if input_trackers[source.0].is_none() {
                                num_tracked_inputs += 1;
                            }
                            input_trackers[source.0] = Some(tracker);
                        } else {
                            // Can't absorb shape ops — fall through to interpreted.
                            let input = node.inputs[0];
                            schedule.push(ScheduleItem::Reduce(ReduceOpItem {
                                root: id,
                                op: node.op.clone(),
                                input,
                                shape: node.shape.clone(),
                            }));
                            continue;
                        }
                    } else {
                        // Elementwise root — collect inputs recursively.
                        collect_kernel_inputs(
                            graph,
                            expr_input,
                            &consumer_counts,
                            &mut inlined,
                            &mut input_buffers,
                            &mut input_trackers,
                            &mut shape_source_map,
                            &mut num_tracked_inputs,
                            &mut num_absorbed_shape_ops,
                            &iter_shape,
                        );
                        inlined.insert(expr_input);
                    }

                    // Remove previously emitted items whose roots were absorbed.
                    schedule.retain(|item| match item {
                        ScheduleItem::Shape(s) => !inlined.contains(&s.root),
                        ScheduleItem::Fused(k) => !inlined.contains(&k.root),
                        _ => true,
                    });

                    schedule.push(ScheduleItem::Fused(FusedKernel {
                        root: id,
                        expr_root: expr_input,
                        input_buffers,
                        numel: node.numel(),
                        output_shape: node.shape.clone(),
                        iter_shape,
                        input_trackers,
                        shape_source_map,
                        num_tracked_inputs,
                        num_absorbed_shape_ops,
                        reduce: Some(reduce_spec),
                    }));
                    continue;
                }
            }

            // Fallback: interpreted reduce.
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
            let mut input_trackers = vec![None; graph.nodes.len()];
            let mut shape_source_map = vec![None; graph.nodes.len()];
            let mut num_tracked_inputs = 0usize;
            let mut num_absorbed_shape_ops = 0usize;
            let output_shape = node.shape.clone();

            collect_kernel_inputs(
                graph,
                id,
                &consumer_counts,
                &mut inlined,
                &mut input_buffers,
                &mut input_trackers,
                &mut shape_source_map,
                &mut num_tracked_inputs,
                &mut num_absorbed_shape_ops,
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
                expr_root: id,
                input_buffers,
                numel: node.numel(),
                output_shape: output_shape.clone(),
                iter_shape: output_shape,
                input_trackers,
                shape_source_map,
                num_tracked_inputs,
                num_absorbed_shape_ops,
                reduce: None,
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

/// Extract a `ReduceSpec` from a reduce `Op`.
fn reduce_spec_from_op(op: &Op) -> Option<ReduceSpec> {
    match op {
        Op::Sum(dims, keepdims) => Some(ReduceSpec {
            op: ReduceKind::Sum,
            dims: dims.clone(),
            keepdims: *keepdims,
        }),
        Op::Prod(dims, keepdims) => Some(ReduceSpec {
            op: ReduceKind::Prod,
            dims: dims.clone(),
            keepdims: *keepdims,
        }),
        Op::Max(dims, keepdims) => Some(ReduceSpec {
            op: ReduceKind::Max,
            dims: dims.clone(),
            keepdims: *keepdims,
        }),
        Op::Min(dims, keepdims) => Some(ReduceSpec {
            op: ReduceKind::Min,
            dims: dims.clone(),
            keepdims: *keepdims,
        }),
        _ => None,
    }
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
