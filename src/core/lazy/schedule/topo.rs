use std::collections::{HashMap, HashSet};

use super::super::graph::{Graph, NodeId, Op};
use super::super::shape_tracker::ShapeTracker;

use super::fused_kernel::{
    collect_kernel_inputs, try_build_tracker, FusedKernel, ReduceKind, ReduceOpItem, ReduceSpec,
    ShapeOpItem,
};
use super::schedule_item::ScheduleItem;

fn build_input_index_map(graph_nodes_len: usize, input_buffers: &[NodeId]) -> Vec<Option<usize>> {
    let mut map = vec![None; graph_nodes_len];
    for (i, &buf_id) in input_buffers.iter().enumerate() {
        map[buf_id.0] = Some(i);
    }
    map
}

fn find_single_consumer(
    graph: &Graph,
    producer: NodeId,
    topo_set: &HashSet<NodeId>,
) -> Option<NodeId> {
    for (idx, node) in graph.nodes.iter().enumerate() {
        let nid = NodeId(idx);
        if !topo_set.contains(&nid) {
            continue;
        }
        if node.inputs.contains(&producer) {
            return Some(nid);
        }
    }
    None
}

fn build_output_tracker_chain(
    graph: &Graph,
    start: NodeId,
    consumer_counts: &HashMap<NodeId, usize>,
    topo_set: &HashSet<NodeId>,
) -> Option<(NodeId, ShapeTracker, Vec<NodeId>)> {
    let mut current = start;
    let mut chain = Vec::new();

    while *consumer_counts.get(&current).unwrap_or(&0) == 1 {
        let Some(next) = find_single_consumer(graph, current, topo_set) else {
            break;
        };
        let next_node = graph.node(next);
        if !next_node.op.is_shape_op() {
            break;
        }
        chain.push(next);
        current = next;
    }

    if chain.is_empty() {
        return None;
    }

    let mut tracker = ShapeTracker::contiguous(&graph.node(start).shape);

    for &shape_id in &chain {
        let node = graph.node(shape_id);
        tracker = match &node.op {
            Op::Reshape => tracker.reshape(&node.shape),
            // Forward-fused output chains must preserve numel. Expand changes
            // cardinality and would require different iteration semantics.
            Op::Expand => return None,
            Op::Permute(axes) => tracker.permute(axes),
            Op::Transpose(d1, d2) => tracker.transpose(*d1, *d2),
            Op::Squeeze => Some(tracker.squeeze()),
            Op::Unsqueeze(new_rank) => tracker.unsqueeze(*new_rank),
            Op::Flip(dims) => Some(tracker.flip(dims)),
            _ => return None,
        }?;
    }

    if graph.node(start).numel() != graph.node(current).numel() {
        return None;
    }

    Some((current, tracker, chain))
}

/// Build a linear execution schedule from the graph, rooted at `root`.
pub fn build_schedule(graph: &Graph, root: NodeId) -> Vec<ScheduleItem> {
    let topo = topo_sort(graph, root);
    let topo_set: HashSet<NodeId> = topo.iter().copied().collect();
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
                    let mut kernel_root = id;
                    let iter_shape = expr_node.shape.clone();
                    let mut input_buffers = Vec::new();
                    let mut input_trackers = vec![None; graph.nodes.len()];
                    let mut shape_source_map = vec![None; graph.nodes.len()];
                    let mut num_absorbed_shape_ops = 0usize;

                    if matches!(expr_node.op, Op::Load | Op::Const(_)) {
                        // Direct leaf — add as input buffer.
                        if matches!(expr_node.op, Op::Load) {
                            input_buffers.push(expr_input);
                        }
                    } else if expr_node.op.is_shape_op() {
                        // Shape op chain: try to absorb via tracker, with the
                        // source Load becoming the input buffer.
                        if let Some((source, tracker, chain)) =
                            try_build_tracker(graph, expr_input, &consumer_counts, &iter_shape)
                        {
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
                            &mut num_absorbed_shape_ops,
                            &iter_shape,
                        );
                        inlined.insert(expr_input);
                    }

                    let output_tracker = if let Some((final_root, tracker, chain)) =
                        build_output_tracker_chain(graph, id, &consumer_counts, &topo_set)
                    {
                        for &shape_id in &chain {
                            inlined.insert(shape_id);
                        }
                        kernel_root = final_root;
                        Some(tracker)
                    } else {
                        None
                    };

                    // Remove previously emitted items whose roots were absorbed.
                    schedule.retain(|item| match item {
                        ScheduleItem::Shape(s) => !inlined.contains(&s.root),
                        ScheduleItem::Fused(k) => !inlined.contains(&k.root),
                        _ => true,
                    });

                    let input_index_map = build_input_index_map(graph.nodes.len(), &input_buffers);
                    schedule.push(ScheduleItem::Fused(FusedKernel {
                        root: kernel_root,
                        expr_root: expr_input,
                        input_buffers,
                        numel: graph.node(kernel_root).numel(),
                        output_shape: node.shape.clone(),
                        iter_shape,
                        has_noncontiguous_trackers: input_trackers
                            .iter()
                            .filter_map(|t| t.as_ref())
                            .any(|t| !t.is_contiguous()),
                        input_index_map,
                        input_trackers,
                        shape_source_map,
                        num_absorbed_shape_ops,
                        output_tracker,
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
            let mut num_absorbed_shape_ops = 0usize;
            let output_shape = node.shape.clone();
            let mut kernel_root = id;

            collect_kernel_inputs(
                graph,
                id,
                &consumer_counts,
                &mut inlined,
                &mut input_buffers,
                &mut input_trackers,
                &mut shape_source_map,
                &mut num_absorbed_shape_ops,
                &output_shape,
            );

            // Remove previously emitted items whose roots were absorbed.
            let output_tracker = if let Some((final_root, tracker, chain)) =
                build_output_tracker_chain(graph, id, &consumer_counts, &topo_set)
            {
                for &shape_id in &chain {
                    inlined.insert(shape_id);
                }
                kernel_root = final_root;
                Some(tracker)
            } else {
                None
            };

            schedule.retain(|item| match item {
                ScheduleItem::Shape(s) => !inlined.contains(&s.root),
                ScheduleItem::Fused(k) => !inlined.contains(&k.root),
                _ => true,
            });

            let input_index_map = build_input_index_map(graph.nodes.len(), &input_buffers);
            schedule.push(ScheduleItem::Fused(FusedKernel {
                root: kernel_root,
                expr_root: id,
                input_buffers,
                numel: graph.node(kernel_root).numel(),
                output_shape: output_shape.clone(),
                iter_shape: output_shape,
                has_noncontiguous_trackers: input_trackers
                    .iter()
                    .filter_map(|t| t.as_ref())
                    .any(|t| !t.is_contiguous()),
                input_index_map,
                input_trackers,
                shape_source_map,
                num_absorbed_shape_ops,
                output_tracker,
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
