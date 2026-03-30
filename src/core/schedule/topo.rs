use std::collections::{HashMap, HashSet};

use crate::core::graph::{Graph, NodeId, Op};
use crate::core::schedule::FusionPolicy;
use crate::core::schedule::fused_kernel::{
    FusedKernel, KernelInputCollector, ReduceKind, ReduceOpItem, ReduceSpec, ShapeOpItem,
    collect_kernel_inputs, try_build_tracker,
};
use crate::core::schedule::schedule_item::ScheduleItem;
use crate::core::shape_tracker::ShapeTracker;

fn build_input_index_map(input_buffers: &[NodeId]) -> HashMap<NodeId, usize> {
    input_buffers
        .iter()
        .enumerate()
        .map(|(i, &buf_id)| (buf_id, i))
        .collect()
}

fn find_single_consumer(
    consumers: &HashMap<NodeId, Vec<NodeId>>,
    producer: NodeId,
) -> Option<NodeId> {
    let consumer_list = consumers.get(&producer)?;
    if consumer_list.len() == 1 {
        Some(consumer_list[0])
    } else {
        None
    }
}

fn build_output_tracker_chain(
    graph: &Graph,
    start: NodeId,
    consumer_counts: &HashMap<NodeId, usize>,
    consumers: &HashMap<NodeId, Vec<NodeId>>,
) -> Option<(NodeId, ShapeTracker, Vec<NodeId>)> {
    let mut current = start;
    let mut chain = Vec::with_capacity(4);

    while *consumer_counts.get(&current).unwrap_or(&0) == 1 {
        let Some(next) = find_single_consumer(consumers, current) else {
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

// ── Analysis types ──────────────────────────────────────────────────────

/// Per-node analysis result from pass 1.
struct NodePlan {
    /// Nodes absorbed into this planned item (inlined elementwise, shape ops, consts).
    absorbed: HashSet<NodeId>,
    /// The schedule item to emit for this node.
    item: ScheduleItem,
}

/// Full analysis result from pass 1.
struct ScheduleAnalysis {
    topo: Vec<NodeId>,
    /// Global set of all absorbed/inlined nodes.
    inlined: HashSet<NodeId>,
    /// Planned item per topo-visit node id. Keyed by the node id at which
    /// the analysis was triggered, NOT by `FusedKernel.root` (which may
    /// differ due to forward fusion).
    planned: Vec<Option<ScheduleItem>>,
}

// ── Pass 1: analyze ─────────────────────────────────────────────────────

fn analyze_schedule(graph: &Graph, root: NodeId, policy: &dyn FusionPolicy) -> ScheduleAnalysis {
    let topo = topo_sort(graph, root);
    let consumer_counts = compute_consumer_counts(graph, &topo);
    let consumers = compute_consumers(graph, &topo);

    let mut inlined: HashSet<NodeId> = HashSet::new();
    let mut planned: Vec<Option<ScheduleItem>> = (0..graph.nodes.len()).map(|_| None).collect();

    for &id in &topo {
        if inlined.contains(&id) {
            continue;
        }

        let node = graph.node(id);

        if matches!(node.op, Op::Load | Op::Const(_)) {
            continue;
        }

        let plan = if node.op.is_shape_op() {
            analyze_shape_node(graph, id)
        } else if node.op.is_reduce_op() {
            analyze_reduce_node(graph, id, &consumer_counts, &consumers, policy)
        } else if node.op.is_elementwise() {
            analyze_elementwise_node(graph, id, &consumer_counts, &consumers, policy)
        } else {
            continue;
        };

        inlined.extend(plan.absorbed.iter().copied());
        planned[id.0] = Some(plan.item);
    }

    ScheduleAnalysis {
        topo,
        inlined,
        planned,
    }
}

fn analyze_shape_node(graph: &Graph, id: NodeId) -> NodePlan {
    let node = graph.node(id);
    NodePlan {
        absorbed: HashSet::new(),
        item: ScheduleItem::Shape(ShapeOpItem {
            root: id,
            op: node.op.clone(),
            input: node.inputs[0],
        }),
    }
}

fn analyze_elementwise_node(
    graph: &Graph,
    id: NodeId,
    consumer_counts: &HashMap<NodeId, usize>,
    consumers: &HashMap<NodeId, Vec<NodeId>>,
    policy: &dyn FusionPolicy,
) -> NodePlan {
    let node_count_estimate = graph.nodes.len() / 8;
    let mut absorbed = HashSet::with_capacity(node_count_estimate);
    let mut input_buffers = Vec::with_capacity(8);
    let mut input_set = HashSet::with_capacity(8);
    let mut input_trackers = HashMap::with_capacity(8);
    let mut shape_source_map = HashMap::with_capacity(8);
    let output_shape = graph.node(id).shape.clone();
    let mut kernel_root = id;

    collect_kernel_inputs(
        graph,
        id,
        &mut KernelInputCollector {
            consumer_counts,
            policy,
            inlined: &mut absorbed,
            inputs: &mut input_buffers,
            input_set: &mut input_set,
            trackers: &mut input_trackers,
            source_map: &mut shape_source_map,
        },
        &output_shape,
    );

    let output_tracker = if let Some((final_root, tracker, chain)) =
        build_output_tracker_chain(graph, id, consumer_counts, consumers)
    {
        absorbed.extend(chain.iter().copied());
        kernel_root = final_root;
        Some(tracker)
    } else {
        None
    };

    let input_index_map = build_input_index_map(&input_buffers);

    NodePlan {
        absorbed,
        item: ScheduleItem::Fused(Box::new(FusedKernel {
            root: kernel_root,
            expr_root: id,
            input_buffers,
            numel: graph.node(kernel_root).numel(),
            output_shape: output_shape.clone(),
            iter_shape: output_shape,
            has_noncontiguous_trackers: input_trackers.values().any(|t| !t.is_contiguous()),
            input_index_map,
            input_trackers,
            shape_source_map,
            output_tracker,
            reduce: None,
        })),
    }
}

fn analyze_reduce_node(
    graph: &Graph,
    id: NodeId,
    consumer_counts: &HashMap<NodeId, usize>,
    consumers: &HashMap<NodeId, Vec<NodeId>>,
    policy: &dyn FusionPolicy,
) -> NodePlan {
    let node = graph.node(id);
    let expr_input = node.inputs[0];

    if let Some(reduce_spec) = reduce_spec_from_op(&node.op)
        && let Some(plan) = try_fused_reduce(
            graph,
            id,
            expr_input,
            reduce_spec,
            consumer_counts,
            consumers,
            policy,
        )
    {
        return plan;
    }

    // Fallback: interpreted reduce.
    NodePlan {
        absorbed: HashSet::new(),
        item: ScheduleItem::Reduce(ReduceOpItem {
            root: id,
            op: node.op.clone(),
            input: expr_input,
        }),
    }
}

/// Try to build a fused reduce kernel. Returns `None` if fusion isn't possible,
/// leaving no partial state behind.
fn try_fused_reduce(
    graph: &Graph,
    id: NodeId,
    expr_input: NodeId,
    reduce_spec: ReduceSpec,
    consumer_counts: &HashMap<NodeId, usize>,
    consumers: &HashMap<NodeId, Vec<NodeId>>,
    policy: &dyn FusionPolicy,
) -> Option<NodePlan> {
    let node = graph.node(id);
    let expr_node = graph.node(expr_input);

    let can_fuse = expr_node.op.is_elementwise()
        || expr_node.op.is_shape_op()
        || matches!(expr_node.op, Op::Load | Op::Const(_));

    if !can_fuse {
        return None;
    }

    let node_count_estimate = graph.nodes.len() / 8;
    let mut absorbed = HashSet::with_capacity(node_count_estimate);
    let mut kernel_root = id;
    let iter_shape = expr_node.shape.clone();
    let mut input_buffers = Vec::with_capacity(8);
    let mut input_set = HashSet::with_capacity(8);
    let mut input_trackers = HashMap::with_capacity(8);
    let mut shape_source_map = HashMap::with_capacity(8);

    match &expr_node.op {
        Op::Load => {
            input_buffers.push(expr_input);
            input_set.insert(expr_input);
        }
        Op::Const(_) => {}
        op if op.is_shape_op() => {
            let (source, tracker, chain) =
                try_build_tracker(graph, expr_input, consumer_counts, &iter_shape)?;
            for &shape_id in &chain {
                absorbed.insert(shape_id);
                shape_source_map.insert(shape_id, source);
            }
            if input_set.insert(source) {
                input_buffers.push(source);
            }
            input_trackers.insert(source, tracker);
        }
        _ => {
            // Elementwise root — collect inputs recursively.
            collect_kernel_inputs(
                graph,
                expr_input,
                &mut KernelInputCollector {
                    consumer_counts,
                    policy,
                    inlined: &mut absorbed,
                    inputs: &mut input_buffers,
                    input_set: &mut input_set,
                    trackers: &mut input_trackers,
                    source_map: &mut shape_source_map,
                },
                &iter_shape,
            );
            absorbed.insert(expr_input);
        }
    }

    let output_tracker = if let Some((final_root, tracker, chain)) =
        build_output_tracker_chain(graph, id, consumer_counts, consumers)
    {
        absorbed.extend(chain.iter().copied());
        kernel_root = final_root;
        Some(tracker)
    } else {
        None
    };

    let input_index_map = build_input_index_map(&input_buffers);

    Some(NodePlan {
        absorbed,
        item: ScheduleItem::Fused(Box::new(FusedKernel {
            root: kernel_root,
            expr_root: expr_input,
            input_buffers,
            numel: graph.node(kernel_root).numel(),
            output_shape: node.shape.clone(),
            iter_shape,
            has_noncontiguous_trackers: input_trackers.values().any(|t| !t.is_contiguous()),
            input_index_map,
            input_trackers,
            shape_source_map,
            output_tracker,
            reduce: Some(reduce_spec),
        })),
    })
}

// ── Pass 2: emit ────────────────────────────────────────────────────────

fn emit_schedule(analysis: &mut ScheduleAnalysis) -> Vec<ScheduleItem> {
    let mut schedule = Vec::new();

    for &id in &analysis.topo {
        if analysis.inlined.contains(&id) {
            continue;
        }
        if let Some(item) = analysis.planned[id.0].take() {
            schedule.push(item);
        }
    }

    schedule
}

// ── Public entry point ──────────────────────────────────────────────────

/// Build a linear execution schedule from the graph with a custom fusion policy.
pub fn build_schedule_with_policy(
    graph: &Graph,
    root: NodeId,
    policy: &dyn FusionPolicy,
) -> Vec<ScheduleItem> {
    let mut analysis = analyze_schedule(graph, root, policy);
    emit_schedule(&mut analysis)
}

// ── Helpers ─────────────────────────────────────────────────────────────

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
    for &id in topo {
        let node = graph.node(id);
        for &input_id in &node.inputs {
            *counts.entry(input_id).or_insert(0) += 1;
        }
    }

    counts
}

fn compute_consumers(graph: &Graph, topo: &[NodeId]) -> HashMap<NodeId, Vec<NodeId>> {
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

    for &id in topo {
        let node = graph.node(id);
        for &input_id in &node.inputs {
            consumers.entry(input_id).or_default().push(id);
        }
    }

    consumers
}
