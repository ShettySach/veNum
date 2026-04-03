use std::collections::HashMap;

use super::{HLIRGraph, NodeId, Op};

#[cfg(test)]
pub fn canonicalize_hlir(graph: &HLIRGraph) -> HLIRGraph {
    // Find terminal nodes (not consumed by any other node).
    let mut consumed = std::collections::HashSet::new();
    for (_, node) in graph.topo_iter() {
        for inp in node.op.inputs() {
            consumed.insert(inp);
        }
    }
    let roots: Vec<NodeId> = graph
        .topo_iter()
        .map(|(id, _)| id)
        .filter(|id| !consumed.contains(id))
        .collect();

    if roots.is_empty() {
        return graph.clone();
    }
    canonicalize_with_roots(graph, &roots)
}

pub fn canonicalize_with_roots(graph: &HLIRGraph, roots: &[NodeId]) -> HLIRGraph {
    canonicalize_with_roots_and_map(graph, roots).0
}

pub fn canonicalize_with_roots_and_map(
    graph: &HLIRGraph,
    roots: &[NodeId],
) -> (HLIRGraph, HashMap<NodeId, NodeId>) {
    // Single pass: egglog handles algebraic simplification + reshape sinking.
    let (g, map) = super::egraph::egglog_algebraic(graph, roots);
    let remapped_roots: Vec<NodeId> = roots
        .iter()
        .map(|id| map.get(id).copied().unwrap_or(*id))
        .collect();

    // Dead node cleanup: collect only reachable nodes.
    let mut out = HLIRGraph::new();
    let mut memo: HashMap<NodeId, NodeId> = HashMap::new();
    for &id in &remapped_roots {
        let _ = copy_reachable(&g, id, &mut out, &mut memo);
    }

    let final_map: HashMap<NodeId, NodeId> = roots
        .iter()
        .zip(remapped_roots.iter())
        .map(|(orig, tmp_id)| (*orig, memo.get(tmp_id).copied().unwrap_or(*tmp_id)))
        .collect();

    (out, final_map)
}

/// Copy only nodes reachable from the given root into `dst`, without any rewrites.
fn copy_reachable(
    src: &HLIRGraph,
    id: NodeId,
    dst: &mut HLIRGraph,
    memo: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&existing) = memo.get(&id) {
        return existing;
    }

    let node = src.node(id);

    let remapped_inputs: Vec<NodeId> = node
        .op
        .inputs()
        .iter()
        .map(|inp| copy_reachable(src, *inp, dst, memo))
        .collect();

    let new_op = remap_op_inputs(&node.op, &remapped_inputs);
    let new_id = dst.add_node(new_op, node.ty.clone());
    memo.insert(id, new_id);
    new_id
}

/// Create a copy of `op` with its input NodeIds replaced by the values in `new_inputs`
/// (in the same order as `op.inputs()`).
pub(super) fn remap_op_inputs(op: &Op, new_inputs: &[NodeId]) -> Op {
    let mut it = new_inputs.iter().copied();
    match op {
        Op::Const {
            value,
            shape,
            dtype,
        } => Op::Const {
            value: value.clone(),
            shape: shape.clone(),
            dtype: *dtype,
        },
        Op::Load { buffer } => Op::Load { buffer: *buffer },
        Op::Store { buffer, .. } => Op::Store {
            buffer: *buffer,
            value: it.next().unwrap(),
        },
        Op::Neg(_) => Op::Neg(it.next().unwrap()),
        Op::Recip(_) => Op::Recip(it.next().unwrap()),
        Op::Exp(_) => Op::Exp(it.next().unwrap()),
        Op::Log(_) => Op::Log(it.next().unwrap()),
        Op::Sqrt(_) => Op::Sqrt(it.next().unwrap()),
        Op::Sin(_) => Op::Sin(it.next().unwrap()),
        Op::Cast { to, .. } => Op::Cast {
            input: it.next().unwrap(),
            to: *to,
        },
        Op::Add(_, _) => {
            let a = it.next().unwrap();
            Op::Add(a, it.next().unwrap())
        }
        Op::Mul(_, _) => {
            let a = it.next().unwrap();
            Op::Mul(a, it.next().unwrap())
        }
        Op::Max(_, _) => {
            let a = it.next().unwrap();
            Op::Max(a, it.next().unwrap())
        }
        Op::Min(_, _) => {
            let a = it.next().unwrap();
            Op::Min(a, it.next().unwrap())
        }
        Op::Cmp { op: cmp_op, .. } => {
            let lhs = it.next().unwrap();
            Op::Cmp {
                op: *cmp_op,
                lhs,
                rhs: it.next().unwrap(),
            }
        }
        Op::Where { .. } => {
            let cond = it.next().unwrap();
            let then_val = it.next().unwrap();
            Op::Where {
                cond,
                then_val,
                else_val: it.next().unwrap(),
            }
        }
        Op::Reduce {
            axes, op, keepdim, ..
        } => Op::Reduce {
            input: it.next().unwrap(),
            axes: axes.clone(),
            op: *op,
            keepdim: *keepdim,
        },
        Op::Reshape { shape, .. } => Op::Reshape {
            input: it.next().unwrap(),
            shape: shape.clone(),
        },
        Op::Permute { axes, .. } => Op::Permute {
            input: it.next().unwrap(),
            axes: axes.clone(),
        },
        Op::Slice { ranges, .. } => Op::Slice {
            input: it.next().unwrap(),
            ranges: ranges.clone(),
        },
        Op::Expand { shape, .. } => Op::Expand {
            input: it.next().unwrap(),
            shape: shape.clone(),
        },
        Op::Concat { axis, inputs } => Op::Concat {
            inputs: (0..inputs.len()).map(|_| it.next().unwrap()).collect(),
            axis: *axis,
        },
    }
}
