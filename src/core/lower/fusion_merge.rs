use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{Result, anyhow, bail};

use crate::core::hlir::{HLIRGraph, NodeId};
use crate::core::llir::program::Kernel;
use crate::core::schedule::FusionTopology;

pub fn merge_fusion_topology(
    _hlir: &HLIRGraph,
    topology: &FusionTopology,
    nodes: &[NodeId],
    kernels: &[Kernel],
) -> Result<Vec<Kernel>> {
    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    let kernel_map: HashMap<NodeId, Kernel> = kernels.iter().map(|k| (k.root, k.clone())).collect();

    match topology {
        FusionTopology::Chain => {
            let ordered = kernels_for_nodes(nodes, &kernel_map)?;
            merge_chain(&ordered)
        }
        FusionTopology::FanIn {
            consumer,
            producers,
        } => merge_fanin(*consumer, producers, &kernel_map),
        FusionTopology::FanOut {
            producer,
            consumers,
        } => merge_fanout(*producer, consumers, &kernel_map),
        FusionTopology::DAG { edges } => merge_dag(nodes, edges, &kernel_map),
    }
}

fn merge_chain(kernels: &[Kernel]) -> Result<Vec<Kernel>> {
    if kernels.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let mut current = kernels[0].clone();

    for next in kernels.iter().skip(1) {
        if current.loop_nest.loops == next.loop_nest.loops {
            current
                .loop_nest
                .body
                .extend(next.loop_nest.body.iter().cloned());
        } else {
            out.push(current);
            current = next.clone();
        }
    }

    out.push(current);
    Ok(out)
}

fn merge_fanin(
    consumer: NodeId,
    producers: &[NodeId],
    kernels: &HashMap<NodeId, Kernel>,
) -> Result<Vec<Kernel>> {
    let mut producer_kernels = kernels_for_nodes(producers, kernels)?;
    producer_kernels = merge_chain(&producer_kernels)?;

    let consumer_kernel = kernels
        .get(&consumer)
        .cloned()
        .ok_or_else(|| anyhow!("missing FanIn consumer kernel for node {:?}", consumer))?;

    producer_kernels.push(consumer_kernel);
    Ok(producer_kernels)
}

fn merge_fanout(
    producer: NodeId,
    consumers: &[NodeId],
    kernels: &HashMap<NodeId, Kernel>,
) -> Result<Vec<Kernel>> {
    let mut out = Vec::new();
    out.push(
        kernels
            .get(&producer)
            .cloned()
            .ok_or_else(|| anyhow!("missing FanOut producer kernel for node {:?}", producer))?,
    );
    out.extend(kernels_for_nodes(consumers, kernels)?);
    Ok(out)
}

fn merge_dag(
    nodes: &[NodeId],
    edges: &[(NodeId, NodeId)],
    kernels: &HashMap<NodeId, Kernel>,
) -> Result<Vec<Kernel>> {
    let order = topo_sort_nodes(nodes, edges)?;
    let mut indegree: HashMap<NodeId, usize> = HashMap::new();
    let mut outdegree: HashMap<NodeId, usize> = HashMap::new();
    let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
    let mut edge_set: HashSet<(NodeId, NodeId)> = HashSet::new();

    for &(src, dst) in edges {
        if !node_set.contains(&src) || !node_set.contains(&dst) {
            continue;
        }
        *indegree.entry(dst).or_insert(0) += 1;
        *outdegree.entry(src).or_insert(0) += 1;
        edge_set.insert((src, dst));
    }

    let ordered = kernels_for_nodes(&order, kernels)?;
    if ordered.is_empty() {
        return Ok(Vec::new());
    }

    let mut merged = Vec::new();
    let mut current = ordered[0].clone();
    for next in ordered.iter().skip(1) {
        let src = current.root;
        let dst = next.root;
        let can_inline_chain = edge_set.contains(&(src, dst))
            && outdegree.get(&src).copied().unwrap_or(0) == 1
            && indegree.get(&dst).copied().unwrap_or(0) == 1
            && current.loop_nest.loops == next.loop_nest.loops;

        if can_inline_chain {
            current
                .loop_nest
                .body
                .extend(next.loop_nest.body.iter().cloned());
        } else {
            merged.push(current);
            current = next.clone();
        }
    }
    merged.push(current);
    Ok(merged)
}

fn kernels_for_nodes(nodes: &[NodeId], kernels: &HashMap<NodeId, Kernel>) -> Result<Vec<Kernel>> {
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        out.push(
            kernels
                .get(node)
                .cloned()
                .ok_or_else(|| anyhow!("missing kernel for node {:?}", node))?,
        );
    }
    Ok(out)
}

fn topo_sort_nodes(nodes: &[NodeId], edges: &[(NodeId, NodeId)]) -> Result<Vec<NodeId>> {
    let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
    let mut indegree: HashMap<NodeId, usize> = nodes.iter().copied().map(|n| (n, 0)).collect();
    let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

    for &(src, dst) in edges {
        if !node_set.contains(&src) || !node_set.contains(&dst) {
            continue;
        }
        adj.entry(src).or_default().push(dst);
        *indegree.entry(dst).or_insert(0) += 1;
    }

    let mut q: VecDeque<NodeId> = indegree
        .iter()
        .filter_map(|(&node, &deg)| if deg == 0 { Some(node) } else { None })
        .collect();
    let mut order = Vec::with_capacity(nodes.len());

    while let Some(node) = q.pop_front() {
        order.push(node);
        if let Some(ns) = adj.get(&node) {
            for &n in ns {
                if let Some(v) = indegree.get_mut(&n) {
                    *v -= 1;
                    if *v == 0 {
                        q.push_back(n);
                    }
                }
            }
        }
    }

    if order.len() != node_set.len() {
        bail!("DAG fusion topology contains a cycle");
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use crate::core::hlir::{BufferId, DType, Dim, HLIRGraph, NodeId, Op, TensorType};
    use crate::core::llir::affine::AffineExpr;
    use crate::core::llir::loop_nest::{Loop, LoopAnnotations, LoopKind, LoopNest};
    use crate::core::llir::program::{Kernel, KernelId};
    use crate::core::llir::stmt::Stmt;
    use crate::core::schedule::FusionTopology;

    use super::merge_fusion_topology;

    #[test]
    fn fanin_merges_compatible_producers_before_consumer() {
        let hlir = HLIRGraph::new();
        let n0 = NodeId(0);
        let n1 = NodeId(1);
        let n2 = NodeId(2);

        let k0 = kernel_for(n0, 1, 0);
        let k1 = kernel_for(n1, 1, 1);
        let k2 = kernel_for(n2, 2, 2);

        let out = merge_fusion_topology(
            &hlir,
            &FusionTopology::FanIn {
                consumer: n2,
                producers: vec![n0, n1],
            },
            &[n0, n1, n2],
            &[k0, k1, k2],
        )
        .expect("fanin merge should succeed");

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].root, n0);
        assert_eq!(out[1].root, n2);
    }

    #[test]
    fn fanout_keeps_consumers_separate() {
        let hlir = HLIRGraph::new();
        let n0 = NodeId(0);
        let n1 = NodeId(1);
        let n2 = NodeId(2);

        let k0 = kernel_for(n0, 1, 0);
        let k1 = kernel_for(n1, 1, 1);
        let k2 = kernel_for(n2, 1, 2);

        let out = merge_fusion_topology(
            &hlir,
            &FusionTopology::FanOut {
                producer: n0,
                consumers: vec![n1, n2],
            },
            &[n0, n1, n2],
            &[k0, k1, k2],
        )
        .expect("fanout merge should succeed");

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].root, n0);
        assert_eq!(out[1].root, n1);
        assert_eq!(out[2].root, n2);
    }

    #[test]
    fn dag_merges_single_use_internal_chain() {
        let hlir = HLIRGraph::new();
        let n0 = NodeId(0);
        let n1 = NodeId(1);
        let n2 = NodeId(2);

        let k0 = kernel_for(n0, 1, 0);
        let k1 = kernel_for(n1, 1, 1);
        let k2 = kernel_for(n2, 2, 2);

        let out = merge_fusion_topology(
            &hlir,
            &FusionTopology::DAG {
                edges: vec![(n0, n1), (n1, n2)],
            },
            &[n0, n1, n2],
            &[k0, k1, k2],
        )
        .expect("dag merge should succeed");

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].root, n0);
        assert_eq!(out[1].root, n2);
    }

    fn kernel_for(root: NodeId, ub: i64, kid: usize) -> Kernel {
        Kernel {
            id: KernelId(kid),
            name: format!("k{kid}"),
            root,
            op: Op::Load {
                buffer: BufferId(root.0),
            },
            ty: TensorType::contiguous(vec![Dim::Const(ub)], DType::F32),
            loop_nest: LoopNest {
                loops: vec![Loop {
                    var: "i0".to_owned(),
                    lower: AffineExpr::constant(0),
                    upper: AffineExpr::constant(ub),
                    step: 1,
                    kind: LoopKind::Sequential,
                    annotations: LoopAnnotations::default(),
                }],
                body: vec![Stmt::Barrier],
            },
            allocs: Vec::new(),
        }
    }
}
