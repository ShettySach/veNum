//! Region planning for the upcoming typed egglog pipeline.

use std::collections::HashSet;

use super::super::{HLIRGraph, NodeId, Op};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionPlan {
    pub region_roots: Vec<NodeId>,
    pub barriers: Vec<BarrierNode>,
    pub symbolic_leaves: Vec<LeafRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarrierNode {
    pub src_id: NodeId,
    pub inputs: Vec<NodeId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LeafRef {
    Load(NodeId),
    BarrierOutput(NodeId),
}

pub fn plan_regions(graph: &HLIRGraph, roots: &[NodeId]) -> RegionPlan {
    let mut planner = RegionPlanner::new(graph);
    for &root in roots {
        planner.visit_root(root);
    }
    planner.finish()
}

struct RegionPlanner<'a> {
    graph: &'a HLIRGraph,
    region_roots: Vec<NodeId>,
    seen_region_roots: HashSet<NodeId>,
    barriers: Vec<BarrierNode>,
    seen_barriers: HashSet<NodeId>,
    symbolic_leaves: Vec<LeafRef>,
    seen_symbolic_leaves: HashSet<LeafRef>,
}

impl<'a> RegionPlanner<'a> {
    fn new(graph: &'a HLIRGraph) -> Self {
        Self {
            graph,
            region_roots: Vec::new(),
            seen_region_roots: HashSet::new(),
            barriers: Vec::new(),
            seen_barriers: HashSet::new(),
            symbolic_leaves: Vec::new(),
            seen_symbolic_leaves: HashSet::new(),
        }
    }

    fn finish(self) -> RegionPlan {
        RegionPlan {
            region_roots: self.region_roots,
            barriers: self.barriers,
            symbolic_leaves: self.symbolic_leaves,
        }
    }

    fn visit_root(&mut self, id: NodeId) {
        if Self::is_barrier(&self.graph.node(id).op) {
            self.visit_barrier(id);
        } else {
            self.record_region_root(id);
            self.visit_algebraic(id);
        }
    }

    fn visit_algebraic(&mut self, id: NodeId) {
        let node = self.graph.node(id);
        match &node.op {
            Op::Load { .. } => {
                self.record_symbolic_leaf(LeafRef::Load(id));
            }
            Op::Const { .. } => {}
            _ => {
                for input in node.op.inputs() {
                    if Self::is_barrier(&self.graph.node(input).op) {
                        self.visit_barrier(input);
                        self.record_symbolic_leaf(LeafRef::BarrierOutput(input));
                    } else {
                        self.visit_algebraic(input);
                    }
                }
            }
        }
    }

    fn visit_barrier(&mut self, id: NodeId) {
        if !self.seen_barriers.insert(id) {
            return;
        }

        let inputs = self.graph.node(id).op.inputs().into_vec();
        for &input in &inputs {
            if Self::is_barrier(&self.graph.node(input).op) {
                self.visit_barrier(input);
            } else {
                if Self::is_extractable_algebraic(&self.graph.node(input).op) {
                    self.record_region_root(input);
                }
                self.visit_algebraic(input);
            }
        }

        self.barriers.push(BarrierNode { src_id: id, inputs });
    }

    fn record_region_root(&mut self, id: NodeId) {
        if self.seen_region_roots.insert(id) {
            self.region_roots.push(id);
        }
    }

    fn record_symbolic_leaf(&mut self, leaf: LeafRef) {
        if self.seen_symbolic_leaves.insert(leaf.clone()) {
            self.symbolic_leaves.push(leaf);
        }
    }

    fn is_extractable_algebraic(op: &Op) -> bool {
        Self::is_algebraic(op) && !matches!(op, Op::Load { .. } | Op::Const { .. })
    }

    fn is_barrier(op: &Op) -> bool {
        !Self::is_algebraic(op)
    }

    fn is_algebraic(op: &Op) -> bool {
        matches!(
            op,
            Op::Const { .. }
                | Op::Load { .. }
                | Op::Neg(_)
                | Op::Recip(_)
                | Op::Exp(_)
                | Op::Log(_)
                | Op::Sqrt(_)
                | Op::Sin(_)
                | Op::Cast { .. }
                | Op::Add(_, _)
                | Op::Mul(_, _)
                | Op::Max(_, _)
                | Op::Min(_, _)
                | Op::Reshape { .. }
                | Op::Permute { .. }
                | Op::Expand { .. }
                | Op::Broadcast { .. }
        )
    }
}
