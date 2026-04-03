#[cfg(test)]
use std::collections::HashSet;

#[cfg(test)]
use crate::core::hlir::{HLIRGraph, NodeId};

#[cfg(test)]
use super::decision::FusionGroup;

#[cfg(test)]
pub fn is_fusion_group_legal(graph: &HLIRGraph, group: &FusionGroup) -> bool {
    if group.nodes.is_empty() {
        return false;
    }

    let node_set: HashSet<NodeId> = group.nodes.iter().copied().collect();

    for &id in &group.nodes {
        for inp in op_inputs(&graph.node(id).op) {
            if node_set.contains(&inp) && inp == id {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
pub fn fusion_groups_form_acyclic_partition(groups: &[FusionGroup]) -> bool {
    let mut seen = HashSet::new();
    for g in groups {
        for &n in &g.nodes {
            if !seen.insert(n) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
fn op_inputs(op: &crate::core::hlir::Op) -> smallvec::SmallVec<[NodeId; 3]> {
    op.inputs()
}
