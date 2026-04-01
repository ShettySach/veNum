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
fn op_inputs(op: &crate::core::hlir::Op) -> Vec<NodeId> {
    use crate::core::hlir::Op;
    match op {
        Op::Const { .. } | Op::Load { .. } => vec![],
        Op::Store { value, .. } => vec![*value],
        Op::Neg(a)
        | Op::Recip(a)
        | Op::Exp(a)
        | Op::Log(a)
        | Op::Sqrt(a)
        | Op::Sin(a)
        | Op::Cos(a) => vec![*a],
        Op::Cast { input, .. }
        | Op::Reshape { input, .. }
        | Op::Permute { input, .. }
        | Op::Slice { input, .. }
        | Op::Expand { input, .. }
        | Op::Reduce { input, .. } => vec![*input],
        Op::Add(a, b) | Op::Mul(a, b) | Op::Max(a, b) | Op::Min(a, b) => vec![*a, *b],
        Op::Cmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        Op::Where {
            cond,
            then_val,
            else_val,
        } => vec![*cond, *then_val, *else_val],
        Op::Concat { inputs, .. } => inputs.clone(),
    }
}
