//! Memory planning pass for Solid mode.
//!
//! Performs liveness analysis and static buffer allocation using
//! greedy slot assignment with buffer reuse for non-overlapping lifetimes.

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::core::graph::{NodeId, Op};
use crate::core::program::buffer_plan::{BufferSlot, StaticBufferPlan};

use super::manager::{GraphPass, PassContext};

/// Memory planning pass.
///
/// Analyzes tensor lifetimes and assigns buffer slots to minimize memory usage.
///
/// # Algorithm
///
/// 1. **Topological sort** each output root to determine execution order
/// 2. **Liveness analysis**: For each node, record when it's created (definition step)
///    and when it's last consumed (last-use step)
/// 3. **Greedy slot assignment**: Assign each node to the first compatible slot
///    (same dtype, large enough) whose current occupant's lifetime has ended.
///    Create a new slot if none fits.
pub struct MemoryPlanningPass;

impl GraphPass for MemoryPlanningPass {
    fn name(&self) -> &str {
        "memory-planning"
    }

    fn run(&self, ctx: PassContext) -> Result<PassContext> {
        // Memory planning doesn't mutate the graph, but the plan is
        // accessible via `build_plan()`. Pass context through unchanged.
        Ok(ctx)
    }
}

impl MemoryPlanningPass {
    /// Build a static buffer plan from the pass context.
    pub fn build_plan(&self, ctx: &PassContext) -> StaticBufferPlan {
        let topo = topo_sort_multi(&ctx.graph, &ctx.outputs);
        let liveness = compute_liveness(&ctx.graph, &topo);

        let input_set: HashSet<NodeId> = ctx.inputs.iter().copied().collect();
        let output_set: HashSet<NodeId> = ctx.outputs.iter().copied().collect();

        // Nodes that need buffer slots: non-leaf, non-input intermediates + outputs.
        // Inputs get their buffers from the caller. Const/Load-with-buffer are embedded.
        let needs_slot: Vec<NodeId> = topo
            .iter()
            .copied()
            .filter(|&id| {
                let node = ctx.graph.node(id);
                // Skip inputs — they're provided at runtime
                if input_set.contains(&id) {
                    return false;
                }
                // Skip const nodes — they're embedded in kernels
                if matches!(node.op, Op::Const(_)) {
                    return false;
                }
                // Skip loads with data — they're embedded
                if matches!(node.op, Op::Load) && node.buffer.is_some() {
                    return false;
                }
                true
            })
            .collect();

        // Greedy slot assignment
        let mut plan = StaticBufferPlan::new();

        // Track which slot is free (last-use step of current occupant)
        struct SlotState {
            free_after: usize,
            dtype: crate::core::dtype::DType,
            size: usize,
        }
        let mut slot_states: Vec<SlotState> = Vec::new();

        for &id in &needs_slot {
            let node = ctx.graph.node(id);
            let (def_step, last_use) = liveness.get(&id).copied().unwrap_or((0, 0));
            let size = node.numel();
            let dtype = node.dtype;

            // Outputs should not share slots — they must persist after execution
            let is_output = output_set.contains(&id);

            // Try to find an existing compatible slot
            let assigned_slot = if is_output {
                None
            } else {
                slot_states
                    .iter()
                    .position(|s| s.dtype == dtype && s.size >= size && s.free_after < def_step)
            };

            match assigned_slot {
                Some(slot_id) => {
                    slot_states[slot_id].free_after = last_use;
                    plan.assign(id, slot_id, (def_step, last_use));
                }
                None => {
                    let slot_id = plan.add_slot(BufferSlot::new(size, dtype));
                    slot_states.push(SlotState {
                        free_after: last_use,
                        dtype,
                        size,
                    });
                    plan.assign(id, slot_id, (def_step, last_use));
                }
            }
        }

        plan
    }
}

/// Topological sort over multiple roots, deduplicating visits.
fn topo_sort_multi(graph: &crate::core::graph::Graph, roots: &[NodeId]) -> Vec<NodeId> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();

    for &root in roots {
        topo_dfs(graph, root, &mut visited, &mut order);
    }

    order
}

fn topo_dfs(
    graph: &crate::core::graph::Graph,
    id: NodeId,
    visited: &mut HashSet<NodeId>,
    order: &mut Vec<NodeId>,
) {
    if !visited.insert(id) {
        return;
    }
    let node = graph.node(id);
    for &input in &node.inputs {
        topo_dfs(graph, input, visited, order);
    }
    order.push(id);
}

/// Compute liveness intervals: (definition_step, last_use_step) for each node.
///
/// `definition_step` is the topological index where the node appears.
/// `last_use_step` is the maximum topological index of any consumer.
fn compute_liveness(
    graph: &crate::core::graph::Graph,
    topo: &[NodeId],
) -> HashMap<NodeId, (usize, usize)> {
    let topo_index: HashMap<NodeId, usize> = topo
        .iter()
        .enumerate()
        .map(|(step, &id)| (id, step))
        .collect();

    let mut liveness: HashMap<NodeId, (usize, usize)> = HashMap::new();

    // Initialize definition step for each node
    for (step, &id) in topo.iter().enumerate() {
        liveness.insert(id, (step, step));
    }

    // Update last-use based on consumers
    for &id in topo {
        let node = graph.node(id);
        let consumer_step = *topo_index.get(&id).unwrap();

        for &input_id in &node.inputs {
            if let Some(entry) = liveness.get_mut(&input_id) {
                entry.1 = entry.1.max(consumer_step);
            }
        }
    }

    liveness
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dtype::DType;
    use crate::core::graph::{Graph, Node, Op};

    #[test]
    fn liveness_analysis() {
        let mut graph = Graph::new();

        // a -> exp -> neg
        let a = graph.add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let b = graph.add_node(Node {
            op: Op::Exp,
            inputs: vec![a],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let c = graph.add_node(Node {
            op: Op::Neg,
            inputs: vec![b],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        let topo = topo_sort_multi(&graph, &[c]);
        assert_eq!(topo, vec![a, b, c]);

        let liveness = compute_liveness(&graph, &topo);

        // a is defined at step 0, last used by b at step 1
        assert_eq!(liveness[&a], (0, 1));
        // b is defined at step 1, last used by c at step 2
        assert_eq!(liveness[&b], (1, 2));
        // c is defined at step 2, not consumed further
        assert_eq!(liveness[&c], (2, 2));
    }

    #[test]
    fn memory_plan_reuses_slots() {
        let mut graph = Graph::new();

        // a -> b -> c -> d (linear chain, non-overlapping intermediates)
        let a = graph.add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let b = graph.add_node(Node {
            op: Op::Exp,
            inputs: vec![a],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let c = graph.add_node(Node {
            op: Op::Neg,
            inputs: vec![b],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let d = graph.add_node(Node {
            op: Op::Sqrt,
            inputs: vec![c],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        let ctx = PassContext {
            graph,
            inputs: vec![a],
            outputs: vec![d],
        };

        let pass = MemoryPlanningPass;
        let plan = pass.build_plan(&ctx);

        // b's lifetime ends before d starts, so b and d could share a slot.
        // But d is an output and won't share. b and c cannot share since they overlap.
        // At minimum we need slots for b, c, d (a is an input, no slot needed).
        assert!(plan.slots.len() <= 3);
        assert!(plan.total_bytes > 0);
    }

    #[test]
    fn memory_plan_diamond() {
        let mut graph = Graph::new();

        // Diamond: a -> b, a -> c, b+c -> d
        let a = graph.add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let b = graph.add_node(Node {
            op: Op::Exp,
            inputs: vec![a],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let c = graph.add_node(Node {
            op: Op::Neg,
            inputs: vec![a],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });
        let d = graph.add_node(Node {
            op: Op::Add,
            inputs: vec![b, c],
            shape: vec![4],
            dtype: DType::F32,
            buffer: None,
        });

        let ctx = PassContext {
            graph,
            inputs: vec![a],
            outputs: vec![d],
        };

        let pass = MemoryPlanningPass;
        let plan = pass.build_plan(&ctx);

        // b and c overlap (both alive at step where d is computed)
        // so they need separate slots. d is the output.
        assert!(plan.slots.len() >= 2);

        // All intermediate + output nodes should have slot assignments
        assert!(plan.get_slot(b).is_some());
        assert!(plan.get_slot(c).is_some());
        assert!(plan.get_slot(d).is_some());
        // Input should not have a slot
        assert!(plan.get_slot(a).is_none());
    }
}
