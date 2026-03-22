use std::hash::{Hash, Hasher};

use super::super::graph::{Graph, NodeId, Op};
use super::super::schedule::FusedKernel;
use super::super::shape_tracker::ShapeTracker;

// -------- kernel signature (structural identity for caching) --------

/// Structural identity of a fused kernel, used as a cache key.
///
/// Two kernels with the same op tree, shapes, and tracker layouts will produce
/// identical machine code, so they can share a single `CompiledKernel`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelSignature(u64);

impl KernelSignature {
    /// Compute the structural signature of a fused kernel by hashing its
    /// expression tree (ops + shapes + trackers), ignoring concrete data.
    pub fn from_kernel(graph: &Graph, kernel: &FusedKernel) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash output shape, numel, and dtype.
        kernel.output_shape.hash(&mut hasher);
        kernel.numel.hash(&mut hasher);
        graph.node(kernel.root).dtype.hash(&mut hasher);

        // Hash iter_shape and reduce spec for reduce-fused kernels.
        kernel.iter_shape.hash(&mut hasher);
        if let Some(ref tracker) = kernel.output_tracker {
            1u8.hash(&mut hasher);
            hash_tracker(tracker, &mut hasher);
        } else {
            0u8.hash(&mut hasher);
        }
        if let Some(ref reduce) = kernel.reduce {
            1u8.hash(&mut hasher);
            reduce.op.hash(&mut hasher);
            reduce.dims.hash(&mut hasher);
            reduce.keepdims.hash(&mut hasher);
        } else {
            0u8.hash(&mut hasher);
        }

        let input_index = &kernel.input_index_map;

        // Hash the expression tree structure (from expr_root, not root).
        hash_expr(
            graph,
            kernel.expr_root,
            &input_index,
            &kernel.input_trackers,
            &kernel.shape_source_map,
            &mut hasher,
        );

        KernelSignature(hasher.finish())
    }
}

/// Recursively hash the expression tree rooted at `id`.
fn hash_expr(
    graph: &Graph,
    id: NodeId,
    input_index: &[Option<usize>],
    trackers: &[Option<ShapeTracker>],
    source_map: &[Option<NodeId>],
    hasher: &mut impl Hasher,
) {
    let node = graph.node(id);

    // Hash a discriminant tag for the op.
    std::mem::discriminant(&node.op).hash(hasher);

    match &node.op {
        Op::Const(v) => v.hash(hasher),
        Op::Load => {
            // Leaf - hash its input index and any tracker.
            let resolved = source_map[id.0].unwrap_or(id);
            if let Some(idx) = input_index[resolved.0] {
                0u8.hash(hasher); // tag: indexed input
                idx.hash(hasher);
                if let Some(tracker) = trackers[resolved.0].as_ref() {
                    hash_tracker(tracker, hasher);
                }
            }
        }
        op if !op.is_elementwise() => {
            // Inlined shape op resolved to a source buffer.
            let resolved = source_map[id.0].unwrap_or(id);
            if let Some(idx) = input_index[resolved.0] {
                1u8.hash(hasher); // tag: resolved shape op
                idx.hash(hasher);
                if let Some(tracker) = trackers[resolved.0].as_ref() {
                    hash_tracker(tracker, hasher);
                }
            }
        }
        _ => {
            // Elementwise ops - recurse into children.
            for &input_id in &node.inputs {
                hash_expr(graph, input_id, input_index, trackers, source_map, hasher);
            }
        }
    }
}

fn hash_tracker(tracker: &ShapeTracker, hasher: &mut impl Hasher) {
    tracker.shape.hash(hasher);
    tracker.strides.hash(hasher);
    tracker.offset.hash(hasher);
}
