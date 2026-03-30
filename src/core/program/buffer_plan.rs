//! Static buffer allocation plan for AOT compilation.

use std::collections::HashMap;

use crate::core::dtype::DType;
use crate::core::graph::NodeId;

/// A buffer slot that can be reused across non-overlapping tensor lifetimes.
#[derive(Debug, Clone)]
pub struct BufferSlot {
    /// Size in number of elements
    pub size: usize,

    /// Data type of elements in this slot
    pub dtype: DType,
}

impl BufferSlot {
    /// Create a new buffer slot.
    pub fn new(size: usize, dtype: DType) -> Self {
        Self { size, dtype }
    }

    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size * self.dtype.size_bytes()
    }
}

/// Static buffer allocation plan for a compiled program.
///
/// This plan:
/// - Assigns each intermediate tensor to a buffer slot
/// - Reuses slots for tensors with non-overlapping lifetimes
/// - Minimizes total memory required
///
/// # Memory Planning Algorithm
///
/// 1. **Liveness Analysis**: Determine when each tensor is first created and last used
/// 2. **Interference Graph**: Two tensors interfere if their lifetimes overlap
/// 3. **Graph Coloring**: Assign non-interfering tensors to the same slot (color)
/// 4. **Slot Sizing**: Each slot must be large enough for all assigned tensors
#[derive(Debug, Clone)]
pub struct StaticBufferPlan {
    /// Buffer slots (indexed by slot ID)
    pub slots: Vec<BufferSlot>,

    /// Maps NodeId to slot index
    pub allocation: HashMap<NodeId, usize>,

    /// Liveness intervals: (first_use_step, last_use_step) per node
    /// Steps correspond to topological execution order
    pub liveness: HashMap<NodeId, (usize, usize)>,

    /// Total memory required in bytes
    pub total_bytes: usize,
}

impl StaticBufferPlan {
    /// Create an empty buffer plan.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            allocation: HashMap::new(),
            liveness: HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Get the slot assigned to a node.
    pub fn get_slot(&self, node_id: NodeId) -> Option<usize> {
        self.allocation.get(&node_id).copied()
    }

    /// Get the liveness interval for a node.
    pub fn get_liveness(&self, node_id: NodeId) -> Option<(usize, usize)> {
        self.liveness.get(&node_id).copied()
    }

    /// Check if two nodes' lifetimes overlap (interfere).
    pub fn interferes(&self, a: NodeId, b: NodeId) -> bool {
        if let (Some((a_start, a_end)), Some((b_start, b_end))) =
            (self.get_liveness(a), self.get_liveness(b))
        {
            // Intervals [a_start, a_end] and [b_start, b_end] overlap if:
            // a_start <= b_end && b_start <= a_end
            a_start <= b_end && b_start <= a_end
        } else {
            false
        }
    }

    /// Add a buffer slot.
    pub fn add_slot(&mut self, slot: BufferSlot) -> usize {
        let id = self.slots.len();
        self.total_bytes += slot.size_bytes();
        self.slots.push(slot);
        id
    }

    /// Assign a node to a slot.
    pub fn assign(&mut self, node_id: NodeId, slot_id: usize, liveness: (usize, usize)) {
        self.allocation.insert(node_id, slot_id);
        self.liveness.insert(node_id, liveness);
    }
}

impl Default for StaticBufferPlan {
    fn default() -> Self {
        Self::new()
    }
}
