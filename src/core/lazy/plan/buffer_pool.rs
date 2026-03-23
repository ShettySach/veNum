use std::collections::HashMap;

use crate::core::shared::dtype::DType;

/// Key for pooled buffer lookup: dtype + element count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PoolKey {
    dtype: DType,
    numel: usize,
}

/// A pool of reusable byte buffers to reduce allocation churn during
/// repeated plan execution.
///
/// Buffers are keyed by `(DType, numel)`. When a buffer is returned to
/// the pool, it becomes available for the next request with matching key.
pub struct BufferPool {
    pool: HashMap<PoolKey, Vec<Vec<u8>>>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            pool: HashMap::new(),
        }
    }

    /// Acquire a zeroed byte buffer for `numel` elements of `dtype`.
    pub fn acquire(&mut self, dtype: DType, numel: usize) -> Vec<u8> {
        let key = PoolKey { dtype, numel };
        if let Some(stack) = self.pool.get_mut(&key) {
            if let Some(buf) = stack.pop() {
                return buf;
            }
        }
        vec![0u8; numel * dtype.size_bytes()]
    }
}
