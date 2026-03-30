//! Common utility functions for execution.

/// Convert a multi-dimensional index to a flat buffer offset.
///
/// Assumes row-major (C-style) layout.
#[inline]
pub(crate) fn idx_to_offset(index: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut off = 0usize;
    for d in (0..shape.len()).rev() {
        off += index[d] * stride;
        stride *= shape[d];
    }
    off
}
