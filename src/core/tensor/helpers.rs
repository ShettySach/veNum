//! Helper functions for tensor operations.

use anyhow::{anyhow, bail, Result};
use std::collections::HashSet;

use crate::core::hlir::Dim;

/// Compute shape after unsqueezing to a target rank (prepend 1s).
pub(crate) fn unsqueeze_shape(shape: &[Dim], target_rank: usize) -> Vec<Dim> {
    let mut new_shape = vec![Dim::Const(1); target_rank - shape.len()];
    new_shape.extend_from_slice(shape);
    new_shape
}

/// Broadcast batch dimensions for matmul.
pub(crate) fn broadcast_batch(a: &[Dim], b: &[Dim]) -> Result<Vec<Dim>> {
    let rank = a.len().max(b.len());
    let mut out = vec![Dim::Const(1); rank];
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else if let Dim::Const(v) = a[i - (rank - a.len())] {
            v
        } else {
            return Err(anyhow!("broadcast_batch requires Const dims"));
        };
        let db = if i < rank - b.len() {
            1
        } else if let Dim::Const(v) = b[i - (rank - b.len())] {
            v
        } else {
            return Err(anyhow!("broadcast_batch requires Const dims"));
        };
        if da != db && da != 1 && db != 1 {
            bail!("batch dimensions not broadcastable: {:?} vs {:?}", a, b);
        }
        out[i] = Dim::Const(da.max(db));
    }
    Ok(out)
}

/// Normalize axes from isize (supporting negative indexing) to usize.
pub(crate) fn normalize_axes(dims: &[isize], ndim: usize) -> Result<Vec<usize>> {
    let ndim_i = ndim as isize;
    dims.iter()
        .map(|&d| {
            let normalized = if d < 0 { ndim_i + d } else { d };
            if normalized < 0 || normalized >= ndim_i {
                bail!("axis {} out of bounds for tensor of rank {}", d, ndim);
            }
            Ok(normalized as usize)
        })
        .collect()
}

/// Compute the shape after reducing along specified dimensions.
pub(crate) fn compute_reduced_shape(shape: &[Dim], dims: &[usize], keepdims: bool) -> Vec<Dim> {
    let dims_set: HashSet<usize> = dims.iter().copied().collect();
    let mut new_shape = Vec::with_capacity(shape.len());
    for (i, s) in shape.iter().enumerate() {
        if dims_set.contains(&i) {
            if keepdims {
                new_shape.push(Dim::Const(1));
            }
        } else {
            new_shape.push(s.clone());
        }
    }
    // Ensure at least rank 1
    if new_shape.is_empty() {
        new_shape.push(Dim::Const(1));
    }
    new_shape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsqueeze_shape() {
        assert_eq!(
            unsqueeze_shape(&[Dim::Const(3), Dim::Const(4)], 4),
            vec![Dim::Const(1), Dim::Const(1), Dim::Const(3), Dim::Const(4)]
        );
        assert_eq!(
            unsqueeze_shape(&[Dim::Const(3), Dim::Const(4)], 2),
            vec![Dim::Const(3), Dim::Const(4)]
        );
        assert_eq!(unsqueeze_shape(&[], 2), vec![Dim::Const(1), Dim::Const(1)]);
    }

    #[test]
    fn test_broadcast_batch() {
        assert_eq!(
            broadcast_batch(&[Dim::Const(2), Dim::Const(3)], &[Dim::Const(3)]).unwrap(),
            vec![Dim::Const(2), Dim::Const(3)]
        );
        assert_eq!(
            broadcast_batch(
                &[Dim::Const(1), Dim::Const(3)],
                &[Dim::Const(2), Dim::Const(1)]
            )
            .unwrap(),
            vec![Dim::Const(2), Dim::Const(3)]
        );
        assert_eq!(
            broadcast_batch(&[], &[Dim::Const(2), Dim::Const(3)]).unwrap(),
            vec![Dim::Const(2), Dim::Const(3)]
        );
        assert!(broadcast_batch(
            &[Dim::Const(2), Dim::Const(3)],
            &[Dim::Const(3), Dim::Const(4)]
        )
        .is_err());
    }

    #[test]
    fn test_normalize_axes() {
        assert_eq!(normalize_axes(&[0, 1], 3).unwrap(), vec![0, 1]);
        assert_eq!(normalize_axes(&[-1], 3).unwrap(), vec![2]);
        assert_eq!(normalize_axes(&[-2, -1], 3).unwrap(), vec![1, 2]);
        assert!(normalize_axes(&[3], 3).is_err());
        assert!(normalize_axes(&[-4], 3).is_err());
    }

    #[test]
    fn test_compute_reduced_shape() {
        assert_eq!(
            compute_reduced_shape(&[Dim::Const(2), Dim::Const(3), Dim::Const(4)], &[1], false),
            vec![Dim::Const(2), Dim::Const(4)]
        );
        assert_eq!(
            compute_reduced_shape(&[Dim::Const(2), Dim::Const(3), Dim::Const(4)], &[1], true),
            vec![Dim::Const(2), Dim::Const(1), Dim::Const(4)]
        );
        assert_eq!(
            compute_reduced_shape(
                &[Dim::Const(2), Dim::Const(3), Dim::Const(4)],
                &[0, 1, 2],
                false
            ),
            vec![Dim::Const(1)]
        );
        assert_eq!(
            compute_reduced_shape(
                &[Dim::Const(2), Dim::Const(3), Dim::Const(4)],
                &[0, 1, 2],
                true
            ),
            vec![Dim::Const(1), Dim::Const(1), Dim::Const(1)]
        );
    }
}
