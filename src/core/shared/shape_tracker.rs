#[derive(Clone, Debug)]
pub struct ShapeTracker {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: isize,
}

fn row_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides: Vec<isize> = shape
        .iter()
        .rev()
        .scan(1, |acc, &s| {
            let stride = *acc;
            *acc *= s as isize;
            Some(stride)
        })
        .collect();
    strides.reverse();
    strides
}

impl ShapeTracker {
    /// Create a new contiguous row-major tracker for the given shape.
    pub fn contiguous(shape: &[usize]) -> Self {
        let strides = row_major_strides(shape);
        ShapeTracker {
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }

    /// Reshape: recompute strides from new shape. Only valid if contiguous.
    pub fn reshape(&self, new_shape: &[usize]) -> Option<ShapeTracker> {
        if !self.is_contiguous() {
            return None;
        }
        let strides = row_major_strides(new_shape);
        Some(ShapeTracker {
            shape: new_shape.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    /// Expand: set stride to 0 on dimensions that are broadcast from size 1.
    pub fn expand(&self, new_shape: &[usize]) -> Option<ShapeTracker> {
        if new_shape.len() != self.shape.len() {
            return None;
        }
        let mut strides = self.strides.clone();
        for (i, (&old, &new)) in self.shape.iter().zip(new_shape).enumerate() {
            if old == 1 && new != 1 {
                strides[i] = 0;
            } else if old != new {
                return None;
            }
        }
        Some(ShapeTracker {
            shape: new_shape.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    /// Permute: reorder shape and strides.
    pub fn permute(&self, axes: &[usize]) -> Option<ShapeTracker> {
        if axes.len() != self.shape.len() {
            return None;
        }
        let shape = axes.iter().map(|&a| self.shape[a]).collect();
        let strides = axes.iter().map(|&a| self.strides[a]).collect();
        Some(ShapeTracker {
            shape,
            strides,
            offset: self.offset,
        })
    }

    /// Transpose: swap two dimensions.
    pub fn transpose(&self, d1: usize, d2: usize) -> Option<ShapeTracker> {
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(d1, d2);
        strides.swap(d1, d2);
        Some(ShapeTracker {
            shape,
            strides,
            offset: self.offset,
        })
    }

    /// Squeeze: remove dimensions of size 1.
    pub fn squeeze(&self) -> ShapeTracker {
        let (shape, strides): (Vec<usize>, Vec<isize>) = self
            .shape
            .iter()
            .zip(&self.strides)
            .filter(|(&s, _)| s != 1)
            .map(|(&s, &st)| (s, st))
            .unzip();

        if shape.is_empty() {
            ShapeTracker {
                shape: vec![1],
                strides: vec![1],
                offset: self.offset,
            }
        } else {
            ShapeTracker {
                shape,
                strides,
                offset: self.offset,
            }
        }
    }

    /// Unsqueeze: prepend dimensions of size 1 to reach `new_rank`.
    pub fn unsqueeze(&self, new_rank: usize) -> Option<ShapeTracker> {
        if new_rank < self.shape.len() {
            return None;
        }
        let extra = new_rank - self.shape.len();
        let mut shape = vec![1; extra];
        shape.extend_from_slice(&self.shape);
        // Stride for size-1 dims doesn't matter (index is always 0), use 0.
        let mut strides = vec![0isize; extra];
        strides.extend_from_slice(&self.strides);
        Some(ShapeTracker {
            shape,
            strides,
            offset: self.offset,
        })
    }

    /// Flip: negate strides on flipped dimensions and adjust offset.
    pub fn flip(&self, dims: &[usize]) -> ShapeTracker {
        let mut strides = self.strides.clone();
        let mut offset = self.offset;
        for &d in dims {
            offset += (self.shape[d] as isize - 1) * strides[d];
            strides[d] = -strides[d];
        }
        ShapeTracker {
            shape: self.shape.clone(),
            strides,
            offset,
        }
    }

    /// Compute the flat buffer index for a logical multi-dim index.
    pub fn index(&self, logical_idx: &[usize]) -> usize {
        let mut off = self.offset;
        for (i, &idx) in logical_idx.iter().enumerate() {
            off += idx as isize * self.strides[i];
        }
        off as usize
    }

    /// Check if this tracker represents a contiguous row-major layout.
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let expected = row_major_strides(&self.shape);
        self.strides == expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_1d() {
        let st = ShapeTracker::contiguous(&[6]);
        assert_eq!(st.strides, vec![1]);
        assert!(st.is_contiguous());
        assert_eq!(st.index(&[3]), 3);
    }

    #[test]
    fn contiguous_2d() {
        let st = ShapeTracker::contiguous(&[2, 3]);
        assert_eq!(st.strides, vec![3, 1]);
        assert!(st.is_contiguous());
        assert_eq!(st.index(&[1, 2]), 5);
    }

    #[test]
    fn reshape_contiguous() {
        let st = ShapeTracker::contiguous(&[6]);
        let st2 = st.reshape(&[2, 3]).unwrap();
        assert_eq!(st2.shape, vec![2, 3]);
        assert_eq!(st2.strides, vec![3, 1]);
        assert_eq!(st2.index(&[1, 0]), 3);
    }

    #[test]
    fn expand_broadcast() {
        let st = ShapeTracker::contiguous(&[3, 1]);
        let st2 = st.expand(&[3, 4]).unwrap();
        assert_eq!(st2.shape, vec![3, 4]);
        assert_eq!(st2.strides, vec![1, 0]);
        // All columns in a row read the same element.
        assert_eq!(st2.index(&[0, 0]), 0);
        assert_eq!(st2.index(&[0, 3]), 0);
        assert_eq!(st2.index(&[2, 0]), 2);
        assert_eq!(st2.index(&[2, 3]), 2);
    }

    #[test]
    fn permute_transpose() {
        let st = ShapeTracker::contiguous(&[2, 3]);
        let st2 = st.permute(&[1, 0]).unwrap();
        assert_eq!(st2.shape, vec![3, 2]);
        assert_eq!(st2.strides, vec![1, 3]);
        // (1, 0) in transposed = (0, 1) in original = offset 1
        assert_eq!(st2.index(&[1, 0]), 1);
    }

    #[test]
    fn squeeze_unsqueeze() {
        let st = ShapeTracker::contiguous(&[1, 3, 1]);
        let sq = st.squeeze();
        assert_eq!(sq.shape, vec![3]);
        assert_eq!(sq.strides, vec![1]);

        let usq = sq.unsqueeze(3).unwrap();
        assert_eq!(usq.shape, vec![1, 1, 3]);
    }

    #[test]
    fn flip_1d() {
        let st = ShapeTracker::contiguous(&[5]);
        let fl = st.flip(&[0]);
        assert_eq!(fl.index(&[0]), 4);
        assert_eq!(fl.index(&[4]), 0);
    }

    #[test]
    fn reshape_then_expand() {
        // Simulates Load [6] -> Reshape [3,1] -> Expand [3,4]
        let st = ShapeTracker::contiguous(&[6]);
        let st = st.reshape(&[3, 1]).unwrap();
        let st = st.expand(&[3, 4]).unwrap();
        assert_eq!(st.shape, vec![3, 4]);
        assert_eq!(st.strides, vec![1, 0]);
        assert!(!st.is_contiguous());
    }
}
