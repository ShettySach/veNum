use anyhow::{bail, Result};
use std::cmp::max;

pub(crate) fn broadcast(lhs_sizes: &[usize], rhs_sizes: &[usize]) -> Result<Vec<usize>> {
    let mut lhs_iter = lhs_sizes.iter();
    let mut rhs_iter = rhs_sizes.iter();

    let max_len = max(lhs_sizes.len(), rhs_sizes.len());
    let mut result = Vec::with_capacity(max_len);

    loop {
        match (lhs_iter.next_back(), rhs_iter.next_back()) {
            (Some(&l), Some(&r)) => {
                if l == r {
                    result.push(l);
                } else if l == 1 {
                    result.push(r);
                } else if r == 1 {
                    result.push(l);
                } else {
                    bail!(
                        "shapes {:?} and {:?} cannot be broadcast together",
                        lhs_sizes,
                        rhs_sizes
                    );
                }
            }
            (Some(&l), None) => result.push(l),
            (None, Some(&r)) => result.push(r),
            (None, None) => break,
        }
    }

    result.reverse();
    Ok(result)
}
