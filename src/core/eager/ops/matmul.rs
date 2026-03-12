use anyhow::{bail, Result};
use std::{cmp::Ordering, iter::Sum, ops::Mul};

use crate::{
    core::eager::ETensor, core::errors::MatmulShapeError, core::iters::Slicer, core::shape::Shape,
};

impl<T> ETensor<T>
where
    T: Copy + Mul<Output = T> + Sum<T> + Default,
{
    pub fn matmul_2d(&self, rhs: &ETensor<T>) -> Result<ETensor<T>> {
        let (n1, n2) = (self.shape.sizes[1], rhs.shape.sizes[0]);

        if n1 != n2 {
            bail!(MatmulShapeError::Matmul2d { n1, n2 });
        }

        let rhs = rhs.transpose(1, 0)?.to_contiguous()?;
        let (m, l) = (self.sizes()[0], rhs.sizes()[0]);

        let (lhs_iter, rhs_iter) = (
            Slicer::new(&self.shape.sizes, &[0], false),
            Slicer::new(&rhs.shape.sizes, &[0], false).collect::<Vec<_>>(),
        );
        let mut data = Vec::with_capacity(m * l);

        for lhs_slice in lhs_iter {
            let row = &self.slicer(&lhs_slice)?;

            for rhs_slice in rhs_iter.iter() {
                let column = rhs.slicer(rhs_slice)?;
                let prodsum = (row * column)?.sum()?;

                data.push(prodsum);
            }
        }

        ETensor::init(data, &[m, l])
    }

    pub fn matmul_nd(&self, rhs: &ETensor<T>) -> Result<ETensor<T>> {
        let (lhs_rank, rhs_rank) = (self.rank(), rhs.rank());

        let (lhs, rhs, max_dims) = match self.rank().cmp(&rhs.rank()) {
            Ordering::Equal => (self, rhs, lhs_rank),
            Ordering::Greater => (self, &rhs.unsqueeze(lhs_rank)?, lhs_rank),
            Ordering::Less => (&self.unsqueeze(rhs_rank)?, rhs, rhs_rank),
        };

        let (first, second) = (max_dims - 1, max_dims - 2);
        let (n1, n2) = (lhs.shape.sizes[first], rhs.shape.sizes[second]);

        if n1 != n2 {
            bail!(MatmulShapeError::MatmulNd { n1, n2 });
        }

        let rhs = rhs.transpose(second, first)?.to_contiguous()?;
        let (m, l) = (lhs.sizes()[second], rhs.sizes()[second]);
        let ml = m * l;

        let broadcast = &Shape::broadcast(&lhs.sizes()[..second], &rhs.sizes()[..second])?;
        let (lhs, rhs) = (
            lhs.expand(&[broadcast.as_slice(), &[m, n1]].concat())?
                .into_contiguous()?,
            rhs.expand(&[broadcast.as_slice(), &[l, n1]].concat())?
                .into_contiguous()?,
        );

        let slice_dim = &[second];
        let (lhs_iter, rhs_iter) = (
            Slicer::new(&lhs.shape.sizes, slice_dim, false),
            Slicer::new(&rhs.shape.sizes, slice_dim, false).collect::<Vec<_>>(),
        );

        let sizes = [broadcast.as_slice(), &[m, l]].concat();
        let mut data = vec![T::default(); sizes.iter().product()];

        for (lhs_i, lhs_slice) in lhs_iter.enumerate() {
            let row = &lhs.slicer(&lhs_slice)?;

            for (rhs_i, rhs_slice) in rhs_iter.iter().enumerate() {
                let column = rhs.slicer(rhs_slice)?;
                let prodsum = (row * column)?.sum_dims(&[first, second], true)?;

                for (iter_i, &value) in prodsum.data_contiguous().iter().enumerate() {
                    let offset = iter_i * ml + lhs_i * l + rhs_i;
                    data[offset] = value
                }
            }
        }

        ETensor::init(data, &sizes)
    }

    pub fn matmul(&self, rhs: &ETensor<T>) -> Result<ETensor<T>> {
        let (lhs_rank, rhs_rank) = (self.rank(), rhs.rank());

        if lhs_rank > 2 || rhs_rank > 2 {
            self.matmul_nd(rhs)
        } else if lhs_rank == 2 && rhs_rank == 2 {
            self.matmul_2d(rhs)
        } else if lhs_rank == 2 || rhs_rank == 2 {
            self.matmul_nd(rhs)
        } else if lhs_rank == 1 && rhs_rank == 1 {
            ETensor::scalar((self * rhs)?.sum()?)
        } else {
            Err(MatmulShapeError::Matmul0d.into())
        }
    }
}
