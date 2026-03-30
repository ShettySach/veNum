//! Operator overloads for tensors.

use anyhow::Result;

use crate::core::graph::Op;

use super::structure::Tensor;

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<'a, 'b> std::ops::$trait<&'b Tensor> for &'a Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: &'b Tensor) -> Self::Output {
                self.binary_op(rhs, $op)
            }
        }

        impl<'b> std::ops::$trait<&'b Tensor> for Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: &'b Tensor) -> Self::Output {
                self.binary_op(rhs, $op)
            }
        }

        impl<'a> std::ops::$trait<Tensor> for &'a Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: Tensor) -> Self::Output {
                self.binary_op(&rhs, $op)
            }
        }

        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: Tensor) -> Self::Output {
                self.binary_op(&rhs, $op)
            }
        }
    };
}

impl_binop!(Add, add, Op::Add);
impl_binop!(Sub, sub, Op::Sub);
impl_binop!(Mul, mul, Op::Mul);
impl_binop!(Div, div, Op::Div);

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        Tensor::neg(&self)
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}
