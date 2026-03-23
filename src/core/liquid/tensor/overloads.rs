use anyhow::Result;

use crate::core::liquid::tensor::Tensor;
use crate::core::shared::graph::Op;

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $op:expr) => {
        // Core implementation: &Tensor op &Tensor
        impl<'a, 'b> std::ops::$trait<&'b Tensor> for &'a Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: &'b Tensor) -> Self::Output {
                self.binary_op(rhs, $op)
            }
        }

        // Tensor op &Tensor
        impl<'b> std::ops::$trait<&'b Tensor> for Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: &'b Tensor) -> Self::Output {
                self.binary_op(rhs, $op)
            }
        }

        // &Tensor op Tensor
        impl<'a> std::ops::$trait<Tensor> for &'a Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: Tensor) -> Self::Output {
                self.binary_op(&rhs, $op)
            }
        }

        // Tensor op Tensor
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
