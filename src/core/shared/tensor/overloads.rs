//! Operator overloads for tensors.
//!
//! Provides `+`, `-`, `*`, `/` operators for all `Tensor<C: Context>`.

use anyhow::Result;

use crate::core::shared::graph::Op;

use super::context::Context;
use super::structure::Tensor;

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $op:expr) => {
        // Core implementation: &Tensor op &Tensor
        impl<'a, 'b, C: Context> std::ops::$trait<&'b Tensor<C>> for &'a Tensor<C> {
            type Output = Result<Tensor<C>>;

            fn $method(self, rhs: &'b Tensor<C>) -> Self::Output {
                self.binary_op(rhs, $op)
            }
        }

        // Tensor op &Tensor
        impl<'b, C: Context> std::ops::$trait<&'b Tensor<C>> for Tensor<C> {
            type Output = Result<Tensor<C>>;

            fn $method(self, rhs: &'b Tensor<C>) -> Self::Output {
                self.binary_op(rhs, $op)
            }
        }

        // &Tensor op Tensor
        impl<'a, C: Context> std::ops::$trait<Tensor<C>> for &'a Tensor<C> {
            type Output = Result<Tensor<C>>;

            fn $method(self, rhs: Tensor<C>) -> Self::Output {
                self.binary_op(&rhs, $op)
            }
        }

        // Tensor op Tensor
        impl<C: Context> std::ops::$trait<Tensor<C>> for Tensor<C> {
            type Output = Result<Tensor<C>>;

            fn $method(self, rhs: Tensor<C>) -> Self::Output {
                self.binary_op(&rhs, $op)
            }
        }
    };
}

impl_binop!(Add, add, Op::Add);
impl_binop!(Sub, sub, Op::Sub);
impl_binop!(Mul, mul, Op::Mul);
impl_binop!(Div, div, Op::Div);

// Unary negation
impl<C: Context> std::ops::Neg for Tensor<C> {
    type Output = Tensor<C>;

    fn neg(self) -> Self::Output {
        Tensor::neg(&self)
    }
}

impl<C: Context> std::ops::Neg for &Tensor<C> {
    type Output = Tensor<C>;

    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}
