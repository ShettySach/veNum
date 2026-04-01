//! Operator overloads for tensors.

use anyhow::Result;

use super::structure::Tensor;

macro_rules! impl_binop_method {
    ($trait:ident, $method:ident, $tensor_method:ident) => {
        impl<'a, 'b> std::ops::$trait<&'b Tensor> for &'a Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: &'b Tensor) -> Self::Output {
                Tensor::$tensor_method(self, rhs)
            }
        }

        impl<'b> std::ops::$trait<&'b Tensor> for Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: &'b Tensor) -> Self::Output {
                Tensor::$tensor_method(&self, rhs)
            }
        }

        impl<'a> std::ops::$trait<Tensor> for &'a Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: Tensor) -> Self::Output {
                Tensor::$tensor_method(self, &rhs)
            }
        }

        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Result<Tensor>;

            fn $method(self, rhs: Tensor) -> Self::Output {
                Tensor::$tensor_method(&self, &rhs)
            }
        }
    };
}

impl_binop_method!(Add, add, add);
impl_binop_method!(Sub, sub, sub);
impl_binop_method!(Mul, mul, mul);
impl_binop_method!(Div, div, div);

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
