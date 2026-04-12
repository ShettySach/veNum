//! Operator overloads for tensors.

use crate::core::hlir::Scalar;

use super::structure::Tensor;

macro_rules! impl_binop_method {
    ($trait:ident, $method:ident, $tensor_method:ident) => {
        impl<'a, 'b> std::ops::$trait<&'b Tensor> for &'a Tensor {
            type Output = Tensor;

            fn $method(self, rhs: &'b Tensor) -> Self::Output {
                Tensor::$tensor_method(self, rhs).unwrap()
            }
        }

        impl<'b> std::ops::$trait<&'b Tensor> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: &'b Tensor) -> Self::Output {
                Tensor::$tensor_method(&self, rhs).unwrap()
            }
        }

        impl<'a> std::ops::$trait<Tensor> for &'a Tensor {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Self::Output {
                Tensor::$tensor_method(self, &rhs).unwrap()
            }
        }

        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Self::Output {
                Tensor::$tensor_method(&self, &rhs).unwrap()
            }
        }
    };
}

impl_binop_method!(Add, add, add);
impl_binop_method!(Sub, sub, sub);
impl_binop_method!(Mul, mul, mul);
impl_binop_method!(Div, div, div);

macro_rules! impl_scalar_rhs_binop_method {
    ($trait:ident, $method:ident, $tensor_method:ident, $scalar_ty:ty) => {
        impl<'a> std::ops::$trait<$scalar_ty> for &'a Tensor {
            type Output = Tensor;

            fn $method(self, rhs: $scalar_ty) -> Self::Output {
                let rhs_scalar = Scalar::from_f64(rhs as f64, self.dtype());
                let rhs_tensor = Tensor::constant_scalar(self.context(), rhs_scalar, vec![]);
                Tensor::$tensor_method(self, &rhs_tensor).unwrap()
            }
        }

        impl std::ops::$trait<$scalar_ty> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: $scalar_ty) -> Self::Output {
                <&Tensor as std::ops::$trait<$scalar_ty>>::$method(&self, rhs)
            }
        }
    };
}

impl_scalar_rhs_binop_method!(Add, add, add, i32);

impl_scalar_rhs_binop_method!(Sub, sub, sub, i32);

impl_scalar_rhs_binop_method!(Mul, mul, mul, i32);

impl_scalar_rhs_binop_method!(Div, div, div, i32);

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

#[cfg(test)]
mod tests {
    use crate::core::hlir::Op;
    use crate::{Buffer, Context, DType, Tensor, run_context};

    fn count_ops(graph: &crate::core::hlir::HLIRGraph, name: &str) -> usize {
        graph
            .topo_iter()
            .filter(|(_, node)| node.op.name() == name)
            .count()
    }

    #[test]
    fn rhs_scalar_mul_executes() {
        let cx = Context::new();
        let x = Tensor::placeholder(&cx, DType::F32, vec![3]);
        let z = &x * 2;

        let outputs = run_context(&cx, &[z.id()], &[Buffer::F32(vec![1.0, 2.0, 3.0])])
            .expect("run_context should succeed");

        match &outputs[0] {
            Buffer::F32(v) => assert_eq!(v, &vec![2.0, 4.0, 6.0]),
            other => panic!("unexpected output buffer: {other:?}"),
        }
    }

    #[test]
    fn rhs_scalar_add_executes() {
        let cx = Context::new();
        let x = Tensor::placeholder(&cx, DType::F32, vec![3]);
        let z = &x + 2;

        let outputs = run_context(&cx, &[z.id()], &[Buffer::F32(vec![1.0, 2.0, 3.0])])
            .expect("run_context should succeed");

        match &outputs[0] {
            Buffer::F32(v) => assert_eq!(v, &vec![3.0, 4.0, 5.0]),
            other => panic!("unexpected output buffer: {other:?}"),
        }
    }

    #[test]
    fn rhs_scalar_mul_emits_broadcast_op() {
        let cx = Context::new();
        let x = Tensor::placeholder(&cx, DType::F32, vec![3]);
        let _z = &x * 2;

        let graph = cx
            .graph()
            .lock()
            .expect("Graph mutex should not be poisoned");
        assert_eq!(count_ops(&graph, "Broadcast"), 1);
        assert_eq!(count_ops(&graph, "Reshape"), 0);
        assert_eq!(count_ops(&graph, "Expand"), 0);

        let broadcast_node = graph
            .topo_iter()
            .find_map(|(_, node)| match node.op {
                Op::Broadcast { .. } => Some(node),
                _ => None,
            })
            .expect("expected a Broadcast node");
        assert_eq!(broadcast_node.ty.shape.len(), 1);
    }

    #[test]
    fn rhs_scalar_add_emits_broadcast_op() {
        let cx = Context::new();
        let x = Tensor::placeholder(&cx, DType::F32, vec![3]);
        let _z = &x + 2;

        let graph = cx
            .graph()
            .lock()
            .expect("Graph mutex should not be poisoned");
        assert_eq!(count_ops(&graph, "Broadcast"), 1);
        assert_eq!(count_ops(&graph, "Reshape"), 0);
        assert_eq!(count_ops(&graph, "Expand"), 0);
    }
}
