#[cfg(test)]
mod tests {
    use crate::core::hlir::{decompose, DType, Dim, HLIRGraph, Op, ReduceOp, TensorType};

    #[test]
    fn hlir_graph_build_and_topo_iter() {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4)], DType::F32);
        let a = g.load(crate::core::hlir::BufferId(0), ty.clone());
        let b = g.load(crate::core::hlir::BufferId(1), ty);
        let c = g.binary(a, b, Op::Add);

        let ids: Vec<usize> = g.topo_iter().map(|(id, _)| id.0).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        assert!(matches!(g.node(c).op, Op::Add(_, _)));
    }

    #[test]
    fn decomposition_sub_div() {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(8)], DType::F32);
        let a = g.load(crate::core::hlir::BufferId(0), ty.clone());
        let b = g.load(crate::core::hlir::BufferId(1), ty);

        let s = decompose::sub(&mut g, a, b);
        let d = decompose::div(&mut g, a, b);

        assert!(matches!(g.node(s).op, Op::Add(_, _)));
        assert!(matches!(g.node(d).op, Op::Mul(_, _)));
    }

    #[test]
    fn decomposition_matmul_is_reduce_sum() {
        let mut g = HLIRGraph::new();
        let a_ty = TensorType::contiguous(vec![Dim::Const(2), Dim::Const(3)], DType::F32);
        let b_ty = TensorType::contiguous(vec![Dim::Const(3), Dim::Const(4)], DType::F32);
        let a = g.load(crate::core::hlir::BufferId(0), a_ty);
        let b = g.load(crate::core::hlir::BufferId(1), b_ty);

        let out = decompose::matmul(&mut g, a, b);
        assert!(matches!(
            g.node(out).op,
            Op::Reduce {
                op: ReduceOp::Sum,
                ..
            }
        ));
    }
}
