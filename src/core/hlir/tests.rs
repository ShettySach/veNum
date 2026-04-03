#[cfg(test)]
mod tests {
    use crate::core::hlir::optimize::{canonicalize_hlir, canonicalize_with_roots};
    use crate::core::hlir::{decompose, BufferId, DType, Dim, HLIRGraph, Op, ReduceOp, TensorType};

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

    #[test]
    fn canonicalize_eliminates_redundant_reshape() {
        let mut g = HLIRGraph::new();
        let a = g.load(
            BufferId(0),
            TensorType::contiguous(vec![Dim::Const(4), Dim::Const(1)], DType::F32),
        );
        let r1 = g.reshape(a, vec![Dim::Const(2), Dim::Const(2)]);
        let _r2 = g.reshape(r1, vec![Dim::Const(2), Dim::Const(2)]);

        let opt = canonicalize_hlir(&g);
        assert!(opt.len() < g.len());
    }

    #[test]
    fn canonicalize_sinks_reshapes_through_elementwise() {
        let mut g = HLIRGraph::new();
        let a = g.load(
            BufferId(0),
            TensorType::contiguous(vec![Dim::Const(4), Dim::Const(1)], DType::F32),
        );
        let b = g.load(
            BufferId(1),
            TensorType::contiguous(vec![Dim::Const(4), Dim::Const(1)], DType::F32),
        );
        let ar = g.reshape(a, vec![Dim::Const(2), Dim::Const(2)]);
        let br = g.reshape(b, vec![Dim::Const(2), Dim::Const(2)]);
        let add = g.binary(ar, br, Op::Add);

        let opt = canonicalize_with_roots(&g, &[add]);
        // Reshapes are sunk: 2 loads, 1 add (on original shape), 1 reshape = 4 nodes
        assert_eq!(opt.len(), 4);
        assert!(opt.topo_iter().any(|(_, n)| matches!(n.op, Op::Add(_, _))));
        // The reshape should come after the add, not before
        let nodes: Vec<_> = opt.topo_iter().collect();
        let add_idx = nodes.iter().position(|(_, n)| matches!(n.op, Op::Add(_, _))).unwrap();
        let reshape_idx = nodes.iter().position(|(_, n)| matches!(n.op, Op::Reshape { .. })).unwrap();
        assert!(add_idx < reshape_idx, "Add should come before Reshape");
    }
}
