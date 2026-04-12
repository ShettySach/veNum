#[cfg(test)]
mod hlir_tests {
    use crate::core::hlir::optimize::{canonicalize_hlir, canonicalize_with_roots};
    use crate::core::hlir::{
        decompose, BufferId, DType, Dim, HLIRGraph, Op, ReduceOp, Scalar, TensorType,
    };

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
    fn broadcast_adds_leading_and_singleton_zero_strides() {
        let mut g = HLIRGraph::new();
        let input = g.load(
            BufferId(0),
            TensorType::contiguous(vec![Dim::Const(1), Dim::Const(4)], DType::F32),
        );
        let broadcasted = g.broadcast(input, vec![Dim::Const(2), Dim::Const(4)]);

        assert!(matches!(g.node(broadcasted).op, Op::Broadcast { .. }));
        assert_eq!(
            g.ty(broadcasted).strides(),
            vec![Dim::Const(0), Dim::Const(1)]
        );
    }

    #[test]
    fn broadcast_can_increase_rank() {
        let mut g = HLIRGraph::new();
        let scalar = g.constant(Scalar::F32(2.0), vec![], DType::F32);
        let broadcasted = g.broadcast(scalar, vec![Dim::Const(2), Dim::Const(3)]);

        assert!(matches!(g.node(broadcasted).op, Op::Broadcast { .. }));
        assert_eq!(g.ty(broadcasted).shape, vec![Dim::Const(2), Dim::Const(3)]);
        assert_eq!(
            g.ty(broadcasted).strides(),
            vec![Dim::Const(0), Dim::Const(0)]
        );
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
        let add_idx = nodes
            .iter()
            .position(|(_, n)| matches!(n.op, Op::Add(_, _)))
            .unwrap();
        let reshape_idx = nodes
            .iter()
            .position(|(_, n)| matches!(n.op, Op::Reshape { .. }))
            .unwrap();
        assert!(add_idx < reshape_idx, "Add should come before Reshape");
    }

    #[test]
    fn canonicalize_combines_linear_like_terms() {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(5)], DType::F32);
        let x = g.load(BufferId(0), ty);
        let two = g.constant(Scalar::F32(2.0), vec![], DType::F32);

        let x2 = g.binary(x, two, Op::Mul);
        let left = g.binary(x2, x, Op::Add);
        let root = g.binary(left, left, Op::Add);

        let opt = canonicalize_with_roots(&g, &[root]);
        let mul_count = opt
            .topo_iter()
            .filter(|(_, node)| matches!(node.op, Op::Mul(_, _)))
            .count();
        let add_count = opt
            .topo_iter()
            .filter(|(_, node)| matches!(node.op, Op::Add(_, _)))
            .count();

        assert_eq!(mul_count, 1, "expected a single Mul after canonicalization");
        assert_eq!(add_count, 0, "expected all Add nodes to be folded away");
    }
}
