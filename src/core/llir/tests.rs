#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::core::hlir::{BufferId, DType, Dim, HLIRGraph, Op, ReduceOp, TensorType};
    use crate::core::llir::{LoopKind, Var};
    use crate::core::lower::lower;
    use crate::core::schedule::search::{ScheduleSearcher, TrivialHardware};

    #[test]
    fn affine_arithmetic_works() {
        let a = crate::core::llir::AffineExpr::constant(2).with_term(3, Var::Loop("i".to_owned()));
        let b = crate::core::llir::AffineExpr::constant(1).with_term(4, Var::Loop("j".to_owned()));
        let c = a.add(&b);
        assert_eq!(c.constant, 3);
        assert_eq!(c.terms.len(), 2);
    }

    #[test]
    fn lower_add_load_load_emits_one_loop_nest() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(16)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let llir = lower(&g, &decision)?;
        assert_eq!(llir.kernels.len(), 3);
        let k = llir.kernels.last().unwrap();
        assert_eq!(k.loop_nest.loops.len(), 1);
        assert!(matches!(k.loop_nest.loops[0].kind, LoopKind::Sequential));
        Ok(())
    }

    #[test]
    fn lower_reduce_sum_emits_reduce_loop() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4), Dim::Const(8)], DType::F32);
        let a = g.load(BufferId(0), ty);
        let _r = g.reduce(a, vec![1], ReduceOp::Sum, false);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let llir = lower(&g, &decision)?;
        let k = llir.kernels.last().unwrap();
        assert!(k
            .loop_nest
            .loops
            .iter()
            .any(|lp| matches!(lp.kind, LoopKind::Reduce { .. })));
        Ok(())
    }
}
