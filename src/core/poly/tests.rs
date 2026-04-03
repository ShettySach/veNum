#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::core::hlir::{BufferId, DType, Dim, HLIRGraph, Op, TensorType};
    use crate::core::lower::lower;
    use crate::core::poly::{shape_to_domain, strides_to_access, NativeDependenceAnalyzer};
    use crate::core::schedule::search::{ScheduleSearcher, TrivialHardware};
    use crate::core::traits::{DependenceAnalyzer, ScheduleTransform};

    #[test]
    fn shape_to_domain_emits_two_constraints_per_dim() {
        let domain = shape_to_domain(&[Dim::Const(8), Dim::Const(16)]);
        assert_eq!(domain.iters, vec!["i0", "i1"]);
        assert_eq!(domain.constraints.len(), 4);
    }

    #[test]
    fn strides_to_access_rejects_symbolic_stride() {
        let err = strides_to_access(&[Dim::Sym(0_u32.into())], &["i0".to_owned()]).unwrap_err();
        assert!(err.to_string().contains("symbolic stride"));
    }

    #[test]
    fn native_analyzer_finds_simple_dependence() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let decision = ScheduleSearcher::new(TrivialHardware).search_best(&g)?.0;
        let analyzer = NativeDependenceAnalyzer;
        let llir = lower(&g, &decision, &analyzer)?;
        let deps = analyzer.analyze_kernel(llir.kernels.last().expect("kernel must exist"))?;
        assert!(!deps.is_empty());
        Ok(())
    }

    #[test]
    fn native_legality_rejects_parallel_positive_distance() -> Result<()> {
        let dep = crate::core::llir::Dependence {
            from: 0,
            to: 1,
            kind: crate::core::llir::DepKind::Raw,
            distance: Some(vec![1]),
            relation: crate::core::llir::DependenceRelation {
                source_vars: vec!["i0".to_owned()],
                sink_vars: vec!["i0".to_owned()],
                constraints: vec![],
            },
        };

        let analyzer = NativeDependenceAnalyzer;
        let legal = analyzer.check_legality(
            &[dep],
            &ScheduleTransform::Parallelize {
                loop_var: "i0".to_owned(),
            },
        )?;
        assert!(!legal);
        Ok(())
    }
}
