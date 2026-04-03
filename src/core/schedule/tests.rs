#[cfg(test)]
mod tests {
    use crate::core::cost::CpuHardwareModel;
    use crate::core::hlir::{BufferId, DType, Dim, HLIRGraph, Op, TensorType};
    use crate::core::schedule::decision::{FusionGroup, FusionGroupId, FusionTopology};
    use crate::core::schedule::fusion::{
        fusion_groups_form_acyclic_partition, is_fusion_group_legal,
    };
    use crate::core::schedule::search::{
        HardwareModel, KernelContext, ScheduleSearcher, TrivialHardware,
    };

    #[test]
    fn trivial_search_creates_one_group_per_node() {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Add);

        let searcher = ScheduleSearcher::new(TrivialHardware);
        let (decision, _cost) = searcher.search_best(&g).unwrap();
        assert_eq!(decision.fusion_groups.len(), 3);
        assert!(decision.fusion_groups.iter().all(|fg| fg.nodes.len() == 1));
        assert!(decision.opts.values().all(|v| v.is_empty()));
    }

    #[test]
    fn cpu_hardware_candidates_generate_nonempty_opts() {
        let hw = CpuHardwareModel::default();
        let opts = hw.opt_candidates(&KernelContext {
            loop_bounds: vec![64, 64],
            reduce_axes: vec![1],
            dtype: DType::F32,
            shared_budget: 0,
            backend: crate::core::schedule::BackendClass::Cpu,
        });
        assert!(!opts.is_empty());
    }

    #[test]
    fn beam_search_returns_ranked_candidates() {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(64), Dim::Const(64)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let _c = g.binary(a, b, Op::Mul);

        let mut searcher = ScheduleSearcher::new(CpuHardwareModel::default());
        searcher.beam_width = 4;
        searcher.max_iterations = 2;

        let ranked = searcher.search(&g);
        assert!(!ranked.is_empty());
        assert!(ranked.len() <= 4);
        for i in 1..ranked.len() {
            assert!(ranked[i - 1].1.total_cycles <= ranked[i].1.total_cycles);
        }
    }

    #[test]
    fn fusion_legality_rejects_duplicate_partition_nodes() {
        let g1 = FusionGroup {
            id: FusionGroupId(0),
            nodes: vec![crate::core::hlir::NodeId(1)],
            topology: FusionTopology::Chain,
        };
        let g2 = FusionGroup {
            id: FusionGroupId(1),
            nodes: vec![crate::core::hlir::NodeId(1)],
            topology: FusionTopology::Chain,
        };
        assert!(!fusion_groups_form_acyclic_partition(&[g1, g2]));
    }

    #[test]
    fn fusion_group_legal_on_simple_chain() {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let c = g.binary(a, b, Op::Add);

        let fg = FusionGroup {
            id: FusionGroupId(0),
            nodes: vec![a, b, c],
            topology: FusionTopology::Chain,
        };
        assert!(is_fusion_group_legal(&g, &fg));
    }
}
