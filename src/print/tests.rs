#[cfg(test)]
mod print_tests {
    use crate::core::compile::SearchConfig;
    use crate::core::hlir::{DType, TensorType};
    use crate::core::tensor::{Context, Tensor};

    #[test]
    fn can_render_graph_schedule_and_llir_mermaid() {
        let cx = Context::new();
        let a = Tensor::placeholder(&cx, DType::F32, vec![4]);
        let b = Tensor::placeholder(&cx, DType::F32, vec![4]);
        let c = a.add(&b).expect("add should succeed");

        let g = cx.graph_mermaid();
        assert!(g.contains("graph TD"));

        let cfg = SearchConfig::default();
        let s = cx.schedule_mermaid(&[c.id()], &cfg);
        assert!(s.contains("flowchart LR"));

        let l = cx.llir_mermaid(&[c.id()], &cfg);
        assert!(l.contains("flowchart TD"));
    }

    #[test]
    fn can_render_optimized_graph_mermaid() {
        let cx = Context::new();
        let a = Tensor::placeholder(&cx, DType::F32, vec![2, 2]);
        let b = a.reshape(vec![4]).expect("reshape should succeed");
        let _ = b
            .reshape(vec![2, 2])
            .expect("second reshape should succeed");

        let m = cx.optimized_graph_mermaid(&[b.id()]);
        assert!(m.contains("graph TD"));
    }

    #[test]
    fn free_functions_render_types() {
        let cx = Context::new();
        let x = Tensor::placeholder(&cx, DType::F32, vec![8]);

        let graph = cx
            .graph()
            .lock()
            .expect("Graph mutex should not be poisoned")
            .clone();
        let g = crate::print::to_mermaid(&graph);
        assert!(g.contains("Load"));

        let schedule = crate::core::schedule::ScheduleDecision::new(vec![
            crate::core::schedule::FusionGroup {
                id: crate::core::schedule::FusionGroupId(0),
                nodes: vec![x.id()],
                topology: crate::core::schedule::FusionTopology::Chain,
            },
        ]);
        let s = crate::print::to_schedule_mermaid(&schedule);
        assert!(s.contains("FusionGroup"));

        let llir = crate::core::llir::LLIRProgram {
            kernels: vec![crate::core::llir::program::Kernel {
                id: crate::core::llir::program::KernelId(0),
                name: "k0".to_owned(),
                root: x.id(),
                op: crate::core::hlir::Op::Load {
                    buffer: crate::core::hlir::BufferId(0),
                },
                ty: TensorType::contiguous(vec![crate::core::hlir::Dim::Const(8)], DType::F32),
                loop_nest: crate::core::llir::loop_nest::LoopNest {
                    loops: vec![],
                    body: vec![],
                },
                allocs: vec![],
            }],
        };
        let l = crate::print::to_llir_mermaid(&llir);
        assert!(l.contains("Kernel"));
    }
}
