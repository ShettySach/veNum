#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::core::compile::{SearchConfig, compile};
    use crate::core::cpu::{Buffer, CpuCodeGenerator};
    use crate::core::dep::NoOpDependenceAnalyzer;
    use crate::core::hlir::{BufferId, DType, Dim, HLIRGraph, Op, TensorType};
    use crate::core::schedule::TrivialHardware;

    #[test]
    fn end_to_end_add_pipeline_works() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let c = g.binary(a, b, Op::Add);

        let cg = CpuCodeGenerator::new(vec![c]);
        let module = compile(
            g,
            &TrivialHardware,
            &NoOpDependenceAnalyzer,
            &cg,
            &SearchConfig::default(),
            &[c],
        )?;

        let out = module.execute(&[
            (BufferId(0), Buffer::F32(vec![1.0, 2.0, 3.0, 4.0])),
            (BufferId(1), Buffer::F32(vec![10.0, 20.0, 30.0, 40.0])),
        ])?;

        match &out[0] {
            Buffer::F32(v) => assert_eq!(v, &vec![11.0, 22.0, 33.0, 44.0]),
            _ => panic!("unexpected output dtype"),
        }

        Ok(())
    }

    #[test]
    fn end_to_end_mul_reduce_pipeline_works() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(2), Dim::Const(3)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let m = g.binary(a, b, Op::Mul);
        let r = g.reduce(m, vec![1], crate::core::hlir::ReduceOp::Sum, false);

        let cg = CpuCodeGenerator::new(vec![r]);
        let module = compile(
            g,
            &TrivialHardware,
            &NoOpDependenceAnalyzer,
            &cg,
            &SearchConfig::default(),
            &[r],
        )?;

        let out = module.execute(&[
            (BufferId(0), Buffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            (BufferId(1), Buffer::F32(vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5])),
        ])?;

        match &out[0] {
            Buffer::F32(v) => assert_eq!(v, &vec![6.0, 7.5]),
            _ => panic!("unexpected output dtype"),
        }
        Ok(())
    }

    #[test]
    fn end_to_end_matmul_decomposition_works() -> Result<()> {
        let mut g = HLIRGraph::new();
        let a = g.load(
            BufferId(0),
            TensorType::contiguous(vec![Dim::Const(2), Dim::Const(3)], DType::F32),
        );
        let b = g.load(
            BufferId(1),
            TensorType::contiguous(vec![Dim::Const(3), Dim::Const(2)], DType::F32),
        );
        let c = crate::core::hlir::decompose::matmul(&mut g, a, b);

        let cg = CpuCodeGenerator::new(vec![c]);
        let module = compile(
            g,
            &TrivialHardware,
            &NoOpDependenceAnalyzer,
            &cg,
            &SearchConfig::default(),
            &[c],
        )?;

        let out = module.execute(&[
            (BufferId(0), Buffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            (BufferId(1), Buffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ])?;

        match &out[0] {
            Buffer::F32(v) => {
                let expected = vec![22.0, 28.0, 49.0, 64.0];
                assert_eq!(v, &expected);
            }
            _ => panic!("unexpected output dtype"),
        }
        Ok(())
    }

    #[test]
    fn end_to_end_cmp_where_pipeline_works() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());
        let b = g.load(BufferId(1), ty);
        let cond = g.cmp(crate::core::hlir::op::CmpOp::Gt, a, b);
        let out_node = g.where_select(cond, a, b);

        let cg = CpuCodeGenerator::new(vec![out_node]);
        let module = compile(
            g,
            &TrivialHardware,
            &NoOpDependenceAnalyzer,
            &cg,
            &SearchConfig::default(),
            &[out_node],
        )?;

        let out = module.execute(&[
            (BufferId(0), Buffer::F32(vec![1.0, 5.0, 2.0, 9.0])),
            (BufferId(1), Buffer::F32(vec![3.0, 4.0, 7.0, 8.0])),
        ])?;

        match &out[0] {
            Buffer::F32(v) => assert_eq!(v, &vec![3.0, 5.0, 7.0, 9.0]),
            _ => panic!("unexpected output dtype"),
        }
        Ok(())
    }

    #[test]
    fn end_to_end_conv2d_pipeline_works() -> Result<()> {
        let cx = crate::core::tensor::Context::new();
        let input = crate::core::tensor::Tensor::placeholder(&cx, DType::F32, vec![1, 1, 5, 5]);
        let weight = crate::core::tensor::Tensor::placeholder(&cx, DType::F32, vec![1, 1, 3, 3]);
        let out = input.conv2d(&weight)?;

        let result = crate::core::runner::run_context(
            &cx,
            &[out.id()],
            &[
                Buffer::F32((1..=25).map(|v| v as f32).collect()),
                Buffer::F32(vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0]),
            ],
        )?;

        match &result[0] {
            Buffer::F32(v) => {
                assert_eq!(v.len(), 9);
                assert_eq!(
                    v,
                    &vec![-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]
                );
            }
            _ => panic!("unexpected output dtype"),
        }

        Ok(())
    }

    #[test]
    fn end_to_end_conv2d_center_kernel_preserves_spatial_variation() -> Result<()> {
        let cx = crate::core::tensor::Context::new();
        let input = crate::core::tensor::Tensor::placeholder(&cx, DType::F32, vec![1, 1, 5, 5]);
        let weight = crate::core::tensor::Tensor::placeholder(&cx, DType::F32, vec![1, 1, 3, 3]);
        let out = input.conv2d(&weight)?;

        let result = crate::core::runner::run_context(
            &cx,
            &[out.id()],
            &[
                Buffer::F32((1..=25).map(|v| v as f32).collect()),
                Buffer::F32(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ],
        )?;

        match &result[0] {
            Buffer::F32(v) => {
                assert_eq!(v.len(), 9);
                assert_eq!(v, &vec![7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 17.0, 18.0, 19.0]);
            }
            _ => panic!("unexpected output dtype"),
        }

        Ok(())
    }

    #[test]
    fn canonicalization_can_be_disabled() -> Result<()> {
        let mut g = HLIRGraph::new();
        let ty = TensorType::contiguous(vec![Dim::Const(4)], DType::F32);
        let a = g.load(BufferId(0), ty.clone());

        // Create a redundant reshape chain that would be optimized away
        let r1 = g.reshape(a, vec![Dim::Const(2), Dim::Const(2)]);
        let r2 = g.reshape(r1, vec![Dim::Const(4)]);

        let config = SearchConfig {
            enable_canonicalization: false,
            ..SearchConfig::default()
        };

        let cg = CpuCodeGenerator::new(vec![r2]);
        let module = compile(
            g,
            &TrivialHardware,
            &NoOpDependenceAnalyzer,
            &cg,
            &config,
            &[r2],
        )?;

        let out = module.execute(&[(BufferId(0), Buffer::F32(vec![1.0, 2.0, 3.0, 4.0]))])?;

        match &out[0] {
            Buffer::F32(v) => assert_eq!(v, &vec![1.0, 2.0, 3.0, 4.0]),
            _ => panic!("unexpected output dtype"),
        }

        Ok(())
    }
}
