#[cfg(test)]
mod solid_tests {
    use crate::core::shared::dtype::{Buffer, DType};
    use crate::core::shared::tensor::Tensor;
    use crate::core::solid::compile::compile;
    use crate::core::solid::context::SolidContext;

    fn assert_f32_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {}: {} != {} (tol={})",
                i,
                a,
                e,
                tol
            );
        }
    }

    // --- Basic unary ops ---

    #[test]
    fn compile_and_execute_exp() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data = Buffer::from_f32_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let results = program.execute(&[&data])?;

        assert_eq!(results.len(), 1);
        let expected: Vec<f32> = [0.0_f32, 1.0, 2.0, 3.0].iter().map(|x| x.exp()).collect();
        assert_f32_close(results[0].as_f32(), &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn compile_and_execute_neg() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = input.neg();

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data = Buffer::from_f32_vec(vec![1.0, -2.0, 3.0]);
        let results = program.execute(&[&data])?;

        assert_f32_close(results[0].as_f32(), &[-1.0, 2.0, -3.0], 1e-6);
        Ok(())
    }

    #[test]
    fn compile_and_execute_sqrt() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.sqrt()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data = Buffer::from_f32_vec(vec![1.0, 4.0, 9.0, 16.0]);
        let results = program.execute(&[&data])?;

        assert_f32_close(results[0].as_f32(), &[1.0, 2.0, 3.0, 4.0], 1e-5);
        Ok(())
    }

    #[test]
    fn compile_and_execute_ln() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = input.ln()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data = Buffer::from_f32_vec(vec![1.0, std::f32::consts::E, 10.0]);
        let results = program.execute(&[&data])?;

        let expected: Vec<f32> = [1.0, std::f32::consts::E, 10.0]
            .iter()
            .map(|x| x.ln())
            .collect();
        assert_f32_close(results[0].as_f32(), &expected, 1e-5);
        Ok(())
    }

    // --- Binary ops ---

    #[test]
    fn compile_and_execute_add() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let a = Tensor::placeholder(&cx, vec![4], DType::F32);
        let b = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = a.add(&b)?;

        let program = compile(&cx, &[a.id(), b.id()], &[output.id()])?;

        let a_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b_data = Buffer::from_f32_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let results = program.execute(&[&a_data, &b_data])?;

        assert_f32_close(results[0].as_f32(), &[11.0, 22.0, 33.0, 44.0], 1e-6);
        Ok(())
    }

    #[test]
    fn compile_and_execute_mul() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let a = Tensor::placeholder(&cx, vec![3], DType::F32);
        let b = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = a.mul(&b)?;

        let program = compile(&cx, &[a.id(), b.id()], &[output.id()])?;

        let a_data = Buffer::from_f32_vec(vec![2.0, 3.0, 4.0]);
        let b_data = Buffer::from_f32_vec(vec![5.0, 6.0, 7.0]);
        let results = program.execute(&[&a_data, &b_data])?;

        assert_f32_close(results[0].as_f32(), &[10.0, 18.0, 28.0], 1e-6);
        Ok(())
    }

    #[test]
    fn compile_and_execute_sub() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let a = Tensor::placeholder(&cx, vec![3], DType::F32);
        let b = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = a.sub(&b)?;

        let program = compile(&cx, &[a.id(), b.id()], &[output.id()])?;

        let a_data = Buffer::from_f32_vec(vec![10.0, 20.0, 30.0]);
        let b_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]);
        let results = program.execute(&[&a_data, &b_data])?;

        assert_f32_close(results[0].as_f32(), &[9.0, 18.0, 27.0], 1e-6);
        Ok(())
    }

    // --- Fused chains ---

    #[test]
    fn compile_and_execute_fused_chain() -> anyhow::Result<()> {
        // input -> exp -> neg (should fuse into one kernel)
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp()?.neg();

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data = Buffer::from_f32_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let results = program.execute(&[&data])?;

        let expected: Vec<f32> = [0.0_f32, 1.0, 2.0, 3.0].iter().map(|x| -x.exp()).collect();
        assert_f32_close(results[0].as_f32(), &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn compile_and_execute_add_then_exp() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let a = Tensor::placeholder(&cx, vec![3], DType::F32);
        let b = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = a.add(&b)?.exp()?;

        let program = compile(&cx, &[a.id(), b.id()], &[output.id()])?;

        let a_data = Buffer::from_f32_vec(vec![0.0, 0.0, 0.0]);
        let b_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]);
        let results = program.execute(&[&a_data, &b_data])?;

        let expected: Vec<f32> = [1.0_f32, 2.0, 3.0].iter().map(|x| x.exp()).collect();
        assert_f32_close(results[0].as_f32(), &expected, 1e-5);
        Ok(())
    }

    // --- With constant data ---

    #[test]
    fn compile_with_constant_weights() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![3], DType::F32);
        let weights = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0], vec![3]);
        let output = input.mul(&weights)?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]);
        let results = program.execute(&[&data])?;

        assert_f32_close(results[0].as_f32(), &[2.0, 6.0, 12.0], 1e-6);
        Ok(())
    }

    // --- Multi-dimensional ---

    #[test]
    fn compile_2d_exp() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![2, 3], DType::F32);
        let output = input.exp()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data = Buffer::from_f32_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let results = program.execute(&[&data])?;

        let expected: Vec<f32> = (0..6).map(|i| (i as f32).exp()).collect();
        assert_f32_close(results[0].as_f32(), &expected, 1e-4);
        Ok(())
    }

    // --- Reduce ops ---

    #[test]
    fn compile_and_execute_reduce_sum() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![2, 3], DType::F32);
        let output = input.sum(&[1], false)?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        // [[1,2,3], [4,5,6]] -> sum along dim 1 -> [6, 15]
        let data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let results = program.execute(&[&data])?;

        assert_f32_close(results[0].as_f32(), &[6.0, 15.0], 1e-5);
        Ok(())
    }

    // --- Program reuse ---

    #[test]
    fn program_can_be_executed_multiple_times() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = input.exp()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        // First execution
        let data1 = Buffer::from_f32_vec(vec![0.0, 1.0, 2.0]);
        let results1 = program.execute(&[&data1])?;

        // Second execution with different data
        let data2 = Buffer::from_f32_vec(vec![3.0, 4.0, 5.0]);
        let results2 = program.execute(&[&data2])?;

        let expected1: Vec<f32> = [0.0_f32, 1.0, 2.0].iter().map(|x| x.exp()).collect();
        let expected2: Vec<f32> = [3.0_f32, 4.0, 5.0].iter().map(|x| x.exp()).collect();

        assert_f32_close(results1[0].as_f32(), &expected1, 1e-5);
        assert_f32_close(results2[0].as_f32(), &expected2, 1e-5);
        Ok(())
    }

    // --- Validation errors ---

    #[test]
    fn execute_wrong_num_inputs_errors() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        // No inputs
        assert!(program.execute(&[]).is_err());

        // Too many inputs
        let data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(program.execute(&[&data, &data]).is_err());
        Ok(())
    }

    #[test]
    fn execute_wrong_dtype_errors() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let bad_data = Buffer::from_i32_vec(vec![1, 2, 3, 4]);
        assert!(program.execute(&[&bad_data]).is_err());
        Ok(())
    }

    #[test]
    fn execute_wrong_size_errors() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let bad_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]); // 3 != 4
        assert!(program.execute(&[&bad_data]).is_err());
        Ok(())
    }

    #[test]
    fn compile_no_inputs_errors() {
        let cx = SolidContext::new();
        let t = Tensor::from_slice(&cx, &[1.0, 2.0], vec![2]);
        assert!(compile(&cx, &[], &[t.id()]).is_err());
    }

    #[test]
    fn compile_no_outputs_errors() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        assert!(compile(&cx, &[input.id()], &[]).is_err());
    }

    // --- Compare with Liquid results ---

    #[test]
    fn solid_matches_liquid_exp() -> anyhow::Result<()> {
        // Solid
        let solid_cx = SolidContext::new();
        let solid_input = Tensor::placeholder(&solid_cx, vec![4], DType::F32);
        let solid_output = solid_input.exp()?;
        let program = compile(&solid_cx, &[solid_input.id()], &[solid_output.id()])?;
        let data = Buffer::from_f32_vec(vec![0.5, 1.5, 2.5, 3.5]);
        let solid_results = program.execute(&[&data])?;

        // Liquid
        let liquid_cx = crate::core::liquid::LiquidContext::new();
        let liquid_t = Tensor::from_slice(&liquid_cx, &[0.5, 1.5, 2.5, 3.5], vec![4]);
        let liquid_result = liquid_t.exp()?.realize()?;

        assert_f32_close(solid_results[0].as_f32(), liquid_result.data(), 1e-6);
        Ok(())
    }

    #[test]
    fn solid_matches_liquid_add() -> anyhow::Result<()> {
        // Solid
        let solid_cx = SolidContext::new();
        let a = Tensor::placeholder(&solid_cx, vec![3], DType::F32);
        let b = Tensor::placeholder(&solid_cx, vec![3], DType::F32);
        let a_id = a.id();
        let b_id = b.id();
        let output = a.add(&b)?;
        let program = compile(&solid_cx, &[a_id, b_id], &[output.id()])?;
        let a_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]);
        let b_data = Buffer::from_f32_vec(vec![4.0, 5.0, 6.0]);
        let solid_results = program.execute(&[&a_data, &b_data])?;

        // Liquid
        let liquid_cx = crate::core::liquid::LiquidContext::new();
        let la = Tensor::from_slice(&liquid_cx, &[1.0, 2.0, 3.0], vec![3]);
        let lb = Tensor::from_slice(&liquid_cx, &[4.0, 5.0, 6.0], vec![3]);
        let liquid_result = la.add(&lb)?.realize()?;

        assert_f32_close(solid_results[0].as_f32(), liquid_result.data(), 1e-6);
        Ok(())
    }

    #[test]
    fn solid_matches_liquid_fused_chain() -> anyhow::Result<()> {
        // add -> exp
        let solid_cx = SolidContext::new();
        let a = Tensor::placeholder(&solid_cx, vec![4], DType::F32);
        let b = Tensor::placeholder(&solid_cx, vec![4], DType::F32);
        let a_id = a.id();
        let b_id = b.id();
        let output = a.add(&b)?.exp()?;
        let program = compile(&solid_cx, &[a_id, b_id], &[output.id()])?;
        let a_data = Buffer::from_f32_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let b_data = Buffer::from_f32_vec(vec![0.5, 0.6, 0.7, 0.8]);
        let solid_results = program.execute(&[&a_data, &b_data])?;

        // Liquid
        let liquid_cx = crate::core::liquid::LiquidContext::new();
        let la = Tensor::from_slice(&liquid_cx, &[0.1, 0.2, 0.3, 0.4], vec![4]);
        let lb = Tensor::from_slice(&liquid_cx, &[0.5, 0.6, 0.7, 0.8], vec![4]);
        let liquid_result = la.add(&lb)?.exp()?.realize()?;

        assert_f32_close(solid_results[0].as_f32(), liquid_result.data(), 1e-5);
        Ok(())
    }

    // --- Program metadata ---

    #[test]
    fn program_reports_metadata() -> anyhow::Result<()> {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp()?.neg();

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        assert_eq!(program.input_specs.len(), 1);
        assert_eq!(program.output_specs.len(), 1);
        assert_eq!(program.input_specs[0].shape, vec![4]);
        assert_eq!(program.input_specs[0].dtype, DType::F32);
        assert!(program.num_kernels() > 0);
        assert!(program.num_steps() > 0);
        Ok(())
    }
}
