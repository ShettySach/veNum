#[cfg(test)]
mod solid_tests {
    use crate::core::dtype::{Buffer, DType};
    use crate::core::tensor::{Context, Tensor};
    use crate::core::compile::compile;

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
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![4]);
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
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![3]);
        let output = input.neg();

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data = Buffer::from_f32_vec(vec![1.0, -2.0, 3.0]);
        let results = program.execute(&[&data])?;

        assert_f32_close(results[0].as_f32(), &[-1.0, 2.0, -3.0], 1e-6);
        Ok(())
    }

    #[test]
    fn compile_and_execute_sqrt() -> anyhow::Result<()> {
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![4]);
        let output = input.sqrt()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data = Buffer::from_f32_vec(vec![1.0, 4.0, 9.0, 16.0]);
        let results = program.execute(&[&data])?;

        assert_f32_close(results[0].as_f32(), &[1.0, 2.0, 3.0, 4.0], 1e-5);
        Ok(())
    }

    #[test]
    fn compile_and_execute_ln() -> anyhow::Result<()> {
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![3]);
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
        let cx = Context::new();
        let a = Tensor::placeholder(&cx, DType::F32, vec![4]);
        let b = Tensor::placeholder(&cx, DType::F32, vec![4]);
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
        let cx = Context::new();
        let a = Tensor::placeholder(&cx, DType::F32, vec![3]);
        let b = Tensor::placeholder(&cx, DType::F32, vec![3]);
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
        let cx = Context::new();
        let a = Tensor::placeholder(&cx, DType::F32, vec![3]);
        let b = Tensor::placeholder(&cx, DType::F32, vec![3]);
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
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![4]);
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
        let cx = Context::new();
        let a = Tensor::placeholder(&cx, DType::F32, vec![3]);
        let b = Tensor::placeholder(&cx, DType::F32, vec![3]);
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
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![3]);
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
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![2, 3]);
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
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![2, 3]);
        let output = input.sum(&[1], false)?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let results = program.execute(&[&data])?;

        assert_f32_close(results[0].as_f32(), &[6.0, 15.0], 1e-5);
        Ok(())
    }

    // --- Program reuse ---

    #[test]
    fn program_can_be_executed_multiple_times() -> anyhow::Result<()> {
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![3]);
        let output = input.exp()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let data1 = Buffer::from_f32_vec(vec![0.0, 1.0, 2.0]);
        let results1 = program.execute(&[&data1])?;

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
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![4]);
        let output = input.exp()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        assert!(program.execute(&[]).is_err());

        let data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(program.execute(&[&data, &data]).is_err());
        Ok(())
    }

    #[test]
    fn execute_wrong_dtype_errors() -> anyhow::Result<()> {
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![4]);
        let output = input.exp()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let bad_data = Buffer::from_i32_vec(vec![1, 2, 3, 4]);
        assert!(program.execute(&[&bad_data]).is_err());
        Ok(())
    }

    #[test]
    fn execute_wrong_size_errors() -> anyhow::Result<()> {
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![4]);
        let output = input.exp()?;

        let program = compile(&cx, &[input.id()], &[output.id()])?;

        let bad_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]);
        assert!(program.execute(&[&bad_data]).is_err());
        Ok(())
    }

    #[test]
    fn compile_no_inputs_errors() {
        let cx = Context::new();
        let t = Tensor::from_slice(&cx, &[1.0, 2.0], vec![2]);
        assert!(compile(&cx, &[], &[t.id()]).is_err());
    }

    #[test]
    fn compile_no_outputs_errors() {
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![4]);
        assert!(compile(&cx, &[input.id()], &[]).is_err());
    }

    // --- Program metadata ---

    #[test]
    fn program_reports_metadata() -> anyhow::Result<()> {
        let cx = Context::new();
        let input = Tensor::placeholder(&cx, DType::F32, vec![4]);
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
