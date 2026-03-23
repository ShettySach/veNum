#[cfg(test)]
mod solid_tests {
    use crate::core::shared::dtype::{Buffer, DType};
    use crate::core::solid::compile::compile;
    use crate::core::solid::context::SolidContext;
    use crate::core::solid::tensor::Tensor;

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
    fn compile_and_execute_exp() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        let data = Buffer::from_f32_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let results = program.execute(&[&data]).unwrap();

        assert_eq!(results.len(), 1);
        let expected: Vec<f32> = [0.0_f32, 1.0, 2.0, 3.0].iter().map(|x| x.exp()).collect();
        assert_f32_close(results[0].as_f32(), &expected, 1e-5);
    }

    #[test]
    fn compile_and_execute_neg() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = input.neg();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        let data = Buffer::from_f32_vec(vec![1.0, -2.0, 3.0]);
        let results = program.execute(&[&data]).unwrap();

        assert_f32_close(results[0].as_f32(), &[-1.0, 2.0, -3.0], 1e-6);
    }

    #[test]
    fn compile_and_execute_sqrt() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.sqrt();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        let data = Buffer::from_f32_vec(vec![1.0, 4.0, 9.0, 16.0]);
        let results = program.execute(&[&data]).unwrap();

        assert_f32_close(results[0].as_f32(), &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn compile_and_execute_ln() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = input.ln();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        let data = Buffer::from_f32_vec(vec![1.0, std::f32::consts::E, 10.0]);
        let results = program.execute(&[&data]).unwrap();

        let expected: Vec<f32> = [1.0, std::f32::consts::E, 10.0]
            .iter()
            .map(|x| x.ln())
            .collect();
        assert_f32_close(results[0].as_f32(), &expected, 1e-5);
    }

    // --- Binary ops ---

    #[test]
    fn compile_and_execute_add() {
        let cx = SolidContext::new();
        let a = Tensor::placeholder(&cx, vec![4], DType::F32);
        let b = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = a.add(&b).unwrap();

        let program = compile(&cx, &[a.id(), b.id()], &[output.id()]).unwrap();

        let a_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b_data = Buffer::from_f32_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let results = program.execute(&[&a_data, &b_data]).unwrap();

        assert_f32_close(results[0].as_f32(), &[11.0, 22.0, 33.0, 44.0], 1e-6);
    }

    #[test]
    fn compile_and_execute_mul() {
        let cx = SolidContext::new();
        let a = Tensor::placeholder(&cx, vec![3], DType::F32);
        let b = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = a.mul(&b).unwrap();

        let program = compile(&cx, &[a.id(), b.id()], &[output.id()]).unwrap();

        let a_data = Buffer::from_f32_vec(vec![2.0, 3.0, 4.0]);
        let b_data = Buffer::from_f32_vec(vec![5.0, 6.0, 7.0]);
        let results = program.execute(&[&a_data, &b_data]).unwrap();

        assert_f32_close(results[0].as_f32(), &[10.0, 18.0, 28.0], 1e-6);
    }

    #[test]
    fn compile_and_execute_sub() {
        let cx = SolidContext::new();
        let a = Tensor::placeholder(&cx, vec![3], DType::F32);
        let b = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = a.sub(&b).unwrap();

        let program = compile(&cx, &[a.id(), b.id()], &[output.id()]).unwrap();

        let a_data = Buffer::from_f32_vec(vec![10.0, 20.0, 30.0]);
        let b_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]);
        let results = program.execute(&[&a_data, &b_data]).unwrap();

        assert_f32_close(results[0].as_f32(), &[9.0, 18.0, 27.0], 1e-6);
    }

    // --- Fused chains ---

    #[test]
    fn compile_and_execute_fused_chain() {
        // input -> exp -> neg (should fuse into one kernel)
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp().neg();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        let data = Buffer::from_f32_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let results = program.execute(&[&data]).unwrap();

        let expected: Vec<f32> = [0.0_f32, 1.0, 2.0, 3.0].iter().map(|x| -x.exp()).collect();
        assert_f32_close(results[0].as_f32(), &expected, 1e-5);
    }

    #[test]
    fn compile_and_execute_add_then_exp() {
        let cx = SolidContext::new();
        let a = Tensor::placeholder(&cx, vec![3], DType::F32);
        let b = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = a.add(&b).unwrap().exp();

        let program = compile(&cx, &[a.id(), b.id()], &[output.id()]).unwrap();

        let a_data = Buffer::from_f32_vec(vec![0.0, 0.0, 0.0]);
        let b_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]);
        let results = program.execute(&[&a_data, &b_data]).unwrap();

        let expected: Vec<f32> = [1.0_f32, 2.0, 3.0].iter().map(|x| x.exp()).collect();
        assert_f32_close(results[0].as_f32(), &expected, 1e-5);
    }

    // --- With constant data ---

    #[test]
    fn compile_with_constant_weights() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![3], DType::F32);
        let weights = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0], vec![3]);
        let output = input.mul(&weights).unwrap();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        let data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]);
        let results = program.execute(&[&data]).unwrap();

        assert_f32_close(results[0].as_f32(), &[2.0, 6.0, 12.0], 1e-6);
    }

    // --- Multi-dimensional ---

    #[test]
    fn compile_2d_exp() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![2, 3], DType::F32);
        let output = input.exp();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        let data = Buffer::from_f32_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let results = program.execute(&[&data]).unwrap();

        let expected: Vec<f32> = (0..6).map(|i| (i as f32).exp()).collect();
        assert_f32_close(results[0].as_f32(), &expected, 1e-4);
    }

    // --- Reduce ops ---

    #[test]
    fn compile_and_execute_reduce_sum() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![2, 3], DType::F32);
        let output = input.sum(&[1], false).unwrap();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        // [[1,2,3], [4,5,6]] -> sum along dim 1 -> [6, 15]
        let data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let results = program.execute(&[&data]).unwrap();

        assert_f32_close(results[0].as_f32(), &[6.0, 15.0], 1e-5);
    }

    // --- Program reuse ---

    #[test]
    fn program_can_be_executed_multiple_times() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![3], DType::F32);
        let output = input.exp();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        // First execution
        let data1 = Buffer::from_f32_vec(vec![0.0, 1.0, 2.0]);
        let results1 = program.execute(&[&data1]).unwrap();

        // Second execution with different data
        let data2 = Buffer::from_f32_vec(vec![3.0, 4.0, 5.0]);
        let results2 = program.execute(&[&data2]).unwrap();

        let expected1: Vec<f32> = [0.0_f32, 1.0, 2.0].iter().map(|x| x.exp()).collect();
        let expected2: Vec<f32> = [3.0_f32, 4.0, 5.0].iter().map(|x| x.exp()).collect();

        assert_f32_close(results1[0].as_f32(), &expected1, 1e-5);
        assert_f32_close(results2[0].as_f32(), &expected2, 1e-5);
    }

    // --- Validation errors ---

    #[test]
    fn execute_wrong_num_inputs_errors() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        // No inputs
        assert!(program.execute(&[]).is_err());

        // Too many inputs
        let data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(program.execute(&[&data, &data]).is_err());
    }

    #[test]
    fn execute_wrong_dtype_errors() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        let bad_data = Buffer::from_i32_vec(vec![1, 2, 3, 4]);
        assert!(program.execute(&[&bad_data]).is_err());
    }

    #[test]
    fn execute_wrong_size_errors() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        let bad_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]); // 3 != 4
        assert!(program.execute(&[&bad_data]).is_err());
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
    fn solid_matches_liquid_exp() {
        // Solid
        let solid_cx = SolidContext::new();
        let solid_input = Tensor::placeholder(&solid_cx, vec![4], DType::F32);
        let solid_output = solid_input.exp();
        let program = compile(&solid_cx, &[solid_input.id()], &[solid_output.id()]).unwrap();
        let data = Buffer::from_f32_vec(vec![0.5, 1.5, 2.5, 3.5]);
        let solid_results = program.execute(&[&data]).unwrap();

        // Liquid
        let liquid_cx = crate::core::liquid::Context::new();
        let liquid_t =
            crate::core::liquid::Tensor::from_slice(&liquid_cx, &[0.5, 1.5, 2.5, 3.5], vec![4]);
        let liquid_result = liquid_t.exp().unwrap().realize().unwrap();

        assert_f32_close(solid_results[0].as_f32(), liquid_result.data(), 1e-6);
    }

    #[test]
    fn solid_matches_liquid_add() {
        use std::ops::Add;

        // Solid
        let solid_cx = SolidContext::new();
        let a = Tensor::placeholder(&solid_cx, vec![3], DType::F32);
        let b = Tensor::placeholder(&solid_cx, vec![3], DType::F32);
        let output = a.add(&b).unwrap();
        let program = compile(&solid_cx, &[a.id(), b.id()], &[output.id()]).unwrap();
        let a_data = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0]);
        let b_data = Buffer::from_f32_vec(vec![4.0, 5.0, 6.0]);
        let solid_results = program.execute(&[&a_data, &b_data]).unwrap();

        // Liquid
        let liquid_cx = crate::core::liquid::Context::new();
        let la = crate::core::liquid::Tensor::from_slice(&liquid_cx, &[1.0, 2.0, 3.0], vec![3]);
        let lb = crate::core::liquid::Tensor::from_slice(&liquid_cx, &[4.0, 5.0, 6.0], vec![3]);
        let liquid_result = (&la).add(&lb).unwrap().realize().unwrap();

        assert_f32_close(solid_results[0].as_f32(), liquid_result.data(), 1e-6);
    }

    #[test]
    fn solid_matches_liquid_fused_chain() {
        use std::ops::Add;

        // add -> exp
        let solid_cx = SolidContext::new();
        let a = Tensor::placeholder(&solid_cx, vec![4], DType::F32);
        let b = Tensor::placeholder(&solid_cx, vec![4], DType::F32);
        let output = a.add(&b).unwrap().exp();
        let program = compile(&solid_cx, &[a.id(), b.id()], &[output.id()]).unwrap();
        let a_data = Buffer::from_f32_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let b_data = Buffer::from_f32_vec(vec![0.5, 0.6, 0.7, 0.8]);
        let solid_results = program.execute(&[&a_data, &b_data]).unwrap();

        // Liquid
        let liquid_cx = crate::core::liquid::Context::new();
        let la =
            crate::core::liquid::Tensor::from_slice(&liquid_cx, &[0.1, 0.2, 0.3, 0.4], vec![4]);
        let lb =
            crate::core::liquid::Tensor::from_slice(&liquid_cx, &[0.5, 0.6, 0.7, 0.8], vec![4]);
        let liquid_result = (&la).add(&lb).unwrap().exp().unwrap().realize().unwrap();

        assert_f32_close(solid_results[0].as_f32(), liquid_result.data(), 1e-5);
    }

    // --- Program metadata ---

    #[test]
    fn program_reports_metadata() {
        let cx = SolidContext::new();
        let input = Tensor::placeholder(&cx, vec![4], DType::F32);
        let output = input.exp().neg();

        let program = compile(&cx, &[input.id()], &[output.id()]).unwrap();

        assert_eq!(program.input_specs.len(), 1);
        assert_eq!(program.output_specs.len(), 1);
        assert_eq!(program.input_specs[0].shape, vec![4]);
        assert_eq!(program.input_specs[0].dtype, DType::F32);
        assert!(program.num_kernels() > 0);
        assert!(program.num_steps() > 0);
    }
}
