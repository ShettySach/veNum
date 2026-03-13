#[cfg(test)]
mod lazy_tests {
    use crate::{Context, Tensor};
    use anyhow::Result;

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
    }

    #[test]
    fn realize_leaf() -> Result<()> {
        let cx = Context::new();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let dt = Tensor::from_slice(&cx, &data, vec![4]);
        let result = dt.realize()?;

        assert_eq!(result.data(), &data);
        assert_eq!(result.shape(), &[4]);
        Ok(())
    }

    #[test]
    fn add_two_tensors() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let c = &a + &b;
        let result = c.realize()?;

        assert_eq!(result.data(), &[11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    #[test]
    fn mul_two_tensors() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 10.0, 10.0, 10.0], vec![4]);
        let c = &a * &b;
        let result = c.realize()?;

        assert_eq!(result.data(), &[20.0, 30.0, 40.0, 50.0]);
        Ok(())
    }

    #[test]
    fn sub_and_div() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);

        let sub_result = (&a - &b).realize()?;
        assert_eq!(sub_result.data(), &[9.0, 18.0, 27.0, 36.0]);

        let div_result = (&a / &b).realize()?;
        assert_eq!(div_result.data(), &[10.0, 10.0, 10.0, 10.0]);

        Ok(())
    }

    #[test]
    fn fused_add_mul() -> Result<()> {
        let cx = Context::new();

        // (a + b) * a should produce ONE fused kernel.
        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let c = &(&a + &b) * &a;
        let result = c.realize()?;

        // (1+10)*1=11, (2+20)*2=44, (3+30)*3=99, (4+40)*4=176
        assert_eq!(result.data(), &[11.0, 44.0, 99.0, 176.0]);
        Ok(())
    }

    #[test]
    fn unary_neg() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0, -4.0], vec![4]);
        let result = a.neg().realize()?;

        assert_eq!(result.data(), &[-1.0, 2.0, -3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn unary_exp() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[0.0, 1.0, 2.0], vec![3]);
        let result = a.exp().realize()?;

        let expected: Vec<f32> = [0.0f32, 1.0, 2.0].iter().map(|x| x.exp()).collect();
        assert!(approx_eq(result.data(), &expected, 1e-5));
        Ok(())
    }

    #[test]
    fn unary_ln() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.718_281_7, 7.389056], vec![3]);
        let result = a.ln().realize()?;

        let expected: Vec<f32> = [1.0f32, 2.718_281_7, 7.389056]
            .iter()
            .map(|x| x.ln())
            .collect();
        assert!(approx_eq(result.data(), &expected, 1e-4));
        Ok(())
    }

    #[test]
    fn unary_sqrt() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 4.0, 9.0, 16.0], vec![4]);
        let result = a.sqrt().realize()?;

        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn chain_fused_unary_binary() -> Result<()> {
        let cx = Context::new();

        // exp(a) + b — should fuse into one kernel.
        let a = Tensor::from_slice(&cx, &[0.0, 0.0, 0.0], vec![3]);
        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let c = &a.exp() + &b;
        let result = c.realize()?;

        // exp(0) + 1 = 2, exp(0) + 2 = 3, exp(0) + 3 = 4
        assert_eq!(result.data(), &[2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn from_slice_add() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let c = &a + &b;
        let result = c.realize()?;

        assert_eq!(result.data(), &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(result.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn larger_tensor() -> Result<()> {
        let cx = Context::new();

        let n = 1024;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

        let a = Tensor::from_slice(&cx, &a_data, vec![n]);
        let b = Tensor::from_slice(&cx, &b_data, vec![n]);
        let c = &a + &b;
        let result = c.realize()?;

        let expected: Vec<f32> = (0..n).map(|_| n as f32).collect();
        assert_eq!(result.data(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn deep_fusion_chain() -> Result<()> {
        let cx = Context::new();

        // a * b + a - b should all fuse into one kernel.
        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[1.0, 1.0, 1.0, 1.0], vec![4]);

        let c = &(&(&a * &b) + &a) - &b;
        let result = c.realize()?;

        // (2*1 + 2 - 1) = 3, (3*1 + 3 - 1) = 5, (4*1 + 4 - 1) = 7, (5*1 + 5 - 1) = 9
        assert_eq!(result.data(), &[3.0, 5.0, 7.0, 9.0]);
        Ok(())
    }

    #[test]
    fn constant_tensor() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let c = Tensor::constant(&cx, 10.0, vec![3]);
        let result = (&a + &c).realize()?;

        assert_eq!(result.data(), &[11.0, 12.0, 13.0]);
        Ok(())
    }

    // --- egglog optimization tests ---

    #[test]
    fn egglog_add_zero_identity() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let zero = Tensor::constant(&cx, 0.0, vec![3]);
        // a + 0 should be optimized to just a.
        let result = (&a + &zero).realize()?;
        assert_eq!(result.data(), &[1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn egglog_mul_one_identity() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        let one = Tensor::constant(&cx, 1.0, vec![3]);
        // a * 1 should be optimized to just a.
        let result = (&a * &one).realize()?;
        assert_eq!(result.data(), &[5.0, 10.0, 15.0]);
        Ok(())
    }

    #[test]
    fn egglog_mul_zero() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        let zero = Tensor::constant(&cx, 0.0, vec![3]);
        // a * 0 should be optimized to 0.
        let result = (&a * &zero).realize()?;
        assert_eq!(result.data(), &[0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn egglog_double_neg() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0], vec![3]);
        // neg(neg(a)) should be optimized to just a.
        let result = a.neg().neg().realize()?;
        assert_eq!(result.data(), &[1.0, -2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn egglog_exp_ln_inverse() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        // exp(ln(a)) should be optimized to just a.
        let result = a.ln().exp().realize()?;
        assert!(approx_eq(result.data(), &[1.0, 2.0, 3.0], 1e-5));
        Ok(())
    }

    #[test]
    fn egglog_sub_self() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        // a - a should be optimized to 0.
        let result = (&a - &a).realize()?;
        assert_eq!(result.data(), &[0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn realized_data_and_shape() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.realize()?;

        assert_eq!(r.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn reduce_sum_dims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.sum_dims(vec![1], false)?.realize()?;

        assert_eq!(r.data(), &[3.0, 7.0]);
        assert_eq!(r.shape(), &[2]);
        Ok(())
    }

    #[test]
    fn reduce_product_dims_keepdims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.product_dims(vec![0], true)?.realize()?;

        assert_eq!(r.data(), &[3.0, 8.0]);
        assert_eq!(r.shape(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn reduce_max_dims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 9.0, 3.0, 4.0, 7.0, 6.0], vec![2, 3]);
        let r = a.max_dims(vec![0], false)?.realize()?;

        assert_eq!(r.data(), &[4.0, 9.0, 6.0]);
        assert_eq!(r.shape(), &[3]);
        Ok(())
    }

    #[test]
    fn reduce_min_dims_keepdims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 9.0, 3.0, 4.0, 7.0, 6.0], vec![2, 3]);
        let r = a.min_dims(vec![1], true)?.realize()?;

        assert_eq!(r.data(), &[1.0, 4.0]);
        assert_eq!(r.shape(), &[2, 1]);
        Ok(())
    }

    #[test]
    fn reduce_sum_all() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.sum()?.realize()?;

        assert_eq!(r.data(), &[10.0]);
        assert_eq!(r.shape(), &[1, 1]);
        Ok(())
    }

    #[test]
    fn shape_then_reduce_sum() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = a.reshape(vec![3, 2])?.sum_dims(vec![1], false)?.realize()?;

        assert_eq!(r.data(), &[3.0, 7.0, 11.0]);
        assert_eq!(r.shape(), &[3]);
        Ok(())
    }

    #[test]
    fn reduce_then_elementwise_add() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let summed = a.sum_dims(vec![1], true)?;
        let bias = Tensor::constant(&cx, 10.0, vec![2, 1]);
        let r = (&summed + &bias).realize()?;

        assert_eq!(r.data(), &[13.0, 17.0]);
        assert_eq!(r.shape(), &[2, 1]);
        Ok(())
    }

    #[test]
    fn permute_then_reduce_keepdims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = a.permute(vec![1, 0])?.sum_dims(vec![1], true)?.realize()?;

        assert_eq!(r.data(), &[5.0, 7.0, 9.0]);
        assert_eq!(r.shape(), &[3, 1]);
        Ok(())
    }

    #[test]
    fn chained_reduce_keepdims_then_reduce() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a
            .sum_dims(vec![1], true)?
            .sum_dims(vec![0], true)?
            .realize()?;

        assert_eq!(r.data(), &[10.0]);
        assert_eq!(r.shape(), &[1, 1]);
        Ok(())
    }

    #[test]
    fn shape_reduce_elementwise_chain() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a
            .transpose(0, 1)?
            .sum_dims(vec![1], false)?
            .unsqueeze(2)?
            .realize()?;

        assert_eq!(r.data(), &[4.0, 6.0]);
        assert_eq!(r.shape(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn matmul_2d() -> Result<()> {
        let cx = Context::new();

        // [2,3] @ [3,2]
        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_slice(&cx, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let r = a.matmul(&b)?.realize()?;

        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        assert_eq!(r.data(), &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(r.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn matmul_batched() -> Result<()> {
        let cx = Context::new();

        // [2, 2, 2] @ [2, 2] -> broadcast b to [2, 2, 2], result [2, 2, 2]
        #[rustfmt::skip]
        let a = Tensor::from_slice(&cx, &[
            1.0, 2.0,  3.0, 4.0,   // batch 0: [[1,2],[3,4]]
            5.0, 6.0,  7.0, 8.0,   // batch 1: [[5,6],[7,8]]
        ], vec![2, 2, 2]);
        let b = Tensor::from_slice(&cx, &[1.0, 0.0, 0.0, 1.0], vec![2, 2]); // identity
        let r = a.matmul(&b)?.realize()?;

        // multiplying by identity gives back the same values
        assert_eq!(r.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(r.shape(), &[2, 2, 2]);
        Ok(())
    }
}
