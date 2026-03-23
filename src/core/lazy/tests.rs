#[cfg(test)]
mod lazy_tests {
    use crate::core::lazy::schedule::{build_schedule, ScheduleItem};
    use crate::core::shared::dtype::Buffer;
    use crate::core::shared::graph::{Graph, Op};
    use crate::{Context, Tensor};
    use anyhow::Result;

    #[test]
    fn realize_leaf() -> Result<()> {
        let cx = Context::new();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let dt = Tensor::from_slice(&cx, &data, vec![4]);
        let result = dt.realize()?;

        assert_eq!(*result.data(), *data);
        assert_eq!(result.sizes(), &[4]);
        Ok(())
    }

    #[test]
    fn add_two_tensors() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let c = (&a + &b)?;
        let result = c.realize()?;

        assert_eq!(*result.data(), [11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    #[test]
    fn mul_two_tensors() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 10.0, 10.0, 10.0], vec![4]);
        let c = (&a * &b)?;
        let result = c.realize()?;

        assert_eq!(*result.data(), [20.0, 30.0, 40.0, 50.0]);
        Ok(())
    }

    #[test]
    fn sub_and_div() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);

        let sub_result = (&a - &b)?.realize()?;
        assert_eq!(*sub_result.data(), [9.0, 18.0, 27.0, 36.0]);

        let div_result = (&a / &b)?.realize()?;
        assert_eq!(*div_result.data(), [10.0, 10.0, 10.0, 10.0]);

        Ok(())
    }

    #[test]
    fn fused_add_mul() -> Result<()> {
        let cx = Context::new();

        // (a + b) * a should produce ONE fused kernel.
        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let sum = (&a + &b)?;
        let c = (&sum * &a)?;
        let result = c.realize()?;

        // (1+10)*1=11, (2+20)*2=44, (3+30)*3=99, (4+40)*4=176
        assert_eq!(*result.data(), [11.0, 44.0, 99.0, 176.0]);
        Ok(())
    }

    #[test]
    fn unary_neg() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0, -4.0], vec![4]);
        let result = a.neg().realize()?;

        assert_eq!(*result.data(), [-1.0, 2.0, -3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn unary_exp() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[0.0, 1.0, 2.0], vec![3]);
        let result = a.exp()?.realize()?;

        let expected: Vec<f32> = [0.0f32, 1.0, 2.0].iter().map(|x| x.exp()).collect();
        assert_eq!(&result.data(), &expected);
        Ok(())
    }

    #[test]
    fn unary_ln() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.718_281_7, 7.389056], vec![3]);
        let result = a.ln()?.realize()?;

        let expected: Vec<f32> = [1.0f32, 2.718_281_7, 7.389056]
            .iter()
            .map(|x| x.ln())
            .collect();
        assert_eq!(result.data(), &expected);
        Ok(())
    }

    #[test]
    fn unary_sqrt() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 4.0, 9.0, 16.0], vec![4]);
        let result = a.sqrt()?.realize()?;

        assert_eq!(*result.data(), [1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn chain_fused_unary_binary() -> Result<()> {
        let cx = Context::new();

        // exp(a) + b — should fuse into one kernel.
        let a = Tensor::from_slice(&cx, &[0.0, 0.0, 0.0], vec![3]);
        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let c = (&a.exp()? + &b)?;
        let result = c.realize()?;

        // exp(0) + 1 = 2, exp(0) + 2 = 3, exp(0) + 3 = 4
        assert_eq!(*result.data(), [2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn from_slice_add() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let c = (&a + &b)?;
        let result = c.realize()?;

        assert_eq!(*result.data(), [11.0, 22.0, 33.0, 44.0]);
        assert_eq!(result.sizes(), &[2, 2]);
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
        let c = (&a + &b)?;
        let result = c.realize()?;

        let expected: Vec<f32> = (0..n).map(|_| n as f32).collect();
        assert_eq!(*result.data(), *expected);
        Ok(())
    }

    #[test]
    fn deep_fusion_chain() -> Result<()> {
        let cx = Context::new();

        // a * b + a - b should all fuse into one kernel.
        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[1.0, 1.0, 1.0, 1.0], vec![4]);

        let ab = (&a * &b)?;
        let ab_a = (&ab + &a)?;
        let c = (&ab_a - &b)?;
        let result = c.realize()?;

        // (2*1 + 2 - 1) = 3, (3*1 + 3 - 1) = 5, (4*1 + 4 - 1) = 7, (5*1 + 5 - 1) = 9
        assert_eq!(*result.data(), [3.0, 5.0, 7.0, 9.0]);
        Ok(())
    }

    #[test]
    fn constant_tensor() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let c = Tensor::constant(&cx, 10.0, vec![3]);
        let result = (&a + &c)?.realize()?;

        assert_eq!(*result.data(), [11.0, 12.0, 13.0]);
        Ok(())
    }

    // --- egglog optimization tests ---

    #[test]
    fn egglog_add_zero_identity() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let zero = Tensor::constant(&cx, 0.0, vec![3]);
        // a + 0 should be optimized to just a.
        let result = (&a + &zero)?.realize()?;
        assert_eq!(*result.data(), [1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn egglog_mul_one_identity() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        let one = Tensor::constant(&cx, 1.0, vec![3]);
        // a * 1 should be optimized to just a.
        let result = (&a * &one)?.realize()?;
        assert_eq!(*result.data(), [5.0, 10.0, 15.0]);
        Ok(())
    }

    #[test]
    fn egglog_mul_zero() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        let zero = Tensor::constant(&cx, 0.0, vec![3]);
        // a * 0 should be optimized to 0.
        let result = (&a * &zero)?.realize()?;
        assert_eq!(*result.data(), [0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn egglog_double_neg() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0], vec![3]);
        // neg(neg(a)) should be optimized to just a.
        let result = a.neg().neg().realize()?;
        assert_eq!(*result.data(), [1.0, -2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn egglog_exp_ln_inverse() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        // exp(ln(a)) should be optimized to just a.
        let result = a.ln()?.exp()?.realize()?;
        assert_eq!(result.data(), &[1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn egglog_sub_self() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        // a - a should be optimized to 0.
        let result = (&a - &a)?.realize()?;
        assert_eq!(*result.data(), [0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn egglog_shape_ops_optimize_and_execute() -> Result<()> {
        let cx = Context::new();

        // Reshape/transpose/squeeze/unsqueeze chain should stay optimize-safe
        // and preserve values.
        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a
            .reshape(vec![1, 2, 2])?
            .transpose(1, 2)?
            .squeeze()?
            .unsqueeze(3)?
            .realize()?;

        assert_eq!(*r.data(), [1.0, 3.0, 2.0, 4.0]);
        assert_eq!(r.sizes(), &[1, 2, 2]);
        Ok(())
    }

    #[test]
    fn forward_fusion_basic_reshape() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let y = (&a + &b)?.reshape(vec![2, 2])?;

        let fused = y.render_fused_dag();
        assert!(!fused.contains("Shape Op"));

        let out = y.realize()?;
        assert_eq!(out.sizes(), &[2, 2]);
        assert_eq!(*out.data(), [11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    #[test]
    fn forward_fusion_transpose() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let y = (&a + &b)?.transpose(0, 1)?;

        let fused = y.render_fused_dag();
        assert!(!fused.contains("Shape Op"));

        let out = y.realize()?;
        assert_eq!(out.sizes(), &[2, 2]);
        assert_eq!(*out.data(), [11.0, 33.0, 22.0, 44.0]);
        Ok(())
    }

    #[test]
    fn forward_fusion_shape_chain() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let y = (&a + &b)?.reshape(vec![1, 2, 2])?.squeeze()?.unsqueeze(3)?;

        let fused = y.render_fused_dag();
        assert!(!fused.contains("Shape Op"));

        let out = y.realize()?;
        assert_eq!(out.sizes(), &[1, 2, 2]);
        assert_eq!(*out.data(), [11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    #[test]
    fn phase2_optimized_fused_dag_has_no_shape_barriers() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);

        let y = (&a.reshape(vec![1, 2, 2])?.reshape(vec![1, 2, 2])?
            + &b.reshape(vec![1, 2, 2])?)?
            .transpose(1, 2)?
            .transpose(1, 2)?
            .squeeze()?
            .unsqueeze(3)?;

        let fused = y.render_optimized_fused_dag()?;
        assert!(!fused.contains("Shape Op"));

        let out = y.realize()?;
        assert_eq!(out.sizes(), &[1, 2, 2]);
        assert_eq!(*out.data(), [11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    #[test]
    fn forward_fusion_reduce_then_reshape() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let y = a.sum_dims(vec![1], true)?.reshape(vec![1, 2])?;

        let fused = y.render_fused_dag();
        assert!(!fused.contains("Shape Op"));

        let out = y.realize()?;
        assert_eq!(out.sizes(), &[1, 2]);
        assert_eq!(*out.data(), [6.0, 15.0]);
        Ok(())
    }

    #[test]
    fn forward_fusion_blocked_by_multiple_consumers() {
        let mut g = Graph::new();
        let a = g.load(Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0]), vec![4]);
        let b = g.load(Buffer::from_f32_vec(vec![10.0, 20.0, 30.0, 40.0]), vec![4]);

        let add = g.binary(Op::Add, a, b);
        let neg = g.unary(Op::Neg, add);
        let reshaped = g.reshape(add, vec![2, 2]);
        let reshaped_back = g.reshape(reshaped, vec![4]);
        let root = g.binary(Op::Add, neg, reshaped_back);

        let schedule = build_schedule(&g, root);

        let add_kernel = schedule.iter().find_map(|item| match item {
            ScheduleItem::Fused(k) if k.expr_root == add => Some(k),
            _ => None,
        });

        let add_kernel = add_kernel.expect("expected separate kernel for shared add node");
        assert!(add_kernel.output_tracker.is_none());
    }

    #[test]
    fn realized_data_and_shape() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.realize()?;

        assert_eq!(*r.data(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r.sizes(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn reduce_sum_dims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.sum_dims(vec![1], false)?.realize()?;

        assert_eq!(*r.data(), [3.0, 7.0]);
        assert_eq!(r.sizes(), &[2]);
        Ok(())
    }

    #[test]
    fn reduce_product_dims_keepdims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.product_dims(vec![0], true)?.realize()?;

        assert_eq!(*r.data(), [3.0, 8.0]);
        assert_eq!(r.sizes(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn reduce_max_dims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 9.0, 3.0, 4.0, 7.0, 6.0], vec![2, 3]);
        let r = a.max_dims(vec![0], false)?.realize()?;

        assert_eq!(*r.data(), [4.0, 9.0, 6.0]);
        assert_eq!(r.sizes(), &[3]);
        Ok(())
    }

    #[test]
    fn reduce_min_dims_keepdims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 9.0, 3.0, 4.0, 7.0, 6.0], vec![2, 3]);
        let r = a.min_dims(vec![1], true)?.realize()?;

        assert_eq!(*r.data(), [1.0, 4.0]);
        assert_eq!(r.sizes(), &[2, 1]);
        Ok(())
    }

    #[test]
    fn reduce_sum_all() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.sum()?.realize()?;

        assert_eq!(*r.data(), [10.0]);
        assert_eq!(r.sizes(), &[1, 1]);
        Ok(())
    }

    #[test]
    fn shape_then_reduce_sum() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = a.reshape(vec![3, 2])?.sum_dims(vec![1], false)?.realize()?;

        assert_eq!(*r.data(), [3.0, 7.0, 11.0]);
        assert_eq!(r.sizes(), &[3]);
        Ok(())
    }

    #[test]
    fn reduce_then_elementwise_add() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let summed = a.sum_dims(vec![1], true)?;
        let bias = Tensor::constant(&cx, 10.0, vec![2, 1]);
        let r = (&summed + &bias)?.realize()?;

        assert_eq!(*r.data(), [13.0, 17.0]);
        assert_eq!(r.sizes(), &[2, 1]);
        Ok(())
    }

    #[test]
    fn permute_then_reduce_keepdims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = a.permute(vec![1, 0])?.sum_dims(vec![1], true)?.realize()?;

        assert_eq!(*r.data(), [5.0, 7.0, 9.0]);
        assert_eq!(r.sizes(), &[3, 1]);
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

        assert_eq!(*r.data(), [10.0]);
        assert_eq!(r.sizes(), &[1, 1]);
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

        assert_eq!(*r.data(), [4.0, 6.0]);
        assert_eq!(r.sizes(), &[1, 2]);
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
        assert_eq!(*r.data(), [58.0, 64.0, 139.0, 154.0]);
        assert_eq!(r.sizes(), &[2, 2]);
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
        assert_eq!(*r.data(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(r.sizes(), &[2, 2, 2]);
        Ok(())
    }

    // --- multi-dtype tests ---

    #[test]
    fn i32_add() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice_i32(&cx, &[1, 2, 3, 4], vec![4]);
        let b = Tensor::from_slice_i32(&cx, &[10, 20, 30, 40], vec![4]);
        let c = (&a + &b)?;
        let r = c.realize()?;

        assert_eq!(*r.data_i32(), [11, 22, 33, 44]);
        assert_eq!(r.sizes(), &[4]);
        Ok(())
    }

    #[test]
    fn i32_sub_mul_div() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice_i32(&cx, &[10, 20, 30, 40], vec![4]);
        let b = Tensor::from_slice_i32(&cx, &[1, 2, 3, 4], vec![4]);

        let sub = (&a - &b)?.realize()?;
        assert_eq!(*sub.data_i32(), [9, 18, 27, 36]);

        let mul = (&a * &b)?.realize()?;
        assert_eq!(*mul.data_i32(), [10, 40, 90, 160]);

        let div = (&a / &b)?.realize()?;
        assert_eq!(*div.data_i32(), [10, 10, 10, 10]);

        Ok(())
    }

    #[test]
    fn i32_neg() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice_i32(&cx, &[1, -2, 3, -4], vec![4]);
        let r = a.neg().realize()?;

        assert_eq!(*r.data_i32(), [-1, 2, -3, 4]);
        Ok(())
    }

    #[test]
    fn i64_add() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice_i64(&cx, &[100, 200, 300], vec![3]);
        let b = Tensor::from_slice_i64(&cx, &[1, 2, 3], vec![3]);
        let r = (&a + &b)?.realize()?;

        assert_eq!(*r.data_i64(), [101, 202, 303]);
        Ok(())
    }

    #[test]
    fn f64_add_mul() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice_f64(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice_f64(&cx, &[10.0, 20.0, 30.0], vec![3]);
        let r = (&a + &b)?.realize()?;

        assert_eq!(*r.data_f64(), [11.0, 22.0, 33.0]);

        let r2 = (&a * &b)?.realize()?;
        assert_eq!(*r2.data_f64(), [10.0, 40.0, 90.0]);

        Ok(())
    }

    #[test]
    fn f64_exp_ln() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice_f64(&cx, &[0.0, 1.0, 2.0], vec![3]);
        let r = a.exp()?.realize()?;

        let expected: Vec<f64> = [0.0f64, 1.0, 2.0].iter().map(|x| x.exp()).collect();
        let data = r.data_f64();
        assert!(data
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-10),);
        Ok(())
    }

    #[test]
    fn i32_reduce_sum() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice_i32(&cx, &[1, 2, 3, 4], vec![2, 2]);
        let r = a.sum_dims(vec![1], false)?.realize()?;

        assert_eq!(*r.data_i32(), [3, 7]);
        assert_eq!(r.sizes(), &[2]);
        Ok(())
    }

    #[test]
    fn i32_reshape_permute() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice_i32(&cx, &[1, 2, 3, 4, 5, 6], vec![2, 3]);
        let r = a.permute(vec![1, 0])?.realize()?;

        assert_eq!(*r.data_i32(), [1, 4, 2, 5, 3, 6]);
        assert_eq!(r.sizes(), &[3, 2]);
        Ok(())
    }

    #[test]
    fn dtype_mismatch_errors() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0], vec![2]);
        let b = Tensor::from_slice_i32(&cx, &[1, 2], vec![2]);

        let result = &a + &b;
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn int_exp_errors() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice_i32(&cx, &[1, 2, 3], vec![3]);
        assert!(a.exp().is_err());
        assert!(a.ln().is_err());
        assert!(a.sqrt().is_err());
        Ok(())
    }

    #[test]
    fn i32_constant_add() -> Result<()> {
        use crate::Scalar;

        let cx = Context::new();

        let a = Tensor::from_slice_i32(&cx, &[1, 2, 3], vec![3]);
        let c = Tensor::constant_scalar(&cx, Scalar::I32(10), vec![3]);
        let r = (&a + &c)?.realize()?;

        assert_eq!(*r.data_i32(), [11, 12, 13]);
        Ok(())
    }

    #[test]
    fn f64_constant_add() -> Result<()> {
        use crate::Scalar;

        let cx = Context::new();

        let a = Tensor::from_slice_f64(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let c = Tensor::constant_scalar(&cx, Scalar::F64(10.0), vec![3]);
        let r = (&a + &c)?.realize()?;

        assert_eq!(*r.data_f64(), [11.0, 12.0, 13.0]);
        Ok(())
    }

    #[test]
    fn realize_leaf_i32() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice_i32(&cx, &[10, 20, 30], vec![3]);
        let r = a.realize()?;

        assert_eq!(*r.data_i32(), [10, 20, 30]);
        assert_eq!(r.sizes(), &[3]);
        Ok(())
    }

    /// Regression test for Load node deduplication bug.
    /// When egglog extracts optimized terms, Load nodes that appear multiple times
    /// should be deduplicated to reuse the same NodeId in the reconstructed graph.
    /// This test creates a diamond pattern where inputs are reused in multiple branches.
    #[test]
    fn load_node_deduplication() -> Result<()> {
        let cx = Context::new();

        // Create input tensors
        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[1.0, 1.0, 1.0, 1.0], vec![4]);
        let c = Tensor::from_slice(&cx, &[2.0, 2.0, 2.0, 2.0], vec![4]);
        let d = Tensor::from_slice(&cx, &[0.5, 0.5, 0.5, 0.5], vec![4]);

        // Branch 1: (a * b) + a -> uses 'a' twice
        let w1 = (&a * &b)?;
        let w2 = (&w1 + &a)?;
        let w3 = w2.reshape(vec![2, 2])?;

        // Branch 2: (c * d) + c -> uses 'c' twice
        let x1 = (&c * &d)?;
        let x2 = (&x1 + &c)?;
        let x3 = x2.reshape(vec![2, 2])?;

        // Final combination
        let y = (&w3 + &x3)?;

        // Get DAG renderings to check node count
        let raw_dag = y.render_dag();
        let opt_dag = y.render_optimized_dag()?;

        println!("=== RAW DAG ===\n{}", raw_dag);
        println!("=== OPTIMIZED DAG ===\n{}", opt_dag);

        // Count Load nodes in the raw DAG - should be 4
        // Lines look like: N0["Load F32\n[...]"]
        let raw_load_count = raw_dag
            .lines()
            .filter(|line| line.contains("\"Load"))
            .count();
        assert_eq!(raw_load_count, 4, "Raw DAG should have 4 Load nodes");

        // Count Load nodes in the optimized DAG
        // The bug would cause 6 Load nodes (duplicates of 'a' and 'c')
        let opt_load_count = opt_dag
            .lines()
            .filter(|line| line.contains("\"Load"))
            .count();

        // With the fix, should still have exactly 4 Load nodes (a, b, c, d)
        assert_eq!(
            opt_load_count, 4,
            "Expected 4 Load nodes in optimized graph, found {}.\nOptimized DAG:\n{}",
            opt_load_count, opt_dag
        );

        // Verify correctness by executing
        let result = y.realize()?;
        assert_eq!(result.sizes(), &[2, 2]);

        // Expected: w3 = reshape((a*b)+a) = reshape([2,4,6,8]) = [[2,4],[6,8]]
        //           x3 = reshape((c*d)+c) = reshape([3,3,3,3]) = [[3,3],[3,3]]
        //           y = w3 + x3 = [[5,7],[9,11]]
        assert_eq!(*result.data(), [5.0, 7.0, 9.0, 11.0]);

        Ok(())
    }
}
