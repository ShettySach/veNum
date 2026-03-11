#[cfg(test)]
mod delayed_tests {
    use crate::{DTensor, Tensor};
    use anyhow::Result;

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
    }

    #[test]
    fn realize_leaf() -> Result<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let dt = DTensor::from_slice(&data, vec![4]);
        let result = dt.realize()?;

        assert_eq!(result.data().as_ref(), &data);
        assert_eq!(result.sizes(), &[4]);
        Ok(())
    }

    #[test]
    fn add_two_tensors() -> Result<()> {
        let a = DTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = DTensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![4]);
        let c = &a + &b;
        let result = c.realize()?;

        assert_eq!(result.data().as_ref(), &[11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    #[test]
    fn mul_two_tensors() -> Result<()> {
        let a = DTensor::from_slice(&[2.0, 3.0, 4.0, 5.0], vec![4]);
        let b = DTensor::from_slice(&[10.0, 10.0, 10.0, 10.0], vec![4]);
        let c = &a * &b;
        let result = c.realize()?;

        assert_eq!(result.data().as_ref(), &[20.0, 30.0, 40.0, 50.0]);
        Ok(())
    }

    #[test]
    fn sub_and_div() -> Result<()> {
        let a = DTensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![4]);
        let b = DTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);

        let sub_result = (&a - &b).realize()?;
        assert_eq!(sub_result.data().as_ref(), &[9.0, 18.0, 27.0, 36.0]);

        let div_result = (&a / &b).realize()?;
        assert_eq!(div_result.data().as_ref(), &[10.0, 10.0, 10.0, 10.0]);

        Ok(())
    }

    #[test]
    fn fused_add_mul() -> Result<()> {
        // (a + b) * a should produce ONE fused kernel.
        let a = DTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = DTensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![4]);
        let c = &(&a + &b) * &a;
        let result = c.realize()?;

        // (1+10)*1=11, (2+20)*2=44, (3+30)*3=99, (4+40)*4=176
        assert_eq!(result.data().as_ref(), &[11.0, 44.0, 99.0, 176.0]);
        Ok(())
    }

    #[test]
    fn unary_neg() -> Result<()> {
        let a = DTensor::from_slice(&[1.0, -2.0, 3.0, -4.0], vec![4]);
        let result = a.neg().realize()?;

        assert_eq!(result.data().as_ref(), &[-1.0, 2.0, -3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn unary_exp() -> Result<()> {
        let a = DTensor::from_slice(&[0.0, 1.0, 2.0], vec![3]);
        let result = a.exp().realize()?;
        let data = result.data();

        let expected: Vec<f32> = vec![0.0f32, 1.0, 2.0].iter().map(|x| x.exp()).collect();
        assert!(approx_eq(&data, &expected, 1e-5));
        Ok(())
    }

    #[test]
    fn unary_ln() -> Result<()> {
        let a = DTensor::from_slice(&[1.0, 2.718281828, 7.389056], vec![3]);
        let result = a.ln().realize()?;
        let data = result.data();

        let expected: Vec<f32> = vec![1.0f32, 2.718281828, 7.389056]
            .iter()
            .map(|x| x.ln())
            .collect();
        assert!(approx_eq(&data, &expected, 1e-4));
        Ok(())
    }

    #[test]
    fn unary_sqrt() -> Result<()> {
        let a = DTensor::from_slice(&[1.0, 4.0, 9.0, 16.0], vec![4]);
        let result = a.sqrt().realize()?;

        assert_eq!(result.data().as_ref(), &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn chain_fused_unary_binary() -> Result<()> {
        // exp(a) + b — should fuse into one kernel.
        let a = DTensor::from_slice(&[0.0, 0.0, 0.0], vec![3]);
        let b = DTensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let c = &a.exp() + &b;
        let result = c.realize()?;

        // exp(0) + 1 = 2, exp(0) + 2 = 3, exp(0) + 3 = 4
        assert_eq!(result.data().as_ref(), &[2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn from_eager_tensor() -> Result<()> {
        let eager = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
        let dt = DTensor::from_tensor(&eager);
        let b = DTensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let c = &dt + &b;
        let result = c.realize()?;

        assert_eq!(result.data().as_ref(), &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(result.sizes(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn larger_tensor() -> Result<()> {
        let n = 1024;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

        let a = DTensor::from_slice(&a_data, vec![n]);
        let b = DTensor::from_slice(&b_data, vec![n]);
        let c = &a + &b;
        let result = c.realize()?;

        let expected: Vec<f32> = (0..n).map(|_| n as f32).collect();
        assert_eq!(result.data().as_ref(), &expected);
        Ok(())
    }

    #[test]
    fn deep_fusion_chain() -> Result<()> {
        // a * b + a - b should all fuse into one kernel.
        let a = DTensor::from_slice(&[2.0, 3.0, 4.0, 5.0], vec![4]);
        let b = DTensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);

        let c = &(&(&a * &b) + &a) - &b;
        let result = c.realize()?;

        // (2*1 + 2 - 1) = 3, (3*1 + 3 - 1) = 5, (4*1 + 4 - 1) = 7, (5*1 + 5 - 1) = 9
        assert_eq!(result.data().as_ref(), &[3.0, 5.0, 7.0, 9.0]);
        Ok(())
    }

    #[test]
    fn constant_tensor() -> Result<()> {
        let a = DTensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let c = DTensor::constant(10.0, vec![3]);
        let result = (&a + &c).realize()?;

        assert_eq!(result.data().as_ref(), &[11.0, 12.0, 13.0]);
        Ok(())
    }
}
