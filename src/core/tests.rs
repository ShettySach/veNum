#[cfg(test)]
mod core_tests {
    use crate::core::naive::NaiveTensor;
    use anyhow::Result;

    #[test]
    fn same_memory() -> Result<()> {
        use std::sync::Arc;

        let tensor = NaiveTensor::new_1d(&[1, 2, 3, 4, 5, 6, 7, 8, 9])?;
        let view = tensor.view(&[3, 3])?;
        let slice = tensor.slice(&[(3, 7)])?;

        let tensor_data_ptr = Arc::as_ptr(&tensor.data);
        let view_data_ptr = Arc::as_ptr(&view.data);
        let slice_data_ptr = Arc::as_ptr(&slice.data);

        assert_eq!(tensor_data_ptr, slice_data_ptr);
        assert_eq!(tensor_data_ptr, view_data_ptr);

        Ok(())
    }

    #[test]
    fn contiguous() -> Result<()> {
        let a = NaiveTensor::arange(1, 28, 1)?;
        let a = a.reshape(&[3, 3, 3])?;

        let flip_0 = a.flip(&[0])?;
        let flip_01 = a.flip(&[0, 1])?;
        let flip_all = a.flip_all()?;

        assert!(a.is_contiguous());
        assert!(flip_all.is_contiguous());

        assert!(!flip_0.is_contiguous());
        assert!(!flip_01.is_contiguous());

        Ok(())
    }

    #[test]
    fn view() -> Result<()> {
        let tensor = NaiveTensor::arange(0, 64, 1)?;

        assert!(tensor.view(&[4, 4, 4]).is_ok());
        assert!(tensor.view(&[8, 8]).is_ok());

        assert!(tensor.view(&[2, 25]).is_err());
        assert!(tensor.view(&[1]).is_err());

        Ok(())
    }

    #[test]
    fn eye() -> Result<()> {
        let tensor = NaiveTensor::arange(0, 64, 1)?.view(&[4, 4, 4])?;
        let eye = NaiveTensor::eye(4)?;
        let result = tensor.matmul(&eye)?;

        assert_eq!(tensor, result);

        Ok(())
    }

    #[test]
    fn empty() -> Result<()> {
        let empty = NaiveTensor::<u8>::new_1d(&[])?;

        let max = empty.max();
        let min = empty.min();

        let sum = empty.sum();
        let prod = empty.product();

        assert!(max.is_err());
        assert!(min.is_err());

        assert!(sum.is_ok_and(|sum| sum == 0));
        assert!(prod.is_ok_and(|prod| prod == 1));

        Ok(())
    }
}
