use venum::NaiveTensor;

fn main() -> anyhow::Result<()> {
    let tensor = NaiveTensor::eye(3)?;
    println!("{}", tensor);

    let unary_fn: fn(f32) -> f32 = |x| (x + 5.0) / 10.0;
    let unary_map = tensor.unary_map(unary_fn)?;
    println!("{}", unary_map);

    let binary_fn: fn(f32, f32) -> f32 = |x, y| (x * y) / 10.0;
    let binary_map = tensor.binary_map(5.0, binary_fn)?; // with Scalar
    let zip = tensor.zip(&binary_map, binary_fn)?; // with another Tensor
    println!("{}", binary_map);
    println!("{}", zip);

    let index_map = tensor.index_map(|v| v + 1.0, &[1, 1])?; // Increment
    println!("{}", index_map);

    let slice_map = tensor.slice_map(|v| v + 1.0, &[(1, 3), (1, 3)])?; // Increment
    println!("{}", slice_map);

    let arr = &[1.0, 2.0, 3.0, 4.0];
    let sz = tensor.slice_zip(arr, |_, y| y, &[(1, 3), (1, 3)])?; // Slice assign
    println!("{}", sz);

    let bool_tensor = NaiveTensor::new(
        &[
            false, true, false, //
            false, true, true, //
            false, true, false, //
        ],
        &[3, 3],
    )?;
    println!("{}", bool_tensor);

    let bool_sum = |bool_tensor: &NaiveTensor<bool>| {
        Ok(bool_tensor.data().iter().filter(|&&b| b).count()) // Count of true values
    };

    let bool_sum_dim0 = bool_tensor.reduce(bool_sum, &[0], false)?; // bool_sum of along dim 1
    println!("{}", bool_sum_dim0);

    let bool_sum_dim1 = bool_tensor.reduce(bool_sum, &[1], false)?; // bool_sum of along dim 1
    println!("{}", bool_sum_dim1);

    Ok(())
}
