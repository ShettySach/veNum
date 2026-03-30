use venum::{compile, Buffer, Context, DType, Tensor};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    let input = Tensor::placeholder(&cx, DType::F32, vec![1, 1, 5, 5]);
    let weight = Tensor::placeholder(&cx, DType::F32, vec![2, 1, 3, 3]);

    let output = input.conv2d(&weight)?;

    let program = compile(&cx, &[input.id(), weight.id()], &[output.id()])?;

    println!("Compiled kernels: {}", program.num_kernels());
    println!("Execution steps:  {}\n", program.num_steps());

    let input_buf = Buffer::from_f32_vec((1..=25).map(|i| i as f32).collect());
    let weight_buf = Buffer::from_f32_vec(vec![
        1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ]);

    let output_data = program.execute_with_metadata(&[&input_buf, &weight_buf])?;

    println!("Output shape: {:?}", output_data.shapes[0]);
    output_data.print_tensor(0);

    Ok(())
}
