use venum::{Buffer, Context, DType, Tensor, compile};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    let x = Tensor::placeholder(&cx, DType::F32, vec![3, 3, 2]);
    let y = Tensor::placeholder(&cx, DType::F32, vec![2, 5]);

    let z = x.matmul(&y)?;

    let program = compile(&cx, &[x.id(), y.id()], &[z.id()])?;

    println!("Input specs: {:?}", program.input_specs);
    println!("Output specs: {:?}", program.output_specs);
    println!("Compiled kernels: {}", program.num_kernels());
    println!("Execution steps: {}\n", program.num_steps());

    println!("Compiled graph:\n{}\n", program.render_compiled_graph());

    let x_buf = Buffer::from_f32_vec((0..18).map(|i| i as f32).collect());
    let y_buf = Buffer::from_f32_vec((0..10).map(|i| i as f32).collect());

    let results = program.execute(&[&x_buf, &y_buf])?;

    println!("Result shape: {:?}", program.output_specs[0].shape);
    println!("Result: {:?}", results[0].as_f32());

    Ok(())
}
