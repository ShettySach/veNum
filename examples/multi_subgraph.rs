use venum::{compile, Buffer, Context, DType, Tensor};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    let a = Tensor::placeholder(&cx, DType::F32, vec![1, 4]);
    let b = Tensor::placeholder(&cx, DType::F32, vec![1, 4]);
    let c = Tensor::placeholder(&cx, DType::F32, vec![4, 1]);
    let d = Tensor::placeholder(&cx, DType::F32, vec![4, 1]);

    let w = a.mul(&b)?.add(&a)?;
    let x = c.mul(&d)?.sub(&c)?;
    let y = w.add(&x)?;

    let program = compile(&cx, &[a.id(), b.id(), c.id(), d.id()], &[y.id()])?;

    println!("Input specs: {:?}", program.input_specs);
    println!("Output specs: {:?}", program.output_specs);
    println!("Compiled kernels: {}", program.num_kernels());
    println!("Execution steps: {}\n", program.num_steps());

    println!("Compiled graph:\n{}\n", program.render_compiled_graph());

    let a_buf = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b_buf = Buffer::from_f32_vec(vec![1.0, 1.0, 1.0, 1.0]);
    let c_buf = Buffer::from_f32_vec(vec![2.0, 2.0, 2.0, 2.0]);
    let d_buf = Buffer::from_f32_vec(vec![0.5, 0.5, 0.5, 0.5]);

    let output = program.execute_with_metadata(&[&a_buf, &b_buf, &c_buf, &d_buf])?;

    println!("Result shape: {:?}", output.shapes[0]);
    output.print_tensor(0);

    Ok(())
}
