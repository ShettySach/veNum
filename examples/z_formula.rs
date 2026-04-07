use anyhow::Result;
use venum::{Buffer, Context, DType, Scalar, Tensor, run_context};

fn main() -> Result<()> {
    let cx = Context::new();

    let x = Tensor::placeholder(&cx, DType::F32, vec![5]);
    let two = Tensor::constant_scalar(&cx, Scalar::F32(2.0), vec![5]);
    let z = (&x * &two + &x) + (&x * &two + &x);

    println!(
        "=== Optimized Graph ===\n{}",
        cx.optimized_graph_mermaid(&[z.id()])
    );

    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("\nx: {:?}", input_data);

    let outputs = run_context(&cx, &[z.id()], &[Buffer::F32(input_data)])?;

    if let Buffer::F32(data) = &outputs[0] {
        println!("z: {:?}", data);
    }

    Ok(())
}
