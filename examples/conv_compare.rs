use venum::{Buffer, Context, DType, Tensor, run_context};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    let input = Tensor::placeholder(&cx, DType::F32, vec![1, 1, 5, 5]);
    let weight = Tensor::placeholder(&cx, DType::F32, vec![1, 1, 3, 3]);

    let output_conv2d = input.conv2d(&weight)?;
    let output_conv2d_tg = input.conv2d_tg(&weight)?;

    let input_data: Vec<f32> = (1..=25).map(|i| i as f32).collect();
    let weight_data: Vec<f32> = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0];

    let out_conv2d = run_context(
        &cx,
        &[output_conv2d.id()],
        &[
            Buffer::F32(input_data.clone()),
            Buffer::F32(weight_data.clone()),
        ],
    )?;
    let out_conv2d_tg = run_context(
        &cx,
        &[output_conv2d_tg.id()],
        &[Buffer::F32(input_data), Buffer::F32(weight_data)],
    )?;

    println!("=== conv2d (shifted-slice decomposition) ===");
    println!("Output: {:?}", out_conv2d[0]);
    println!("Nodes: {}", cx.num_nodes());

    println!("\n=== conv2d_tg (pool-based) ===");
    println!("Output: {:?}", out_conv2d_tg[0]);

    let cx2 = Context::new();
    let input2 = Tensor::placeholder(&cx2, DType::F32, vec![1, 1, 5, 5]);
    let weight2 = Tensor::placeholder(&cx2, DType::F32, vec![1, 1, 3, 3]);
    let _ = input2.conv2d_tg(&weight2)?;
    println!("\nconv2d_tg graph nodes: {}", cx2.num_nodes());

    let cx3 = Context::new();
    let input3 = Tensor::placeholder(&cx3, DType::F32, vec![1, 1, 5, 5]);
    let weight3 = Tensor::placeholder(&cx3, DType::F32, vec![1, 1, 3, 3]);
    let _ = input3.conv2d(&weight3)?;
    println!("conv2d (shifted-slice) graph nodes: {}", cx3.num_nodes());

    Ok(())
}
