use venum::{Buffer, Context, DType, Tensor, run_context};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    let input = Tensor::placeholder(&cx, DType::F32, vec![1, 1, 5, 5]);
    let weight = Tensor::placeholder(&cx, DType::F32, vec![1, 1, 3, 3]);
    let output = input.conv2d(&weight)?;

    let out = run_context(
        &cx,
        &[output.id()],
        &[
            Buffer::F32((1..=25).map(|i| i as f32).collect()),
            Buffer::F32(vec![1.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0]),
        ],
    )?;
    println!("conv2d output id: {:?}", output.id());
    println!("conv2d output: {:?}", out[0]);

    println!("graph nodes: {}", cx.num_nodes());
    println!("\nMermaid graph:\n{}", cx.graph_mermaid());
    println!(
        "\nOptimized Mermaid graph:\n{}",
        cx.optimized_graph_mermaid(&[output.id()])
    );

    Ok(())
}
