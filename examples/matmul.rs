use venum::{run_context, Buffer, Context, DType, Tensor};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    let x = Tensor::placeholder(&cx, DType::F32, vec![3, 3, 2]);
    let y = Tensor::placeholder(&cx, DType::F32, vec![2, 5]);

    let z = x.matmul(&y)?;

    let out = run_context(
        &cx,
        &[z.id()],
        &[
            Buffer::F32((0..18).map(|i| i as f32).collect()),
            Buffer::F32((0..10).map(|i| i as f32).collect()),
        ],
    )?;

    println!("matmul graph nodes: {}", cx.num_nodes());
    println!("\nMermaid graph:\n{}", cx.graph_mermaid());
    println!(
        "\nOptimized Mermaid graph:\n{}",
        cx.optimized_graph_mermaid(&[z.id()])
    );
    match &out[0] {
        Buffer::F32(v) => {
            println!("output len: {}", v.len());
            println!("output: {:?}", v);
        }
        _ => println!("unexpected output dtype"),
    }

    Ok(())
}
