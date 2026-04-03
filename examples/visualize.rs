use venum::{Buffer, Context, DType, SearchConfig, Tensor, run_context};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    let a = Tensor::placeholder(&cx, DType::F32, vec![8]);
    let b = Tensor::placeholder(&cx, DType::F32, vec![8]);
    let c = a.mul(&b)?.add(&a)?;

    let out = run_context(
        &cx,
        &[c.id()],
        &[
            Buffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            Buffer::F32(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
        ],
    )?;

    let cfg = SearchConfig {
        beam_width: 4,
        max_iterations: 2,
        ..SearchConfig::default()
    };

    println!("nodes: {}", cx.num_nodes());
    println!("\n=== HLIR ===\n{}", cx.graph_mermaid());
    println!(
        "\n=== HLIR (optimized) ===\n{}",
        cx.optimized_graph_mermaid(&[c.id()])
    );
    println!(
        "\n=== ScheduleDecision ===\n{}",
        cx.schedule_mermaid(&[c.id()], &cfg)
    );
    println!("\n=== LLIR ===\n{}", cx.llir_mermaid(&[c.id()], &cfg));

    match &out[0] {
        Buffer::F32(v) => println!("\nresult: {:?}", v),
        _ => println!("\nunexpected output dtype"),
    }

    Ok(())
}
