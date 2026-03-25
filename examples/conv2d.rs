use venum::{Buffer, DType, LiquidContext, SolidContext, Tensor, compile};

fn main() -> anyhow::Result<()> {
    // ── Liquid mode (JIT) ────────────────────────────────────────────────
    println!("═══ Liquid (JIT) ═══\n");

    let cx = LiquidContext::new();

    // MNIST-style: 1 image, 1 channel, 5x5 input
    let input = Tensor::from_slice(
        &cx,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0, 9.0, 10.0, //
            11.0, 12.0, 13.0, 14.0, 15.0, //
            16.0, 17.0, 18.0, 19.0, 20.0, //
            21.0, 22.0, 23.0, 24.0, 25.0, //
        ],
        vec![1, 1, 5, 5],
    );

    // 2 output channels, 1 input channel, 3x3 kernel
    let weight = Tensor::from_slice(
        &cx,
        &[
            // Filter 0: edge-detect style
            1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0,
            // Filter 1: all ones (box blur)
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 1, 3, 3],
    );

    let output = input.conv2d(&weight)?;

    println!("Input shape:  {:?}", input.shape());
    println!("Weight shape: {:?}", weight.shape());
    println!("Output shape: {:?}\n", output.shape());

    let result = output.realize()?;
    println!("{result}");

    // ── Solid mode (AOT) ─────────────────────────────────────────────────
    println!("═══ Solid (AOT) ═══\n");

    let cx1 = SolidContext::new();

    let input1 = Tensor::placeholder(&cx1, DType::F32, vec![1, 1, 5, 5]);
    let weight1 = Tensor::placeholder(&cx1, DType::F32, vec![2, 1, 3, 3]);

    let output1 = input1.conv2d(&weight1)?;

    let program = compile(&cx1, &[input1.id(), weight1.id()], &[output1.id()])?;

    println!("Compiled kernels: {}", program.num_kernels());
    println!("Execution steps:  {}\n", program.num_steps());

    let input_buf = Buffer::from_f32_vec((1..=25).map(|i| i as f32).collect());
    let weight_buf = Buffer::from_f32_vec(vec![
        1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ]);

    let results = program.execute(&[&input_buf, &weight_buf])?;

    // ── Verify ───────────────────────────────────────────────────────────
    println!("═══ Verification ═══\n");

    let liquid = result.data();
    let solid = results[0].as_f32();

    if liquid == solid {
        println!("Liquid and Solid results match!");
    } else {
        println!("Results differ!");
        println!("Liquid: {:?}", liquid);
        println!("Solid:  {:?}", solid);
    }

    // ── DAGs ─────────────────────────────────────────────────────────────
    println!("\n═══ Liquid DAG ═══\n");
    println!("{}", output.render_dag());

    println!("═══ Liquid Optimized Fused DAG ═══\n");
    println!("{}", output.render_optimized_fused_dag()?);

    println!("═══ Solid Compiled DAG ═══\n");
    println!("{}", program.render_compiled_graph());

    Ok(())
}
