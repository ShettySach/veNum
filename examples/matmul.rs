use venum::{solid_compile, Buffer, DType, LiquidContext, SolidContext, Tensor};

fn main() -> anyhow::Result<()> {
    // ── Liquid mode (JIT, per-tensor) ───────────────────────────────────
    println!("═══ Liquid (JIT) ═══\n");

    let cx = LiquidContext::new();

    let x = Tensor::arange(&cx, 0.0, 18.0, 1.0)?.reshape(vec![3, 3, 2])?;
    let y = Tensor::arange(&cx, 0.0, 10.0, 1.0)?.reshape(vec![2, 5])?;

    let z = x.matmul(&y)?;

    println!("Graph:\n{}\n", z.render_dag());
    println!("Optimized fused:\n{}\n", z.render_optimized_fused_dag()?);

    let result = z.realize()?;
    println!("Result shape: {:?}", result.shape());
    println!("Result: {:?}\n", result.data());

    // ── Solid mode (AOT, whole-program) ─────────────────────────────────
    println!("═══ Solid (AOT) ═══\n");

    let solid_cx = SolidContext::new();

    // Create placeholder inputs (shapes known at compile time, data provided at runtime)
    let x_solid = Tensor::placeholder(&solid_cx, vec![3, 3, 2], DType::F32);
    let y_solid = Tensor::placeholder(&solid_cx, vec![2, 5], DType::F32);

    // Build computation graph
    let z_solid = x_solid.matmul(&y_solid)?;

    // Compile the program (AOT compilation)
    let program = solid_compile(&solid_cx, &[x_solid.id(), y_solid.id()], &[z_solid.id()])?;

    println!("Input specs: {:?}", program.input_specs);
    println!("Output specs: {:?}", program.output_specs);
    println!("Buffer plan slots: {}\n", program.buffer_plan.slots.len());

    // Prepare input data (same as Liquid mode)
    let x_data: Vec<f32> = (0..18).map(|i| i as f32).collect();
    let y_data: Vec<f32> = (0..10).map(|i| i as f32).collect();

    let x_buf = Buffer::from_f32_vec(x_data);
    let y_buf = Buffer::from_f32_vec(y_data);

    // Execute the compiled program
    let results = program.execute(&[&x_buf, &y_buf])?;

    println!("Result shape: {:?}", program.output_specs[0].shape);
    println!("Result: {:?}\n", results[0].as_f32());

    // Verify results match
    println!("═══ Verification ═══\n");
    let liquid_result = result.data();
    let solid_result = results[0].as_f32();

    if liquid_result == solid_result {
        println!("Results match!");
    } else {
        println!("Results differ!");
        println!("Liquid: {:?}", liquid_result);
        println!("Solid:  {:?}", solid_result);
    }

    Ok(())
}
