use venum::{Buffer, DType, LiquidContext, SolidContext, Tensor, compile};

fn main() -> anyhow::Result<()> {
    // ── Liquid mode (JIT, per-tensor) ───────────────────────────────────
    println!("═══ Liquid (JIT) ═══\n");

    let cx = LiquidContext::new();

    let a = Tensor::from_slice_f32_1d(&cx, &[1.0, 2.0, 3.0, 4.0]).reshape(vec![1, 4])?;
    let b = Tensor::from_slice_f32_1d(&cx, &[1.0, 1.0, 1.0, 1.0]).reshape(vec![1, 4])?;

    let c = Tensor::from_slice_f32_1d(&cx, &[2.0, 2.0, 2.0, 2.0]).reshape(vec![4, 1])?;
    let d = Tensor::from_slice_f32_1d(&cx, &[0.5, 0.5, 0.5, 0.5]).reshape(vec![4, 1])?;

    let w = a.mul(&b)?.add(&a)?;
    let x = c.mul(&d)?.sub(&c)?;
    let y = w.add(&x)?;

    println!("Graph:\n{}\n", y.render_dag());
    println!("Optimized fused:\n{}\n", y.render_optimized_fused_dag()?);

    let out = y.realize()?;
    println!("Result shape: {:?}", out.sizes());
    println!("Result data: {:?}\n", out.data());

    // ── Solid mode (AOT, whole-program) ─────────────────────────────────
    println!("═══ Solid (AOT) ═══\n");

    let cx1 = SolidContext::new();

    // Create placeholder inputs (shapes known at compile time, data provided at runtime)
    let a1 = Tensor::placeholder(&cx1, DType::F32, vec![1, 4]);
    let b1 = Tensor::placeholder(&cx1, DType::F32, vec![1, 4]);
    let c1 = Tensor::placeholder(&cx1, DType::F32, vec![4, 1]);
    let d1 = Tensor::placeholder(&cx1, DType::F32, vec![4, 1]);

    // Build computation graph (same structure as Liquid)
    let w1 = a1.mul(&b1)?.add(&a1)?;
    let x1 = c1.mul(&d1)?.sub(&c1)?;
    let y1 = w1.add(&x1)?;

    // Compile the program (AOT compilation)
    let program = compile(&cx1, &[a1.id(), b1.id(), c1.id(), d1.id()], &[y1.id()])?;

    println!("Input specs: {:?}", program.input_specs);
    println!("Output specs: {:?}", program.output_specs);
    println!("Compiled kernels: {}", program.num_kernels());
    println!("Execution steps: {}\n", program.num_steps());

    println!("Compiled graph:\n{}\n", program.render_compiled_graph());

    // Prepare input data (same as Liquid mode)
    let a_buf = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b_buf = Buffer::from_f32_vec(vec![1.0, 1.0, 1.0, 1.0]);
    let c_buf = Buffer::from_f32_vec(vec![2.0, 2.0, 2.0, 2.0]);
    let d_buf = Buffer::from_f32_vec(vec![0.5, 0.5, 0.5, 0.5]);

    // Execute the compiled program
    let results = program.execute(&[&a_buf, &b_buf, &c_buf, &d_buf])?;

    println!("Result shape: {:?}", program.output_specs[0].shape);
    println!("Result data: {:?}\n", results[0].as_f32());

    // Verify results match
    println!("═══ Verification ═══\n");
    let liquid_result = out.data();
    let solid_result = results[0].as_f32();

    if liquid_result == solid_result {
        println!("✓ Results match!");
    } else {
        println!("✗ Results differ!");
        println!("Liquid: {:?}", liquid_result);
        println!("Solid:  {:?}", solid_result);
    }

    Ok(())
}
