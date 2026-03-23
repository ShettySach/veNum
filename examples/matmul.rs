use venum::{solid_compile, Context, DType, SolidContext, SolidTensor, Tensor};

fn main() -> anyhow::Result<()> {
    // ── Liquid mode (JIT, per-tensor) ───────────────────────────────────
    println!("═══ Liquid (JIT) ═══\n");

    let cx = Context::new();

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

    let scx = SolidContext::new();

    let a = SolidTensor::placeholder(&scx, vec![3, 3, 2], DType::F32);
    let b = SolidTensor::placeholder(&scx, vec![2, 5], DType::F32);

    let c = a.matmul(&b)?;

    // Compile entire graph ahead-of-time
    let program = solid_compile(&scx, &[a.id(), b.id()], &[c.id()])?;
    println!(
        "Compiled: {} kernel(s), {} step(s)\n",
        program.num_kernels(),
        program.num_steps()
    );

    // Execute with concrete data (reusable)
    let a_data: Vec<f32> = (0..18).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let a_buf = venum::Scalar::F32(0.0); // unused, just for Buffer construction
    let _ = a_buf;
    let a_buf = venum::DType::F32;
    let _ = a_buf;

    let a_buf = crate_buffer_f32(&a_data);
    let b_buf = crate_buffer_f32(&b_data);

    let results = program.execute(&[&a_buf, &b_buf])?;

    println!("Result shape: {:?}", program.output_specs[0].shape);
    println!("Result: {:?}\n", results[0].as_f32());

    // Run again with different data to show program reuse
    let a2: Vec<f32> = (0..18).map(|i| (i as f32) * 0.1).collect();
    let b2: Vec<f32> = (0..10).map(|i| (i as f32) * 2.0).collect();
    let a2_buf = crate_buffer_f32(&a2);
    let b2_buf = crate_buffer_f32(&b2);

    let results2 = program.execute(&[&a2_buf, &b2_buf])?;
    println!("Reuse with new data: {:?}", results2[0].as_f32());

    Ok(())
}

fn crate_buffer_f32(data: &[f32]) -> venum::DType {
    todo!()
}
