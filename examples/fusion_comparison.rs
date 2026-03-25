use venum::{Buffer, DType, LiquidContext, SolidContext, Tensor, compile};

/// This example demonstrates the differences between Liquid and Solid fusion policies.
///
/// Key scenario: An intermediate value is computed and then reused multiple times.
///
/// Computation:
///   temp = a * 2.0        (intermediate with multiple consumers)
///   x = temp + b
///   y = temp - c
///   result = x * y
///
/// Expected behavior:
/// - Liquid (conservative): temp must materialize since it has 2 consumers
///   Result: 3 kernels (temp, x and y separately, result)
///
/// - Solid (aggressive): temp can be inlined into both x and y consumers
///   Result: Potentially 1-2 kernels with temp computation duplicated but fused
fn main() -> anyhow::Result<()> {
    println!("═══ Liquid (JIT - Conservative Fusion) ═══\n");

    let cx = LiquidContext::new();

    let a = Tensor::from_slice_f32_1d(&cx, &[1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice_f32_1d(&cx, &[0.5, 0.5, 0.5, 0.5]);
    let c = Tensor::from_slice_f32_1d(&cx, &[1.0, 1.0, 1.0, 1.0]);
    let two = Tensor::from_slice_f32_1d(&cx, &[2.0, 2.0, 2.0, 2.0]);

    // Create multi-consumer intermediate
    let temp = a.mul(&two)?;

    // Use temp in two different computations
    let x = temp.add(&b)?;
    let y = temp.sub(&c)?;

    // Combine results
    let result = x.mul(&y)?;

    println!("Graph structure:");
    println!("  temp = a * 2.0    (intermediate with 2 consumers)");
    println!("  x = temp + b");
    println!("  y = temp - c");
    println!("  result = x * y\n");

    println!("Raw DAG:\n{}\n", result.render_dag());
    let fused_dag = result.render_optimized_fused_dag()?;
    println!("Optimized fused DAG:\n{}\n", fused_dag);

    let out = result.realize()?;
    println!("Result: {:?}\n", out.data());

    // Count kernels in Liquid schedule
    let liquid_kernel_count = count_kernels_from_output(&fused_dag);
    println!("Liquid: {} fused kernels\n", liquid_kernel_count);

    // ── Solid mode (AOT, aggressive fusion) ────────────────────────────
    println!("═══ Solid (AOT - Aggressive Fusion) ═══\n");

    let cx1 = SolidContext::new();

    let a1 = Tensor::placeholder(&cx1, DType::F32, vec![4]);
    let b1 = Tensor::placeholder(&cx1, DType::F32, vec![4]);
    let c1 = Tensor::placeholder(&cx1, DType::F32, vec![4]);
    let two1 = Tensor::placeholder(&cx1, DType::F32, vec![4]);

    // Same computation structure
    let temp1 = a1.mul(&two1)?;
    let x1 = temp1.add(&b1)?;
    let y1 = temp1.sub(&c1)?;
    let result1 = x1.mul(&y1)?;

    // Compile
    let program = compile(
        &cx1,
        &[a1.id(), two1.id(), b1.id(), c1.id()],
        &[result1.id()],
    )?;

    println!("Compiled program stats:");
    println!("  Kernels: {}", program.num_kernels());
    println!("  Execution steps: {}\n", program.num_steps());

    println!("Compiled graph:\n{}\n", program.render_compiled_graph());

    // Execute with same data
    let a_buf = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let two_buf = Buffer::from_f32_vec(vec![2.0, 2.0, 2.0, 2.0]);
    let b_buf = Buffer::from_f32_vec(vec![0.5, 0.5, 0.5, 0.5]);
    let c_buf = Buffer::from_f32_vec(vec![1.0, 1.0, 1.0, 1.0]);

    let results = program.execute(&[&a_buf, &two_buf, &b_buf, &c_buf])?;
    println!("Result: {:?}\n", results[0].as_f32());

    println!("Solid: {} compiled kernels\n", program.num_kernels());

    // ── Comparison ──────────────────────────────────────────────────────
    println!("═══ Comparison ═══\n");

    let liquid_result = out.data();
    let solid_result = results[0].as_f32();

    if liquid_result == solid_result {
        println!("Results match!");
    } else {
        println!("Results differ!");
        println!("Liquid: {:?}", liquid_result);
        println!("Solid:  {:?}", solid_result);
    }

    println!("\nFusion efficiency:");
    println!(
        "  Liquid: {} kernels (conservative - must materialize multi-consumer 'temp')",
        liquid_kernel_count
    );
    println!(
        "  Solid:  {} kernels (aggressive - can inline 'temp' into consumers)",
        program.num_kernels()
    );

    if program.num_kernels() < liquid_kernel_count {
        println!("\nSolid achieved better fusion by inlining multi-consumer intermediate!");
    } else if program.num_kernels() == liquid_kernel_count {
        println!("\nBoth modes achieved similar fusion for this graph.");
    }

    Ok(())
}

/// Count the number of "Fused Kernel" subgraphs in the Mermaid output
fn count_kernels_from_output(mermaid: &str) -> usize {
    mermaid
        .lines()
        .filter(|line| line.contains("Fused Kernel"))
        .count()
}
