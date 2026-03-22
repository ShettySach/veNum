use venum::{Context, Tensor};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);

    // Build a shape-heavy graph that Phase 2 can optimize:
    // - double reshape collapses
    // - transpose twice on same dims cancels
    // - elementwise on matching reshapes can canonicalize
    let a_view = a.reshape(vec![1, 2, 2])?.reshape(vec![1, 2, 2])?;
    let b_view = b.reshape(vec![1, 2, 2])?;

    let y = (&a_view + &b_view)?
        .transpose(1, 2)?
        .transpose(1, 2)?
        .squeeze()?
        .unsqueeze(3)?;

    println!("=== RAW DAG ===");
    println!("{}", y.render_dag());
    println!();

    println!("=== RAW FUSED DAG ===");
    println!("{}", y.render_fused_dag());
    println!();

    println!("=== OPTIMIZED DAG (Phase 2) ===");
    println!("{}", y.render_optimized_dag()?);
    println!();

    println!("=== OPTIMIZED FUSED DAG (Phase 2) ===");
    println!("{}", y.render_optimized_fused_dag()?);
    println!();

    let out = y.realize()?;
    println!("result shape: {:?}", out.sizes());
    println!("result data: {:?}", out.data());

    Ok(())
}
