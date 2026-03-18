use venum::{Context, Tensor};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    // 1. Simple add — flat indexing, no trackers.
    println!("=== a + b (flat) ===");
    let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
    let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
    let c = (&a + &b)?;
    println!("{}", c.render_kernels()?);

    // 2. Expand + add — shape op absorbed, tracker with stride=0.
    println!("=== expand([3,4]) + b (tracker) ===");
    let cx2 = Context::new();
    let a2 = Tensor::from_slice(&cx2, &[1.0, 2.0, 3.0], vec![3, 1]);
    let b2 = Tensor::from_slice(&cx2, &[10.0, 20.0, 30.0, 40.0,
                                          50.0, 60.0, 70.0, 80.0,
                                          90.0, 100.0, 110.0, 120.0], vec![3, 4]);
    let c2 = (&a2.expand(vec![3, 4])? + &b2)?;
    println!("{}", c2.render_kernels()?);

    // 3. Matmul — reshape + expand absorbed.
    println!("=== matmul [2,3] @ [3,2] ===");
    let cx3 = Context::new();
    let a3 = Tensor::from_slice(&cx3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b3 = Tensor::from_slice(&cx3, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let mm = a3.matmul(&b3)?;
    println!("{}", mm.render_kernels()?);

    Ok(())
}
