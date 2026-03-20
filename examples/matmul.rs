use std::time::Instant;
use venum::{NaiveTensor, Tensor};

fn main() -> anyhow::Result<()> {
    let a = NaiveTensor::arange(0.0, 18.0, 1.0)?.view(&[3, 3, 2])?;
    let b = NaiveTensor::arange(0.0, 10.0, 1.0)?.view(&[2, 5])?;

    println!("{}", a);
    println!("{}", b);

    let c = a.matmul(&b)?;
    println!("{}", c);

    let cx = venum::Context::new();

    let x = Tensor::arange(&cx, 0.0, 18.0, 1.0)?.reshape(vec![3, 3, 2])?;
    let y = Tensor::arange(&cx, 0.0, 10.0, 1.0)?.reshape(vec![2, 5])?;

    let z = x.matmul(&y)?;

    println!("RAW DAG");
    println!("{}", z.render_dag());
    println!();
    println!("FUSED DAG");
    println!("{}", z.render_fused_dag());
    println!();
    println!("KERNELS");
    println!("{}", z.render_kernels()?);

    // Demonstrate the graph-level JIT plan cache + kernel cache:
    // the first realize builds the execution plan / compiles kernels,
    // subsequent realizes should be much faster.
    let t0 = Instant::now();
    let out0 = z.realize()?;
    let dt0 = t0.elapsed();

    let iters = 200usize;
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = z.realize()?;
    }
    let dt1 = t1.elapsed();
    let avg_ms = (dt1.as_secs_f64() * 1000.0) / (iters as f64);

    // Build an equivalent expression again; it should hit the plan cache.
    let z2 = x.matmul(&y)?;
    let t2 = Instant::now();
    let _ = z2.realize()?;
    let dt2 = t2.elapsed();

    println!("realize (first): {:?}", dt0);
    println!("realize (cached): {:?} total, {:.4} ms/iter", dt1, avg_ms);
    println!("realize (same graph signature, new node): {:?}", dt2);
    println!("{:?}", out0);

    Ok(())
}
