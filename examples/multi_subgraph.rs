use std::ops::{Add, Mul, Sub};

use venum::{Context, Tensor};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    let a = Tensor::from_slice_f32_1d(&cx, &[1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice_f32_1d(&cx, &[1.0, 1.0, 1.0, 1.0]);
    let c = Tensor::from_slice_f32_1d(&cx, &[2.0, 2.0, 2.0, 2.0]);
    let d = Tensor::from_slice_f32_1d(&cx, &[0.5, 0.5, 0.5, 0.5]);

    let w = (&a).mul(&b)?.add(&a)?.reshape(vec![2, 2])?;
    let x = (&c).mul(&d)?.sub(&c)?.reshape(vec![2, 2])?;
    let y = w.add(x)?;

    println!("{}\n", y.render_dag());
    println!("{}\n", y.render_optimized_fused_dag()?);

    let out = y.realize()?;
    println!("result shape: {:?}", out.sizes());
    println!("result data: {:?}", out.data());

    Ok(())
}
