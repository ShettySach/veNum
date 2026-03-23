use std::ops::{Add, Sub};
use venum::{LiquidContext, Tensor};

fn main() -> anyhow::Result<()> {
    let cx = LiquidContext::new();

    let a = Tensor::from_slice_f32_1d(&cx, &[1.0, 2.0, 3.0, 4.0]).reshape(vec![1, 4])?;
    let b = Tensor::from_slice_f32_1d(&cx, &[1.0, 1.0, 1.0, 1.0]).reshape(vec![1, 4])?;

    // TODO: Fix error

    let c = Tensor::from_slice_f32_1d(&cx, &[2.0, 2.0, 2.0, 2.0]).reshape(vec![4, 1])?;
    let d = Tensor::from_slice_f32_1d(&cx, &[0.5, 0.5, 0.5, 0.5]).reshape(vec![4, 1])?;

    let w = a.mul(&b)?.add(&a)?;
    let x = c.mul(&d)?.sub(&c)?;
    let y = w.add(x)?;

    println!("{}\n", y.render_dag());
    println!("{}\n", y.render_optimized_fused_dag()?);

    let out = y.realize()?;
    println!("result shape: {:?}", out.sizes());
    println!("result data: {:?}", out.data());

    Ok(())
}
