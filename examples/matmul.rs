use venum::{Context, Tensor};

fn main() -> anyhow::Result<()> {
    // let a = NaiveTensor::arange(0.0, 18.0, 1.0)?.view(&[3, 3, 2])?;
    // let b = NaiveTensor::arange(0.0, 10.0, 1.0)?.view(&[2, 5])?;
    //
    // println!("{}", a);
    // println!("{}", b);
    //
    // let c = a.matmul(&b)?;
    // println!("{}", c);

    let cx = Context::new();

    let x = Tensor::arange(&cx, 0.0, 18.0, 1.0)?.reshape(vec![3, 3, 2])?;
    let y = Tensor::arange(&cx, 0.0, 10.0, 1.0)?.reshape(vec![2, 5])?;

    let z = x.matmul(&y)?;

    println!("{}", z.render_dag());
    println!("{}", z.render_fused_dag());
    println!("{}", z.render_kernels()?);

    Ok(())
}
