use venum::{NaiveTensor, Tensor};

fn main() -> anyhow::Result<()> {
    let a = NaiveTensor::arange(0.25, 50.25, 1.0)?.view(&[5, 5, 2])?;
    let b = NaiveTensor::arange(0.75, 10.75, 1.0)?.view(&[2, 5])?;
    println!("{}", a);
    println!("{}", b);

    let c = a.matmul(&b)?;
    println!("{}", c);

    let cx = venum::Context::new();
    let x = Tensor::from_slice(&cx, &[1., 2., 3., 4.], vec![2, 2]);
    let y = Tensor::from_slice(&cx, &[1., 2., 3., 4.], vec![2, 2]);

    let z = x.matmul(&y)?;

    println!("RAW DAG");
    println!("{}", z.render_dag());
    println!();
    println!("FUSED DAG");
    println!("{}", z.render_fused_dag());

    println!("{:?}", z.realize()?.data());

    Ok(())
}
