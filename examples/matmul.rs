use venum::ETensor;

fn main() -> anyhow::Result<()> {
    let a = ETensor::arange(0.25, 50.25, 1.0)?.view(&[5, 5, 2])?;
    let b = ETensor::arange(0.75, 10.75, 1.0)?.view(&[2, 5])?;
    println!("{}", a);
    println!("{}", b);

    let c = a.matmul_nd(&b)?;
    println!("{}", c);

    Ok(())
}
