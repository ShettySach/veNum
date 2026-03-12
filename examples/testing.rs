use venum::ETensor;

fn main() -> anyhow::Result<()> {
    let a = ETensor::arange(0, 27, 1)?.view(&[3, 3, 3])?;
    println!("{}", a);

    let x = a.index_dims(&[2], &[1])?;
    println!("{x}");

    Ok(())
}
