use venum::ETensor;

fn main() -> anyhow::Result<()> {
    let tensor = ETensor::arange(0, 9, 1)?.view(&[3, 3])?.flip(&[0])?;
    println!("{}", tensor);

    Ok(())
}
