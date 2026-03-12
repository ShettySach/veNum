use venum::ETensor;

fn main() -> anyhow::Result<()> {
    let a = ETensor::linspace(1, 5, 5)?;
    println!("a");
    println!("{}", &a);

    let b = ETensor::linspace(1, 5, 5)?.view(&[5, 1])?;
    println!("b");
    println!("{}", &b);

    let prod = (&a * &b)?;
    println!("a * b");
    println!("{}", &prod);

    Ok(())
}
