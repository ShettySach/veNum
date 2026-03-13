use venum::NaiveTensor;

fn main() -> anyhow::Result<()> {
    let a = NaiveTensor::linspace(1, 5, 5)?;
    println!("a");
    println!("{}", &a);

    let b = NaiveTensor::linspace(1, 5, 5)?.view(&[5, 1])?;
    println!("b");
    println!("{}", &b);

    let prod = (&a * &b)?;
    println!("a * b");
    println!("{}", &prod);

    Ok(())
}
