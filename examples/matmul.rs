use std::error::Error;
use venum::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let a = Tensor::arange(0, 50, 1)?.view(&[5, 5, 2])?;
    let b = Tensor::arange(0, 10, 1)?.view(&[2, 5])?;
    println!("{}", a);
    println!("{}", b);

    let c = &a.matmul(&b)?;
    println!("{}", c);

    Ok(())
}
