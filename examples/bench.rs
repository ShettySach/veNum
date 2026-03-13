use venum::NaiveTensor;

fn main() -> anyhow::Result<()> {
    let a = NaiveTensor::arange(0, 1_000_000, 1)?.view(&[1000, 1000])?;
    let b = NaiveTensor::arange(1_000_000, 0, -1)?.view(&[1000, 1000])?;

    for _ in 0..10 {
        let now = std::time::Instant::now();

        let _c = &a.matmul(&b)?;

        let end = now.elapsed();
        println!("{:?}", end);
    }

    Ok(())
}
