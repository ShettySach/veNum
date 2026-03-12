use venum::LTensor;

fn main() {
    let a = LTensor::from_slice(&[2.0, 3.0, 4.0, 5.0], vec![4]);
    let b = LTensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);

    // (a * b + a) - b
    let c = &(&(&a * &b) + &a) - &b;

    println!("=== RAW DAG ===");
    println!("{}", c.render_dag());
    println!();
    println!("=== FUSED DAG ===");
    println!("{}", c.render_fused_dag());
}
