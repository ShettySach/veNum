use venum::{Context, Tensor};

fn main() {
    let cx = Context::new();

    let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);
    let b = Tensor::from_slice(&cx, &[1.0, 1.0, 1.0, 1.0], vec![4]);

    let c = (&a * &b + &a) - &b;

    println!("=== RAW DAG ===");
    println!("{}", c.render_dag());
    println!();
    println!("=== FUSED DAG ===");
    println!("{}", c.render_fused_dag());
}
