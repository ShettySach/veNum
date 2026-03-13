use venum::{Context, Tensor};

fn main() {
    let cx = Context::new();

    let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
    let b = Tensor::from_slice(&cx, &[1.0, 1.0, 1.0, 1.0], vec![4]);
    let c = Tensor::from_slice(&cx, &[2.0, 2.0, 2.0, 2.0], vec![4]);
    let d = Tensor::from_slice(&cx, &[0.5, 0.5, 0.5, 0.5], vec![4]);

    // Branch 1: (a * b) + a -> [4] then reshape to [2,2]
    let w1 = &a * &b;
    let w2 = &w1 + &a;
    let w3 = w2.reshape(vec![2, 2]).expect("reshape failed"); // [2,2]

    // Branch 2: (c * d) + c -> [4] then reshape to [2,2]
    let x1 = &c * &d;
    let x2 = &x1 + &c;
    let x3 = x2.reshape(vec![2, 2]).expect("reshape failed"); // [2,2]

    // Final combination: w3 + x3 -> [2,2]
    let y = &w3 + &x3;

    println!("=== RAW DAG ===");
    println!("{}", y.render_dag());
    println!();
    println!("=== FUSED DAG ===");
    println!("{}", y.render_fused_dag());
}
