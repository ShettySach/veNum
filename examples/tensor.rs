use anyhow::Result;

use venum::{Buffer, Context, DType, Tensor, run_context};

fn main() -> Result<()> {
    let cx = Context::new();

    // Create a 5x5 tensor
    let a = Tensor::placeholder(&cx, DType::F32, vec![5, 5]);
    println!("Original shape: {:?}", a.shape());

    // Slice to get a 2x2 view: a[1:3, 2:4]
    let b = a.slice(vec![(1, 3), (2, 4)])?;
    println!("Sliced shape: {:?}", b.shape());

    // Run with actual data
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
    ];

    println!("Input matrix (5x5):");
    for i in 0..5 {
        let row: Vec<f32> = input_data[i * 5..(i + 1) * 5].to_vec();
        println!("  {:?}", row);
    }

    let outputs = run_context(&cx, &[b.id()], &[Buffer::F32(input_data)])?;

    println!("\nSliced output (a[1:3, 2:4]):");
    if let Buffer::F32(data) = &outputs[0] {
        for i in 0..2 {
            let row: Vec<f32> = data[i * 2..(i + 1) * 2].to_vec();
            println!("  {:?}", row);
        }
    }

    Ok(())
}
