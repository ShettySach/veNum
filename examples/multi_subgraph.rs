use venum::{run_context, Buffer, Context, DType, Tensor};

fn main() -> anyhow::Result<()> {
    let cx = Context::new();

    let a = Tensor::placeholder(&cx, DType::F32, vec![1, 4]);
    let b = Tensor::placeholder(&cx, DType::F32, vec![1, 4]);
    let c = Tensor::placeholder(&cx, DType::F32, vec![4, 1]);
    let d = Tensor::placeholder(&cx, DType::F32, vec![4, 1]);

    let w = a.mul(&b)?.add(&a)?;
    let x = c.mul(&d)?.sub(&c)?;
    let y = w.add(&x)?;

    let out = run_context(
        &cx,
        &[y.id()],
        &[
            Buffer::F32(vec![1.0, 2.0, 3.0, 4.0]),
            Buffer::F32(vec![1.0, 1.0, 1.0, 1.0]),
            Buffer::F32(vec![2.0, 2.0, 2.0, 2.0]),
            Buffer::F32(vec![0.5, 0.5, 0.5, 0.5]),
        ],
    )?;

    println!("multi-subgraph graph nodes: {}", cx.num_nodes());
    match &out[0] {
        Buffer::F32(v) => println!("result: {:?}", v),
        _ => println!("unexpected output dtype"),
    }

    Ok(())
}
