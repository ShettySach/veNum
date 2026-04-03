use super::Dim;
use super::graph::HLIRGraph;
use super::op::ReduceOp;
use super::types::{NodeId, Scalar};

pub fn sub(graph: &mut HLIRGraph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let neg_rhs = graph.unary(rhs, super::op::Op::Neg);
    graph.binary(lhs, neg_rhs, super::op::Op::Add)
}

/// cos(x) = sin(x + π/2)
pub fn cos(graph: &mut HLIRGraph, input: NodeId) -> NodeId {
    let ty = graph.ty(input);
    let dtype = ty.dtype;
    let shape = ty.shape.clone();
    let half_pi = graph.constant(
        Scalar::from_f64(std::f64::consts::FRAC_PI_2, dtype),
        shape,
        dtype,
    );
    let shifted = graph.binary(input, half_pi, super::op::Op::Add);
    graph.unary(shifted, super::op::Op::Sin)
}

pub fn div(graph: &mut HLIRGraph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let recip_rhs = graph.unary(rhs, super::op::Op::Recip);
    graph.binary(lhs, recip_rhs, super::op::Op::Mul)
}

pub fn matmul(graph: &mut HLIRGraph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let lhs_ty = graph.ty(lhs).clone();
    let rhs_ty = graph.ty(rhs).clone();

    let lhs_rank = lhs_ty.shape.len();
    let rhs_rank = rhs_ty.shape.len();
    assert!(lhs_rank >= 2 && rhs_rank >= 2, "matmul requires rank >= 2");

    let m = lhs_ty.shape[lhs_rank - 2].clone();
    let k = lhs_ty.shape[lhs_rank - 1].clone();
    let n = rhs_ty.shape[rhs_rank - 1].clone();

    let lhs_batch = lhs_ty.shape[..lhs_rank - 2].to_vec();
    let rhs_batch = rhs_ty.shape[..rhs_rank - 2].to_vec();
    let batch_rank = lhs_batch.len().max(rhs_batch.len());

    let mut batch = Vec::with_capacity(batch_rank);
    for i in 0..batch_rank {
        let l = if i < batch_rank - lhs_batch.len() {
            Dim::constant(1)
        } else {
            lhs_batch[i - (batch_rank - lhs_batch.len())].clone()
        };
        let r = if i < batch_rank - rhs_batch.len() {
            Dim::constant(1)
        } else {
            rhs_batch[i - (batch_rank - rhs_batch.len())].clone()
        };
        batch.push(broadcast_dim(l, r));
    }

    let mut lhs_reshape = vec![Dim::constant(1); batch_rank - lhs_batch.len()];
    lhs_reshape.extend(lhs_batch);
    lhs_reshape.push(m.clone());
    lhs_reshape.push(k.clone());
    lhs_reshape.push(Dim::constant(1));

    let mut rhs_reshape = vec![Dim::constant(1); batch_rank - rhs_batch.len()];
    rhs_reshape.extend(rhs_batch);
    rhs_reshape.push(Dim::constant(1));
    rhs_reshape.push(k.clone());
    rhs_reshape.push(n.clone());

    let mut lhs_expand = batch.clone();
    lhs_expand.push(m.clone());
    lhs_expand.push(k.clone());
    lhs_expand.push(n.clone());

    let mut rhs_expand = batch;
    rhs_expand.push(m);
    rhs_expand.push(k);
    rhs_expand.push(n);

    let lhs_rs = graph.reshape(lhs, lhs_reshape);
    let rhs_rs = graph.reshape(rhs, rhs_reshape);
    let lhs_exp = graph.expand(lhs_rs, lhs_expand);
    let rhs_exp = graph.expand(rhs_rs, rhs_expand);

    let prod = graph.binary(lhs_exp, rhs_exp, super::op::Op::Mul);
    let k_axis = graph.ty(prod).shape.len() - 2;
    graph.reduce(prod, vec![k_axis], ReduceOp::Sum, false)
}

fn broadcast_dim(lhs: Dim, rhs: Dim) -> Dim {
    match (&lhs, &rhs) {
        (Dim::Const(a), Dim::Const(b)) if a == b => lhs,
        (Dim::Const(1), _) => rhs,
        (_, Dim::Const(1)) => lhs,
        _ if lhs == rhs => lhs,
        _ => panic!("incompatible dims for broadcast: {:?} vs {:?}", lhs, rhs),
    }
}
