use anyhow::{bail, Result};
use std::borrow::Cow;

use super::super::dtype::{Buffer, DType, Scalar};
use super::super::graph::{Graph, NodeId};

#[allow(dead_code)]
pub(crate) fn get_realized_buffer<'a>(
    graph: &'a Graph,
    node_id: NodeId,
    realized: &'a std::collections::HashMap<NodeId, Buffer>,
    context: &str,
) -> Result<Cow<'a, Buffer>> {
    if let Some(buf) = realized.get(&node_id) {
        Ok(Cow::Borrowed(buf))
    } else if let Some(ref buf) = graph.node(node_id).buffer {
        Ok(Cow::Borrowed(buf))
    } else {
        bail!("{} input {:?} is not realized", context, node_id);
    }
}

pub(crate) fn scalar_fill_buffer(val: Scalar, numel: usize) -> Buffer {
    match val {
        Scalar::F32(v) => Buffer::from_f32_vec(vec![v; numel]),
        Scalar::F64(v) => Buffer::from_f64_vec(vec![v; numel]),
        Scalar::I32(v) => Buffer::from_i32_vec(vec![v; numel]),
        Scalar::I64(v) => Buffer::from_i64_vec(vec![v; numel]),
    }
}

pub(crate) fn buffer_from_bytes(bytes: Vec<u8>, dtype: DType) -> Buffer {
    match dtype {
        DType::F32 => {
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            Buffer::from_f32_vec(data)
        }
        DType::F64 => {
            let data: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            Buffer::from_f64_vec(data)
        }
        DType::I32 => {
            let data: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            Buffer::from_i32_vec(data)
        }
        DType::I64 => {
            let data: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            Buffer::from_i64_vec(data)
        }
    }
}
