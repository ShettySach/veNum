use crate::core::dtype::{Buffer, DType};

use super::{NodeId, Op};

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub buffer: Option<Buffer>,
}

impl Node {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}
