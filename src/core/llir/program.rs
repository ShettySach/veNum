use crate::core::hlir::{NodeId, Op, TensorType};

use super::loop_nest::LoopNest;
use super::memory::BufferAlloc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    pub id: KernelId,
    pub name: String,
    pub root: NodeId,
    pub op: Op,
    pub ty: TensorType,
    pub loop_nest: LoopNest,
    pub allocs: Vec<BufferAlloc>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LLIRProgram {
    pub kernels: Vec<Kernel>,
}
