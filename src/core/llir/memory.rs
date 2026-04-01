use crate::core::hlir::{BufferId, DType};

use super::affine::AffineExpr;

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryAccess {
    pub buffer: BufferId,
    pub indices: Vec<AffineExpr>,
    pub access_kind: AccessKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BufferAlloc {
    pub id: BufferId,
    pub shape: Vec<AffineExpr>,
    pub dtype: DType,
    pub memory_space: MemorySpace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemorySpace {
    Global,
    Shared,
    Local,
    Constant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessKind {
    Read,
    Write,
    ReadWrite,
}
