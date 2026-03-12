use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Buffer {
    F32(Arc<Vec<f32>>),
}

impl Buffer {
    pub fn dtype(&self) -> DType {
        match self {
            Buffer::F32(_) => DType::F32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Buffer::F32(v) => v.len(),
        }
    }

    pub fn as_f32(&self) -> &[f32] {
        match self {
            Buffer::F32(v) => v,
        }
    }

    pub fn as_f32_ptr(&self) -> *const f32 {
        match self {
            Buffer::F32(v) => v.as_ptr(),
        }
    }
}
