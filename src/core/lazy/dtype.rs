use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
}

impl Scalar {
    pub fn dtype(&self) -> DType {
        match self {
            Scalar::F32(_) => DType::F32,
            Scalar::F64(_) => DType::F64,
            Scalar::I32(_) => DType::I32,
            Scalar::I64(_) => DType::I64,
        }
    }

    pub fn as_f32(&self) -> f32 {
        match self {
            Scalar::F32(v) => *v,
            _ => panic!("Scalar is not F32"),
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            Scalar::F64(v) => *v,
            _ => panic!("Scalar is not F64"),
        }
    }

    pub fn as_i32(&self) -> i32 {
        match self {
            Scalar::I32(v) => *v,
            _ => panic!("Scalar is not I32"),
        }
    }

    pub fn as_i64(&self) -> i64 {
        match self {
            Scalar::I64(v) => *v,
            _ => panic!("Scalar is not I64"),
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            Scalar::F32(v) => *v as f64,
            Scalar::F64(v) => *v,
            Scalar::I32(v) => *v as f64,
            Scalar::I64(v) => *v as f64,
        }
    }

    pub fn from_f64(val: f64, dtype: DType) -> Scalar {
        match dtype {
            DType::F32 => Scalar::F32(val as f32),
            DType::F64 => Scalar::F64(val),
            DType::I32 => Scalar::I32(val as i32),
            DType::I64 => Scalar::I64(val as i64),
        }
    }
}

impl Hash for Scalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Scalar::F32(v) => v.to_bits().hash(state),
            Scalar::F64(v) => v.to_bits().hash(state),
            Scalar::I32(v) => v.hash(state),
            Scalar::I64(v) => v.hash(state),
        }
    }
}

impl Eq for Scalar {}

#[derive(Clone, Debug)]
pub enum Buffer {
    F32(Arc<Vec<f32>>),
    F64(Arc<Vec<f64>>),
    I32(Arc<Vec<i32>>),
    I64(Arc<Vec<i64>>),
}

impl Buffer {
    pub fn dtype(&self) -> DType {
        match self {
            Buffer::F32(_) => DType::F32,
            Buffer::F64(_) => DType::F64,
            Buffer::I32(_) => DType::I32,
            Buffer::I64(_) => DType::I64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Buffer::F32(v) => v.len(),
            Buffer::F64(v) => v.len(),
            Buffer::I32(v) => v.len(),
            Buffer::I64(v) => v.len(),
        }
    }

    pub fn as_f32(&self) -> &[f32] {
        match self {
            Buffer::F32(v) => v,
            _ => panic!("Buffer is not F32"),
        }
    }

    pub fn as_f64(&self) -> &[f64] {
        match self {
            Buffer::F64(v) => v,
            _ => panic!("Buffer is not F64"),
        }
    }

    pub fn as_i32(&self) -> &[i32] {
        match self {
            Buffer::I32(v) => v,
            _ => panic!("Buffer is not I32"),
        }
    }

    pub fn as_i64(&self) -> &[i64] {
        match self {
            Buffer::I64(v) => v,
            _ => panic!("Buffer is not I64"),
        }
    }

    pub fn as_f32_ptr(&self) -> *const f32 {
        match self {
            Buffer::F32(v) => v.as_ptr(),
            _ => panic!("Buffer is not F32"),
        }
    }

    pub fn as_ptr_u8(&self) -> *const u8 {
        match self {
            Buffer::F32(v) => v.as_ptr() as *const u8,
            Buffer::F64(v) => v.as_ptr() as *const u8,
            Buffer::I32(v) => v.as_ptr() as *const u8,
            Buffer::I64(v) => v.as_ptr() as *const u8,
        }
    }

    pub fn from_f32_vec(data: Vec<f32>) -> Buffer {
        Buffer::F32(Arc::new(data))
    }

    pub fn from_f64_vec(data: Vec<f64>) -> Buffer {
        Buffer::F64(Arc::new(data))
    }

    pub fn from_i32_vec(data: Vec<i32>) -> Buffer {
        Buffer::I32(Arc::new(data))
    }

    pub fn from_i64_vec(data: Vec<i64>) -> Buffer {
        Buffer::I64(Arc::new(data))
    }
}

pub fn zeros(dtype: DType, numel: usize) -> Vec<u8> {
    vec![0u8; numel * dtype.size_bytes()]
}

#[derive(Clone, Debug)]
pub struct RealizedTensor {
    buffer: Buffer,
    shape: Vec<usize>,
}

impl RealizedTensor {
    pub fn new(buffer: Buffer, shape: Vec<usize>) -> Self {
        Self { buffer, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn sizes(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.buffer.dtype()
    }

    pub fn data(&self) -> &[f32] {
        self.buffer.as_f32()
    }

    pub fn data_f64(&self) -> &[f64] {
        self.buffer.as_f64()
    }

    pub fn data_i32(&self) -> &[i32] {
        self.buffer.as_i32()
    }

    pub fn data_i64(&self) -> &[i64] {
        self.buffer.as_i64()
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn numel(&self) -> usize {
        self.buffer.len()
    }
}
