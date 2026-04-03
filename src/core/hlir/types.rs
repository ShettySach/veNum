use super::dim::Dim;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    F32(f32),
    F16(u16),
    BF16(u16),
    F64(f64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Bool(bool),
}

impl Scalar {
    pub fn dtype(&self) -> DType {
        match self {
            Scalar::F32(_) => DType::F32,
            Scalar::F16(_) => DType::F16,
            Scalar::BF16(_) => DType::BF16,
            Scalar::F64(_) => DType::F64,
            Scalar::I8(_) => DType::I8,
            Scalar::I16(_) => DType::I16,
            Scalar::I32(_) => DType::I32,
            Scalar::I64(_) => DType::I64,
            Scalar::U8(_) => DType::U8,
            Scalar::U16(_) => DType::U16,
            Scalar::U32(_) => DType::U32,
            Scalar::U64(_) => DType::U64,
            Scalar::Bool(_) => DType::Bool,
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            Scalar::F32(v) => *v as f64,
            Scalar::F16(v) => *v as f64,
            Scalar::BF16(v) => *v as f64,
            Scalar::F64(v) => *v,
            Scalar::I8(v) => *v as f64,
            Scalar::I16(v) => *v as f64,
            Scalar::I32(v) => *v as f64,
            Scalar::I64(v) => *v as f64,
            Scalar::U8(v) => *v as f64,
            Scalar::U16(v) => *v as f64,
            Scalar::U32(v) => *v as f64,
            Scalar::U64(v) => *v as f64,
            Scalar::Bool(v) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn is_exact_zero(&self) -> bool {
        match self {
            Scalar::F32(v) => *v == 0.0,
            Scalar::F64(v) => *v == 0.0,
            Scalar::I8(v) => *v == 0,
            Scalar::I16(v) => *v == 0,
            Scalar::I32(v) => *v == 0,
            Scalar::I64(v) => *v == 0,
            Scalar::U8(v) => *v == 0,
            Scalar::U16(v) => *v == 0,
            Scalar::U32(v) => *v == 0,
            Scalar::U64(v) => *v == 0,
            Scalar::Bool(v) => !v,
            Scalar::F16(v) => *v == 0,
            Scalar::BF16(v) => *v == 0,
        }
    }

    pub fn is_exact_one(&self) -> bool {
        match self {
            Scalar::F32(v) => *v == 1.0,
            Scalar::F64(v) => *v == 1.0,
            Scalar::I8(v) => *v == 1,
            Scalar::I16(v) => *v == 1,
            Scalar::I32(v) => *v == 1,
            Scalar::I64(v) => *v == 1,
            Scalar::U8(v) => *v == 1,
            Scalar::U16(v) => *v == 1,
            Scalar::U32(v) => *v == 1,
            Scalar::U64(v) => *v == 1,
            Scalar::Bool(v) => *v,
            Scalar::F16(v) => *v == 0x3C00,  // IEEE 754 half-precision 1.0
            Scalar::BF16(v) => *v == 0x3F80,  // bfloat16 1.0
        }
    }

    pub fn from_f64(val: f64, dtype: DType) -> Scalar {
        match dtype {
            DType::F32 => Scalar::F32(val as f32),
            DType::F16 => Scalar::F16(val as u16),
            DType::BF16 => Scalar::BF16(val as u16),
            DType::F64 => Scalar::F64(val),
            DType::I8 => Scalar::I8(val as i8),
            DType::I16 => Scalar::I16(val as i16),
            DType::I32 => Scalar::I32(val as i32),
            DType::I64 => Scalar::I64(val as i64),
            DType::U8 => Scalar::U8(val as u8),
            DType::U16 => Scalar::U16(val as u16),
            DType::U32 => Scalar::U32(val as u32),
            DType::U64 => Scalar::U64(val as u64),
            DType::Bool => Scalar::Bool(val != 0.0),
        }
    }
}

impl DType {
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    Contiguous,
    Strided(Vec<Dim>),
    View {
        base: BufferId,
        offset: Dim,
        strides: Vec<Dim>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorType {
    pub shape: Vec<Dim>,
    pub dtype: DType,
    pub layout: Layout,
}

impl TensorType {
    pub fn contiguous(shape: Vec<Dim>, dtype: DType) -> Self {
        Self {
            shape,
            dtype,
            layout: Layout::Contiguous,
        }
    }
}
