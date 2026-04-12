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

macro_rules! cast_to_int_scalar {
    ($value:expr, $variant:ident, $ty:ty) => {
        match $value {
            Scalar::F32(v) => Scalar::$variant(*v as $ty),
            Scalar::F16(bits) => Scalar::$variant(f16_bits_to_f32(*bits) as $ty),
            Scalar::BF16(bits) => Scalar::$variant(bf16_bits_to_f32(*bits) as $ty),
            Scalar::F64(v) => Scalar::$variant(*v as $ty),
            Scalar::I8(v) => Scalar::$variant(*v as $ty),
            Scalar::I16(v) => Scalar::$variant(*v as $ty),
            Scalar::I32(v) => Scalar::$variant(*v as $ty),
            Scalar::I64(v) => Scalar::$variant(*v as $ty),
            Scalar::U8(v) => Scalar::$variant(*v as $ty),
            Scalar::U16(v) => Scalar::$variant(*v as $ty),
            Scalar::U32(v) => Scalar::$variant(*v as $ty),
            Scalar::U64(v) => Scalar::$variant(*v as $ty),
            Scalar::Bool(v) => Scalar::$variant((*v) as $ty),
        }
    };
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
            Scalar::F16(bits) => f16_bits_to_f32(*bits) as f64,
            Scalar::BF16(bits) => bf16_bits_to_f32(*bits) as f64,
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
            Scalar::F16(v) => (*v & 0x7FFF) == 0,
            Scalar::BF16(v) => (*v & 0x7FFF) == 0,
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
            Scalar::F16(v) => *v == 0x3C00, // IEEE 754 half-precision 1.0
            Scalar::BF16(v) => *v == 0x3F80, // bfloat16 1.0
        }
    }

    pub fn from_f64(val: f64, dtype: DType) -> Scalar {
        match dtype {
            DType::F32 => Scalar::F32(val as f32),
            DType::F16 => Scalar::F16(f32_to_f16_bits(val as f32)),
            DType::BF16 => Scalar::BF16(f32_to_bf16_bits(val as f32)),
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

    pub fn cast(&self, dtype: DType) -> Scalar {
        if self.dtype() == dtype {
            return self.clone();
        }

        match dtype {
            DType::F32 => Scalar::F32(self.as_f32()),
            DType::F16 => Scalar::F16(f32_to_f16_bits(self.as_f32())),
            DType::BF16 => Scalar::BF16(f32_to_bf16_bits(self.as_f32())),
            DType::F64 => Scalar::F64(self.to_f64()),
            DType::I8 => cast_to_int_scalar!(self, I8, i8),
            DType::I16 => cast_to_int_scalar!(self, I16, i16),
            DType::I32 => cast_to_int_scalar!(self, I32, i32),
            DType::I64 => cast_to_int_scalar!(self, I64, i64),
            DType::U8 => cast_to_int_scalar!(self, U8, u8),
            DType::U16 => cast_to_int_scalar!(self, U16, u16),
            DType::U32 => cast_to_int_scalar!(self, U32, u32),
            DType::U64 => cast_to_int_scalar!(self, U64, u64),
            DType::Bool => Scalar::Bool(match self {
                Scalar::F32(v) => *v != 0.0,
                Scalar::F16(bits) => (bits & 0x7FFF) != 0,
                Scalar::BF16(bits) => (bits & 0x7FFF) != 0,
                Scalar::F64(v) => *v != 0.0,
                Scalar::I8(v) => *v != 0,
                Scalar::I16(v) => *v != 0,
                Scalar::I32(v) => *v != 0,
                Scalar::I64(v) => *v != 0,
                Scalar::U8(v) => *v != 0,
                Scalar::U16(v) => *v != 0,
                Scalar::U32(v) => *v != 0,
                Scalar::U64(v) => *v != 0,
                Scalar::Bool(v) => *v,
            }),
        }
    }

    fn as_f32(&self) -> f32 {
        match self {
            Scalar::F32(v) => *v,
            Scalar::F16(bits) => f16_bits_to_f32(*bits),
            Scalar::BF16(bits) => bf16_bits_to_f32(*bits),
            Scalar::F64(v) => *v as f32,
            Scalar::I8(v) => *v as f32,
            Scalar::I16(v) => *v as f32,
            Scalar::I32(v) => *v as f32,
            Scalar::I64(v) => *v as f32,
            Scalar::U8(v) => *v as f32,
            Scalar::U16(v) => *v as f32,
            Scalar::U32(v) => *v as f32,
            Scalar::U64(v) => *v as f32,
            Scalar::Bool(v) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x03FF) as u32;

    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut mantissa = frac;
            let mut exp_unbiased = -14;
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1;
                exp_unbiased -= 1;
            }
            let mantissa = mantissa & 0x03FF;
            let exp_bits = ((exp_unbiased + 127) as u32) << 23;
            sign | exp_bits | (mantissa << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (frac << 13)
    } else {
        let exp_bits = ((exp - 15 + 127) as u32) << 23;
        sign | exp_bits | (frac << 13)
    };

    f32::from_bits(f32_bits)
}

pub(crate) fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;

    if exp == 0xFF {
        if frac == 0 {
            return sign | 0x7C00;
        }
        let payload = ((frac >> 13) as u16) | 1;
        return sign | 0x7C00 | payload;
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1F {
        return sign | 0x7C00;
    }

    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }

        let mantissa = frac | 0x0080_0000;
        let rounded = round_to_nearest_even(mantissa, (14 - half_exp) as u32);
        if rounded == 0x0400 {
            return sign | 0x0400;
        }
        return sign | (rounded as u16);
    }

    let rounded_frac = round_to_nearest_even(frac, 13);
    if rounded_frac == 0x0400 {
        let next_exp = half_exp + 1;
        if next_exp >= 0x1F {
            return sign | 0x7C00;
        }
        return sign | ((next_exp as u16) << 10);
    }

    sign | ((half_exp as u16) << 10) | (rounded_frac as u16)
}

pub(crate) fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        let upper = (bits >> 16) as u16;
        return upper | 0x0040;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn round_to_nearest_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }

    let truncated = value >> shift;
    let remainder_mask = (1u32 << shift) - 1;
    let remainder = value & remainder_mask;
    let halfway = 1u32 << (shift - 1);

    if remainder > halfway || (remainder == halfway && (truncated & 1) == 1) {
        truncated + 1
    } else {
        truncated
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

    /// Create a TensorType with explicit strides.
    pub fn strided(shape: Vec<Dim>, dtype: DType, strides: Vec<Dim>) -> Self {
        Self {
            shape,
            dtype,
            layout: Layout::Strided(strides),
        }
    }

    /// Compute the contiguous strides for a shape (row-major order).
    /// For shape [A, B, C], strides are [B*C, C, 1].
    pub fn compute_contiguous_strides(shape: &[Dim]) -> Vec<Dim> {
        if shape.is_empty() {
            return vec![];
        }
        let mut strides = vec![Dim::Const(1); shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = &strides[i + 1] * &shape[i + 1];
        }
        strides
    }

    /// Get the effective strides for this tensor type.
    /// Returns contiguous strides if layout is Contiguous.
    pub fn strides(&self) -> Vec<Dim> {
        match &self.layout {
            Layout::Contiguous => Self::compute_contiguous_strides(&self.shape),
            Layout::Strided(s) => s.clone(),
        }
    }

    /// Check if this tensor has a contiguous memory layout.
    pub fn is_contiguous(&self) -> bool {
        match &self.layout {
            Layout::Contiguous => true,
            Layout::Strided(strides) => *strides == Self::compute_contiguous_strides(&self.shape),
        }
    }
}
