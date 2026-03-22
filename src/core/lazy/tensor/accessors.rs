use crate::core::lazy::dtype::DType;
use crate::core::lazy::tensor::Tensor;

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}
