use anyhow::Result;

use super::super::exec;
use super::Tensor;

impl Tensor {
    pub fn sum_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = exec::reduced_shape(&self.shape, &dimensions, keepdims)?;
        let id = self.with_graph_mut(|g| g.sum(self.id, dimensions, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn product_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = exec::reduced_shape(&self.shape, &dimensions, keepdims)?;
        let id = self.with_graph_mut(|g| g.prod(self.id, dimensions, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn max_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = exec::reduced_shape(&self.shape, &dimensions, keepdims)?;
        let id = self.with_graph_mut(|g| g.max(self.id, dimensions, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn min_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = exec::reduced_shape(&self.shape, &dimensions, keepdims)?;
        let id = self.with_graph_mut(|g| g.min(self.id, dimensions, keepdims, shape.clone()));
        Ok(self.derived(id, shape))
    }

    pub fn sum(&self) -> Result<Tensor> {
        self.sum_dims((0..self.shape.len()).collect(), true)
    }

    pub fn product(&self) -> Result<Tensor> {
        self.product_dims((0..self.shape.len()).collect(), true)
    }

    pub fn max(&self) -> Result<Tensor> {
        self.max_dims((0..self.shape.len()).collect(), true)
    }

    pub fn min(&self) -> Result<Tensor> {
        self.min_dims((0..self.shape.len()).collect(), true)
    }
}
