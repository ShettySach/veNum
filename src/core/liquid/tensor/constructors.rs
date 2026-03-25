//! Liquid-specific tensor constructors.

use anyhow::{bail, Result};
use std::cmp::Ordering;
use std::iter::successors;

use crate::core::liquid::context::LiquidContext;
use crate::core::shared::tensor::Tensor;

/// Liquid-specific constructor extensions.
impl Tensor<LiquidContext> {
    /// Create a 1D tensor with evenly spaced values.
    ///
    /// # Arguments
    /// * `cx` - The liquid context
    /// * `start` - Start value (inclusive)
    /// * `end` - End value (exclusive)
    /// * `step` - Step size between values
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cx = LiquidContext::new();
    /// let t = LiquidTensor::arange(&cx, 0.0, 5.0, 1.0)?;  // [0, 1, 2, 3, 4]
    /// ```
    pub fn arange(cx: &LiquidContext, start: f32, end: f32, step: f32) -> Result<Self> {
        let ascending = match step
            .partial_cmp(&0.0)
            .ok_or_else(|| anyhow::anyhow!("step cannot be compared with zero"))?
        {
            Ordering::Greater if end > start => true,
            Ordering::Less if start > end => false,
            Ordering::Greater => bail!("step is positive, but start > end"),
            Ordering::Less => bail!("step is negative, but end > start"),
            Ordering::Equal => bail!("step size cannot be zero"),
        };

        let data: Vec<_> = successors(Some(start), |&prev| {
            let curr = prev + step;
            let cond = end > curr;
            (ascending == cond).then_some(curr)
        })
        .collect();

        let shape = vec![data.len()];
        Ok(Self::from_slice(cx, &data, shape))
    }
}
