//! Liquid-specific tensor constructors.

use anyhow::Result;
use std::cmp::Ordering;
use std::iter::successors;

use crate::core::errors::ArangeError;
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
        let ascending = match step.partial_cmp(&0.0).ok_or(ArangeError::Comparison)? {
            Ordering::Greater if end > start => Ok(true),
            Ordering::Less if start > end => Ok(false),
            Ordering::Greater => Err(ArangeError::Positive),
            Ordering::Less => Err(ArangeError::Negative),
            Ordering::Equal => Err(ArangeError::Zero),
        }?;

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
