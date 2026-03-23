use crate::core::lazy::schedule::fused_kernel::{FusedKernel, ReduceOpItem, ShapeOpItem};

/// An item in the execution schedule.
#[derive(Debug)]
pub enum ScheduleItem {
    Fused(Box<FusedKernel>),
    Shape(ShapeOpItem),
    Reduce(ReduceOpItem),
}
