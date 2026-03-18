use super::fused_kernel::{FusedKernel, ReduceOpItem, ShapeOpItem};

/// An item in the execution schedule.
#[derive(Debug)]
pub enum ScheduleItem {
    Fused(FusedKernel),
    Shape(ShapeOpItem),
    Reduce(ReduceOpItem),
}
