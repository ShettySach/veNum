use anyhow::Result;

use crate::core::llir::program::Kernel;
use crate::core::schedule::{Opt, OptOp};
use crate::core::traits::{DependenceAnalyzer, ScheduleTransform};

pub fn check_opt_legality(
    dep: &impl DependenceAnalyzer,
    kernel: &Kernel,
    opt: &Opt,
) -> Result<bool> {
    let deps = dep.analyze_kernel(kernel)?;
    let transform = schedule_transform_for_opt(kernel, opt);
    dep.check_legality(&deps, &transform)
}

fn schedule_transform_for_opt(kernel: &Kernel, opt: &Opt) -> ScheduleTransform {
    let loop_var = kernel
        .loop_nest
        .loops
        .get(opt.axis)
        .map(|lp| lp.var.clone())
        .unwrap_or_else(|| format!("axis_{}", opt.axis));

    match opt.op {
        OptOp::Tile => ScheduleTransform::Tile {
            loop_var,
            factor: opt.amt,
        },
        OptOp::Vectorize => ScheduleTransform::Vectorize {
            loop_var,
            width: usize::try_from(opt.amt.max(1)).unwrap_or(1),
        },
        OptOp::Unroll => ScheduleTransform::Unroll {
            loop_var,
            factor: usize::try_from(opt.amt.max(1)).unwrap_or(1),
        },
        OptOp::Parallelize | OptOp::GroupReduce => ScheduleTransform::Parallelize { loop_var },
        OptOp::PadTo => ScheduleTransform::Interchange {
            outer: loop_var.clone(),
            inner: loop_var,
        },
    }
}
