use anyhow::Result;

use crate::core::llir::loop_nest::LoopKind;
use crate::core::llir::program::Kernel;
use crate::core::schedule::{Opt, OptOp};
use crate::core::traits::{DependenceAnalyzer, ScheduleTransform};

/// Check whether an opt is legal on the current kernel state.
///
/// This runs both structural checks (e.g. axis bounds, loop kind
/// compatibility) and polyhedral dependence-based legality.
pub fn check_opt_legality(
    dep: &impl DependenceAnalyzer,
    kernel: &Kernel,
    opt: &Opt,
) -> Result<bool> {
    // 1. Structural pre-checks that don't need dependence analysis.
    if !structural_precondition(kernel, opt) {
        return Ok(false);
    }

    // 2. Polyhedral legality via dependence analysis.
    let deps = dep.analyze_kernel(kernel)?;
    let transform = schedule_transform_for_opt(kernel, opt);
    dep.check_legality(&deps, &transform, kernel)
}

/// Fast structural checks before dependence analysis.
fn structural_precondition(kernel: &Kernel, opt: &Opt) -> bool {
    let loops = &kernel.loop_nest.loops;

    // Axis must be in bounds.
    if opt.axis >= loops.len() {
        return false;
    }

    let target_loop = &loops[opt.axis];

    match opt.op {
        OptOp::GroupReduce => {
            // GroupReduce is only meaningful on Reduce-kinded loops.
            matches!(target_loop.kind, LoopKind::Reduce { .. })
        }
        OptOp::Vectorize => {
            // Cannot vectorize a reduction loop.
            !matches!(target_loop.kind, LoopKind::Reduce { .. })
        }
        OptOp::Tile | OptOp::PadTo => {
            // Tile and PadTo require constant upper bounds.
            target_loop.upper.as_const_value().is_some()
        }
        OptOp::Parallelize => {
            // Parallelize (which internally tiles) requires constant bounds.
            target_loop.upper.as_const_value().is_some()
        }
        OptOp::Unroll => true,
    }
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
        OptOp::Parallelize => ScheduleTransform::Parallelize { loop_var },
        OptOp::GroupReduce => ScheduleTransform::GroupReduce { loop_var },
        OptOp::PadTo => ScheduleTransform::PadTo { loop_var },
    }
}
