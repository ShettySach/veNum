use crate::core::schedule::{BackendClass, KernelContext, Opt, OptOp};

pub fn opt_candidates(ctx: &KernelContext) -> Vec<Opt> {
    let mut out = Vec::new();

    for (axis, &bound) in ctx.loop_bounds.iter().enumerate() {
        let is_reduce = ctx.reduce_axes.contains(&axis);

        for amt in tile_amounts(bound) {
            out.push(Opt {
                op: OptOp::Tile,
                axis,
                amt,
            });

            out.push(Opt {
                op: OptOp::PadTo,
                axis,
                amt,
            });

            if !is_reduce {
                out.push(Opt {
                    op: OptOp::Parallelize,
                    axis,
                    amt,
                });
            }
        }

        if !is_reduce {
            for width in vector_widths(ctx) {
                out.push(Opt {
                    op: OptOp::Vectorize,
                    axis,
                    amt: width,
                });
            }
        }

        for factor in unroll_amounts(bound) {
            out.push(Opt {
                op: OptOp::Unroll,
                axis,
                amt: factor,
            });
        }

        if is_reduce && matches!(ctx.backend, BackendClass::Gpu | BackendClass::Wgsl) {
            for threads in group_reduce_amounts(bound, ctx.shared_budget) {
                out.push(Opt {
                    op: OptOp::GroupReduce,
                    axis,
                    amt: threads,
                });
            }
        }
    }

    out
}

fn tile_amounts(bound: i64) -> Vec<i64> {
    let mut cands = vec![2, 4, 8, 16, 32, 64];
    cands.retain(|v| *v > 1 && *v <= bound.max(2));
    cands
}

fn unroll_amounts(bound: i64) -> Vec<i64> {
    let mut cands = vec![2, 4, 8];
    cands.retain(|v| *v > 1 && *v <= bound.max(2));
    cands
}

fn vector_widths(ctx: &KernelContext) -> Vec<i64> {
    match ctx.backend {
        BackendClass::Cpu => vec![4, 8, 16],
        BackendClass::Gpu | BackendClass::Wgsl => vec![2, 4],
    }
}

fn group_reduce_amounts(bound: i64, shared_budget: usize) -> Vec<i64> {
    let mut cands = vec![2, 4, 8, 16, 32, 64];
    let budget_elems = i64::try_from(shared_budget / 4).unwrap_or(i64::MAX);
    cands.retain(|v| *v <= bound.max(2) && *v <= budget_elems.max(1));
    cands
}
