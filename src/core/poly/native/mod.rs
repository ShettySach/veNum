pub mod distance;
pub mod feasibility;
pub mod fm;

use anyhow::Result;

use super::analysis::dependence::analyze_kernel_poly;
use super::analysis::legality;

use crate::core::llir::{Dependence, DependenceRelation, Kernel, Loop, MemoryAccess};
use crate::core::traits::{DependenceAnalyzer, ScheduleTransform};

#[derive(Default)]
pub struct NativeDependenceAnalyzer;

impl DependenceAnalyzer for NativeDependenceAnalyzer {
    fn analyze_kernel(&self, kernel: &Kernel) -> Result<Vec<Dependence>> {
        Ok(analyze_kernel_poly(kernel))
    }

    fn check_legality(
        &self,
        deps: &[Dependence],
        transform: &ScheduleTransform,
        kernel: &Kernel,
    ) -> Result<bool> {
        let legal = match transform {
            ScheduleTransform::Parallelize { loop_var } => {
                legality::can_parallelize(deps, loop_var)
            }
            ScheduleTransform::Interchange { outer, inner } => {
                legality::can_interchange(deps, outer, inner)
            }
            ScheduleTransform::Vectorize { loop_var, .. } => {
                legality::can_vectorize(deps, loop_var, kernel)
            }
            ScheduleTransform::Tile { loop_var, .. } => legality::can_tile(deps, loop_var),
            ScheduleTransform::Unroll { loop_var, .. } => legality::can_unroll(deps, loop_var),
            ScheduleTransform::GroupReduce { loop_var } => {
                legality::can_group_reduce(deps, loop_var, kernel)
            }
            ScheduleTransform::PadTo { loop_var } => legality::can_pad_to(deps, loop_var),
            _ => true,
        };
        Ok(legal)
    }

    fn access_dependence(
        &self,
        write: &MemoryAccess,
        read: &MemoryAccess,
        loops: &[Loop],
    ) -> Result<Option<DependenceRelation>> {
        if write.buffer != read.buffer {
            return Ok(None);
        }
        if write.indices.len() != read.indices.len() {
            return Ok(None);
        }

        let source_vars: Vec<String> = loops.iter().map(|l| l.var.clone()).collect();
        let sink_vars = source_vars.clone();

        Ok(Some(DependenceRelation {
            source_vars,
            sink_vars,
            constraints: Vec::new(),
        }))
    }
}
