use anyhow::Result;

use crate::core::hlir::BufferId;
use crate::core::llir::{Dependence, DependenceRelation, Kernel, LLIRProgram, Loop, MemoryAccess};

pub trait DependenceAnalyzer {
    fn analyze_kernel(&self, kernel: &Kernel) -> Result<Vec<Dependence>>;

    fn check_legality(&self, deps: &[Dependence], transform: &ScheduleTransform) -> Result<bool>;

    fn access_dependence(
        &self,
        write: &MemoryAccess,
        read: &MemoryAccess,
        loops: &[Loop],
    ) -> Result<Option<DependenceRelation>>;
}

pub enum ScheduleTransform {
    Tile {
        loop_var: String,
        factor: i64,
    },
    Interchange {
        outer: String,
        inner: String,
    },
    Parallelize {
        loop_var: String,
    },
    Unroll {
        loop_var: String,
        factor: usize,
    },
    Vectorize {
        loop_var: String,
        width: usize,
    },
    ComputeAt {
        producer: usize,
        consumer: usize,
        loop_var: String,
    },
    CacheRead {
        buffer: BufferId,
        at_loop: String,
    },
    CacheWrite {
        buffer: BufferId,
        at_loop: String,
    },
}

pub trait CodeGenerator {
    type Output;
    fn generate(&self, program: &LLIRProgram) -> Result<Self::Output>;
}
