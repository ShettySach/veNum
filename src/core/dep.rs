use anyhow::Result;

use crate::core::llir::{Dependence, DependenceRelation, Kernel, Loop, MemoryAccess};
use crate::core::traits::{DependenceAnalyzer, ScheduleTransform};

#[derive(Default)]
pub struct NoOpDependenceAnalyzer;

impl DependenceAnalyzer for NoOpDependenceAnalyzer {
    fn analyze_kernel(&self, _kernel: &Kernel) -> Result<Vec<Dependence>> {
        Ok(Vec::new())
    }

    fn check_legality(&self, _deps: &[Dependence], _transform: &ScheduleTransform, _kernel: &Kernel) -> Result<bool> {
        Ok(true)
    }

    fn access_dependence(
        &self,
        _write: &MemoryAccess,
        _read: &MemoryAccess,
        _loops: &[Loop],
    ) -> Result<Option<DependenceRelation>> {
        Ok(None)
    }
}
