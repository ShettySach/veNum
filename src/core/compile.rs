use anyhow::Result;

use crate::core::hlir::HLIRGraph;
use crate::core::llir::LLIRProgram;
use crate::core::lower::lower;
use crate::core::schedule::{HardwareModel, ScheduleSearcher};
use crate::core::traits::{CodeGenerator, DependenceAnalyzer};

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub beam_width: usize,
    pub max_iterations: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 1,
            max_iterations: 1,
        }
    }
}

pub fn optimize_hlir(hlir: HLIRGraph) -> Result<HLIRGraph> {
    Ok(hlir)
}

pub fn optimize_llir(program: LLIRProgram, _dep: &impl DependenceAnalyzer) -> Result<LLIRProgram> {
    Ok(program)
}

pub fn compile<H, D, C>(
    hlir: HLIRGraph,
    hw: &H,
    dep: &D,
    cg: &C,
    config: &SearchConfig,
) -> Result<C::Output>
where
    H: HardwareModel + Clone,
    D: DependenceAnalyzer,
    C: CodeGenerator,
{
    let hlir = optimize_hlir(hlir)?;

    let mut searcher = ScheduleSearcher::new(hw.clone());
    searcher.beam_width = config.beam_width;
    searcher.max_iterations = config.max_iterations;

    let candidates = searcher.search(&hlir);
    let mut last_err: Option<anyhow::Error> = None;
    for (decision, _cost) in candidates {
        let llir = match lower(&hlir, &decision) {
            Ok(v) => v,
            Err(e) => {
                last_err = Some(e.context("lowering candidate failed"));
                continue;
            }
        };
        let llir = match optimize_llir(llir, dep) {
            Ok(v) => v,
            Err(e) => {
                last_err = Some(e.context("llir optimization candidate failed"));
                continue;
            }
        };
        match cg.generate(&llir) {
            Ok(v) => return Ok(v),
            Err(e) => {
                last_err = Some(e.context("codegen candidate failed"));
                continue;
            }
        }
    }

    if let Some(err) = last_err {
        Err(err.context("No valid schedule candidate survived pipeline"))
    } else {
        Err(anyhow::anyhow!("No valid schedule candidate"))
    }
}
