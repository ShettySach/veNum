use anyhow::Result;

use crate::core::hlir::{HLIRGraph, NodeId, canonicalize_with_roots_and_map};
use crate::core::llir::loop_nest::LoopKind;
use crate::core::llir::program::Kernel;
use crate::core::llir::LLIRProgram;
use crate::core::lower::lower;
use crate::core::schedule::{HardwareModel, ScheduleSearcher};
use crate::core::traits::{CodeGenerator, DependenceAnalyzer};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub beam_width: usize,
    pub max_iterations: usize,
    pub enable_canonicalization: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 1,
            max_iterations: 1,
            enable_canonicalization: true,
        }
    }
}

/// Apply HLIR-level canonicalization optimizations
pub fn optimize_hlir_with_outputs(
    hlir: HLIRGraph,
    outputs: &[NodeId],
) -> Result<(HLIRGraph, HashMap<NodeId, NodeId>)> {
    let (opt, remap) = canonicalize_with_roots_and_map(&hlir, outputs);
    Ok((opt, remap))
}

/// Apply LLIR-level polyhedral optimizations.
///
/// Currently implements:
/// - Loop interchange when proven legal by dependence analysis and the
///   inner loop has a larger trip count (better spatial locality).
///
/// Transformations are conservative: each is checked for legality before
/// being applied, and at most one interchange is attempted per kernel.
pub fn optimize_llir(program: LLIRProgram, dep: &impl DependenceAnalyzer) -> Result<LLIRProgram> {
    let mut kernels = program.kernels;

    for kernel in &mut kernels {
        try_interchange(kernel, dep)?;
    }

    Ok(LLIRProgram { kernels })
}

/// Try to find and apply a beneficial loop interchange on a kernel.
///
/// Scans adjacent sequential loop pairs and interchanges the first pair
/// where:
/// 1. Both loops are sequential (not parallel, vectorized, or reduce).
/// 2. The inner loop has a strictly larger constant trip count (bringing
///    the larger dimension inward improves spatial locality).
/// 3. Dependence analysis confirms the interchange is legal.
fn try_interchange(kernel: &mut Kernel, dep: &impl DependenceAnalyzer) -> Result<()> {
    let loops = &kernel.loop_nest.loops;
    if loops.len() < 2 {
        return Ok(());
    }

    // Find the first beneficial interchange candidate.
    let candidate = (0..loops.len() - 1).find(|&i| {
        let outer = &loops[i];
        let inner = &loops[i + 1];

        // Both must be sequential.
        if !matches!(outer.kind, LoopKind::Sequential)
            || !matches!(inner.kind, LoopKind::Sequential)
        {
            return false;
        }

        // Inner must have a strictly larger trip count for locality benefit.
        match (outer.upper.as_const_value(), inner.upper.as_const_value()) {
            (Some(ou), Some(iu)) => iu > ou,
            _ => false,
        }
    });

    let candidate_idx = match candidate {
        Some(i) => i,
        None => return Ok(()),
    };

    // Check legality via dependence analysis.
    let deps = dep.analyze_kernel(kernel)?;
    let outer_var = &kernel.loop_nest.loops[candidate_idx].var;
    let inner_var = &kernel.loop_nest.loops[candidate_idx + 1].var;

    use crate::core::poly::analysis::legality::can_interchange;
    if !can_interchange(&deps, outer_var, inner_var) {
        return Ok(());
    }

    // Apply the interchange.
    kernel.loop_nest.loops.swap(candidate_idx, candidate_idx + 1);

    Ok(())
}

/// Compile HLIR graph to executable module
///
/// This function:
/// 1. Optionally applies HLIR canonicalization (reshape elimination, etc.)
/// 2. Remaps output node IDs to account for canonicalization changes
/// 3. Searches for optimal schedule using the provided hardware model
/// 4. Lowers each schedule candidate to LLIR
/// 5. Applies LLIR optimizations
/// 6. Generates code using the provided code generator
pub fn compile<H, D, C>(
    hlir: HLIRGraph,
    hw: &H,
    dep: &D,
    cg: &C,
    config: &SearchConfig,
    outputs: &[NodeId],
) -> Result<C::Output>
where
    H: HardwareModel + Clone,
    D: DependenceAnalyzer,
    C: CodeGenerator + OutputRemapper,
{
    // Apply HLIR canonicalization if enabled
    let (hlir, output_remap) = if config.enable_canonicalization {
        optimize_hlir_with_outputs(hlir, outputs)?
    } else {
        (hlir, HashMap::new())
    };

    // Remap outputs to account for canonicalization changes
    let remapped_outputs: Vec<NodeId> = outputs
        .iter()
        .map(|id| output_remap.get(id).copied().unwrap_or(*id))
        .collect();

    // Create code generator with remapped outputs
    let cg = cg.with_remapped_outputs(remapped_outputs);

    // Search for optimal schedule
    let mut searcher = ScheduleSearcher::new(hw.clone());
    searcher.beam_width = config.beam_width;
    searcher.max_iterations = config.max_iterations;

    let candidates = searcher.search(&hlir);
    let mut last_err: Option<anyhow::Error> = None;

    // Try each schedule candidate
    for (decision, _cost) in candidates {
        // Lower to LLIR
        let llir = match lower(&hlir, &decision, dep) {
            Ok(v) => v,
            Err(e) => {
                last_err = Some(e.context("lowering candidate failed"));
                continue;
            }
        };

        // Optimize LLIR
        let llir = match optimize_llir(llir, dep) {
            Ok(v) => v,
            Err(e) => {
                last_err = Some(e.context("llir optimization candidate failed"));
                continue;
            }
        };

        // Generate code
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

/// Trait for code generators that support output remapping
///
/// This is needed to handle HLIR canonicalization, which may change node IDs.
/// Code generators should create a new instance with remapped outputs.
pub trait OutputRemapper: CodeGenerator {
    fn with_remapped_outputs(&self, outputs: Vec<NodeId>) -> Self;
}
