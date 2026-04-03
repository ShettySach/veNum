use anyhow::{bail, Result};

use crate::core::hlir::HLIRGraph;
use crate::core::llir::program::Kernel;
use crate::core::schedule::FusionTopology;

pub fn merge_fusion_topology(
    _hlir: &HLIRGraph,
    topology: &FusionTopology,
    nodes: &[crate::core::hlir::NodeId],
    kernels: &[Kernel],
) -> Result<Vec<Kernel>> {
    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    match topology {
        FusionTopology::Chain => merge_chain(kernels),
        FusionTopology::FanIn { .. } => materialize_kernels(kernels),
        FusionTopology::FanOut { .. } => materialize_kernels(kernels),
        FusionTopology::DAG { .. } => materialize_kernels(kernels),
    }
}

fn merge_chain(kernels: &[Kernel]) -> Result<Vec<Kernel>> {
    if kernels.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let mut current = kernels[0].clone();

    for next in kernels.iter().skip(1) {
        if current.loop_nest.loops == next.loop_nest.loops {
            current
                .loop_nest
                .body
                .extend(next.loop_nest.body.iter().cloned());
        } else {
            out.push(current);
            current = next.clone();
        }
    }

    out.push(current);
    Ok(out)
}

fn materialize_kernels(kernels: &[Kernel]) -> Result<Vec<Kernel>> {
    if kernels.is_empty() {
        bail!("cannot materialize empty fusion kernel list");
    }
    Ok(kernels.to_vec())
}
