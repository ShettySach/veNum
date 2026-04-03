use anyhow::{bail, Result};

use crate::core::compile::{compile, SearchConfig};
use crate::core::cpu::{Buffer, CpuCodeGenerator};
use crate::core::dep::NoOpDependenceAnalyzer;
use crate::core::hlir::{BufferId, NodeId};
use crate::core::schedule::TrivialHardware;
use crate::core::tensor::Context;

pub fn run_context(cx: &Context, outputs: &[NodeId], inputs: &[Buffer]) -> Result<Vec<Buffer>> {
    let input_ids = cx.input_buffers();
    if input_ids.len() != inputs.len() {
        bail!(
            "input buffer mismatch: graph expects {}, got {}",
            input_ids.len(),
            inputs.len()
        );
    }

    let graph = cx
        .graph()
        .lock()
        .expect("Graph mutex should not be poisoned")
        .clone();

    let cg = CpuCodeGenerator::new(outputs.to_vec());
    let module = compile(
        graph,
        &TrivialHardware,
        &NoOpDependenceAnalyzer,
        &cg,
        &SearchConfig::default(),
        outputs,
    )?;

    let bound: Vec<(BufferId, Buffer)> =
        input_ids.into_iter().zip(inputs.iter().cloned()).collect();
    module.execute(&bound)
}
