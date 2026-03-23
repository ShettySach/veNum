//! Compile entry point for Solid mode.
//!
//! Takes a `SolidContext` with a built computation graph and compiles
//! it into a `CompiledProgram` ready for execution.

use anyhow::{bail, Result};
use std::sync::Arc;

use crate::core::liquid::kernel::ExecutableKernel;
use crate::core::liquid::schedule::ScheduleItem;
use crate::core::shared::graph::{NodeId, Op};

use super::pass::manager::GraphPass;

use super::backend::{CpuSolidBackend, SolidBackend};
use super::context::SolidContext;
use super::pass::fusion::FusionPass;
use super::pass::manager::PassContext;
use super::pass::memory::MemoryPlanningPass;
use super::pass::optimize::OptimizationPass;
use super::program::compiled_program::{CompiledProgram, ExecutionStep};
use super::program::spec::TensorSpec;

/// Compile a Solid context into an executable program.
///
/// # Arguments
/// * `cx` - The Solid context containing the computation graph
/// * `inputs` - Input (placeholder) node IDs
/// * `outputs` - Output (result) node IDs
///
/// # Example
///
/// ```rust,ignore
/// let cx = SolidContext::new();
/// let input = Tensor::placeholder(&cx, vec![4], DType::F32);
/// let output = input.exp();
/// let program = compile(&cx, &[input.id()], &[output.id()])?;
/// let results = program.execute(&[&input_buffer])?;
/// ```
pub fn compile(cx: &SolidContext, inputs: &[NodeId], outputs: &[NodeId]) -> Result<CompiledProgram> {
    compile_with_backend(cx, inputs, outputs, &CpuSolidBackend::new())
}

/// Compile with a specific backend.
pub fn compile_with_backend(
    cx: &SolidContext,
    inputs: &[NodeId],
    outputs: &[NodeId],
    backend: &dyn SolidBackend,
) -> Result<CompiledProgram> {
    if inputs.is_empty() {
        bail!("compile requires at least one input");
    }
    if outputs.is_empty() {
        bail!("compile requires at least one output");
    }

    let graph = cx.graph().lock().unwrap().clone();

    // Validate inputs are placeholders (Load with no buffer)
    for &id in inputs {
        let node = graph.node(id);
        if !matches!(node.op, Op::Load) || node.buffer.is_some() {
            bail!(
                "Input node {:?} is not a placeholder (must be Op::Load with no buffer)",
                id
            );
        }
    }

    // Build input/output specs before optimization (shapes are preserved)
    let input_specs: Vec<TensorSpec> = inputs
        .iter()
        .map(|&id| {
            let node = graph.node(id);
            TensorSpec::new(id, node.shape.clone(), node.dtype)
        })
        .collect();

    let output_specs: Vec<TensorSpec> = outputs
        .iter()
        .map(|&id| {
            let node = graph.node(id);
            TensorSpec::new(id, node.shape.clone(), node.dtype)
        })
        .collect();

    // --- Pass 1: Optimization ---
    let ctx = PassContext {
        graph,
        inputs: inputs.to_vec(),
        outputs: outputs.to_vec(),
    };

    let opt_pass = OptimizationPass;
    let ctx = opt_pass.run(ctx)?;

    // --- Pass 2: Fusion ---
    let mut boundary_nodes: Vec<NodeId> = Vec::new();
    boundary_nodes.extend_from_slice(&ctx.inputs);
    boundary_nodes.extend_from_slice(&ctx.outputs);

    let fusion_pass = FusionPass::new(boundary_nodes);
    let fusion_result = fusion_pass.build_schedules(&ctx);

    // --- Pass 3: Memory planning ---
    let mem_pass = MemoryPlanningPass;
    let buffer_plan = mem_pass.build_plan(&ctx);

    // --- Compile kernels from schedules ---
    let mut compiled_kernels: Vec<Arc<dyn ExecutableKernel>> = Vec::new();
    let mut execution_steps: Vec<ExecutionStep> = Vec::new();

    for schedule in &fusion_result.schedules {
        for item in schedule {
            match item {
                ScheduleItem::Fused(kernel) => {
                    let compiled = backend.compile_kernel(&ctx.graph, kernel, false)?;
                    let kernel_idx = compiled_kernels.len();
                    compiled_kernels.push(compiled);

                    execution_steps.push(ExecutionStep::Kernel {
                        kernel_idx,
                        input_nodes: kernel.input_buffers.clone(),
                        output_node: kernel.root,
                        numel: kernel.numel,
                    });
                }
                ScheduleItem::Shape(shape_item) => {
                    execution_steps.push(ExecutionStep::Shape {
                        op: shape_item.op.clone(),
                        input_node: shape_item.input,
                        output_node: shape_item.root,
                    });
                }
                ScheduleItem::Reduce(reduce_item) => {
                    execution_steps.push(ExecutionStep::Reduce {
                        op: reduce_item.op.clone(),
                        input_node: reduce_item.input,
                        output_node: reduce_item.root,
                    });
                }
            }
        }
    }

    Ok(CompiledProgram {
        input_specs,
        output_specs,
        buffer_plan,
        graph: ctx.graph,
        output_nodes: ctx.outputs,
        kernels: compiled_kernels,
        steps: execution_steps,
    })
}


