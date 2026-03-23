//! Compiled program representation and execution.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{bail, Result};

use crate::core::liquid::kernel::ExecutableKernel;
use crate::core::shared::dtype::Buffer;
use crate::core::shared::exec;
use crate::core::shared::graph::{Graph, NodeId, Op};

use super::buffer_plan::StaticBufferPlan;
use super::spec::TensorSpec;

/// A single step in the compiled program's execution schedule.
#[derive(Debug)]
pub enum ExecutionStep {
    /// Execute a compiled fused kernel.
    Kernel {
        kernel_idx: usize,
        input_nodes: Vec<NodeId>,
        output_node: NodeId,
        numel: usize,
    },
    /// Execute an interpreted shape operation.
    Shape {
        op: Op,
        input_node: NodeId,
        output_node: NodeId,
    },
    /// Execute an interpreted reduce operation.
    Reduce {
        op: Op,
        input_node: NodeId,
        output_node: NodeId,
    },
}

/// A compiled program ready for execution.
///
/// Contains the entire computation compiled ahead-of-time:
/// - Input/output specifications
/// - Compiled native kernels
/// - Execution schedule
/// - Static buffer allocation plan
///
/// # Example
///
/// ```rust,ignore
/// let program = compile(&cx, &[input.id()], &[output.id()])?;
/// let input_buffer = Buffer::from_f32_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let results = program.execute(&[&input_buffer])?;
/// ```
pub struct CompiledProgram {
    /// Input tensor specifications (original, pre-optimization).
    pub input_specs: Vec<TensorSpec>,

    /// Output tensor specifications (original, pre-optimization).
    pub output_specs: Vec<TensorSpec>,

    /// Static buffer allocation plan.
    pub buffer_plan: StaticBufferPlan,

    /// The optimized computation graph.
    pub(crate) graph: Graph,

    /// Output node IDs in the optimized graph.
    pub(crate) output_nodes: Vec<NodeId>,

    /// Compiled kernels (native code).
    pub(crate) kernels: Vec<Arc<dyn ExecutableKernel>>,

    /// Execution schedule (order of operations).
    pub(crate) steps: Vec<ExecutionStep>,
}

impl CompiledProgram {
    /// Execute the compiled program with runtime inputs.
    ///
    /// # Arguments
    /// * `inputs` - Input buffers in the same order as `self.input_specs`
    ///
    /// # Returns
    /// Output buffers in the same order as `self.output_specs`
    pub fn execute(&self, inputs: &[&Buffer]) -> Result<Vec<Buffer>> {
        if inputs.len() != self.input_specs.len() {
            bail!(
                "Expected {} inputs, got {}",
                self.input_specs.len(),
                inputs.len()
            );
        }

        // Validate input shapes and dtypes
        for (i, (input, spec)) in inputs.iter().zip(&self.input_specs).enumerate() {
            if input.dtype() != spec.dtype {
                bail!(
                    "Input {} dtype mismatch: expected {:?}, got {:?}",
                    i,
                    spec.dtype,
                    input.dtype()
                );
            }
            if input.len() != spec.numel() {
                bail!(
                    "Input {} size mismatch: expected {} elements (shape {:?}), got {}",
                    i,
                    spec.numel(),
                    spec.shape,
                    input.len()
                );
            }
        }

        // Map original input NodeIds to provided buffers.
        // After optimization the graph may have different NodeIds, so we need
        // to find Load nodes without buffers in the optimized graph and map them
        // to the provided input buffers in order.
        let placeholder_ids: Vec<NodeId> = self
            .graph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| {
                if matches!(node.op, Op::Load) && node.buffer.is_none() {
                    Some(NodeId(i))
                } else {
                    None
                }
            })
            .collect();

        let mut realized: HashMap<NodeId, Buffer> = HashMap::new();

        // Bind runtime inputs to their placeholder nodes
        for (placeholder_id, input_buf) in placeholder_ids.iter().zip(inputs.iter()) {
            realized.insert(*placeholder_id, (*input_buf).clone());
        }

        // Execute each step in order
        for step in &self.steps {
            match step {
                ExecutionStep::Kernel {
                    kernel_idx,
                    input_nodes,
                    output_node,
                    numel,
                } => {
                    let kernel = &self.kernels[*kernel_idx];

                    let input_ptrs: Vec<*const u8> = input_nodes
                        .iter()
                        .map(|&node_id| -> Result<*const u8> {
                            if let Some(buf) = realized.get(&node_id) {
                                Ok(buf.as_ptr_u8())
                            } else {
                                let node = self.graph.node(node_id);
                                if let Some(ref buffer) = node.buffer {
                                    Ok(buffer.as_ptr_u8())
                                } else {
                                    bail!(
                                        "Input buffer {:?} not available for kernel execution",
                                        node_id
                                    );
                                }
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let out_node = self.graph.node(*output_node);
                    let dtype = out_node.dtype;
                    let byte_size = *numel * dtype.size_bytes();
                    let mut output_bytes = vec![0u8; byte_size];

                    kernel.execute(&input_ptrs, output_bytes.as_mut_ptr(), *numel);

                    let buffer = exec::buffer_from_bytes(output_bytes, dtype);
                    realized.insert(*output_node, buffer);
                }
                ExecutionStep::Shape {
                    op,
                    input_node,
                    output_node,
                } => {
                    let input_buf = resolve_buffer(&self.graph, *input_node, &realized)?;
                    let input_shape = self.graph.node(*input_node).shape.clone();
                    let output_shape = self.graph.node(*output_node).shape.clone();
                    let result =
                        exec::execute_shape_op_typed(op, &input_buf, &input_shape, &output_shape)?;
                    realized.insert(*output_node, result);
                }
                ExecutionStep::Reduce {
                    op,
                    input_node,
                    output_node,
                } => {
                    let input_buf = resolve_buffer(&self.graph, *input_node, &realized)?;
                    let input_shape = self.graph.node(*input_node).shape.clone();
                    let output_shape = self.graph.node(*output_node).shape.clone();
                    let result =
                        exec::execute_reduce_op_typed(op, &input_buf, &input_shape, &output_shape)?;
                    realized.insert(*output_node, result);
                }
            }
        }

        // Collect output buffers
        let mut results = Vec::with_capacity(self.output_nodes.len());
        for &output_id in &self.output_nodes {
            if let Some(buf) = realized.remove(&output_id) {
                results.push(buf);
            } else {
                // Check if it's a Const node
                let node = self.graph.node(output_id);
                if let Op::Const(val) = node.op {
                    let numel = node.numel();
                    results.push(exec::scalar_fill_buffer(val, numel));
                } else if let Some(ref buffer) = node.buffer {
                    results.push(buffer.clone());
                } else {
                    bail!("Output node {:?} was not computed", output_id);
                }
            }
        }

        Ok(results)
    }

    /// Number of compiled kernels.
    pub fn num_kernels(&self) -> usize {
        self.kernels.len()
    }

    /// Number of execution steps.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }
}

/// Resolve a buffer from realized outputs or the graph's embedded data.
fn resolve_buffer(
    graph: &Graph,
    node_id: NodeId,
    realized: &HashMap<NodeId, Buffer>,
) -> Result<Buffer> {
    if let Some(buf) = realized.get(&node_id) {
        Ok(buf.clone())
    } else {
        let node = graph.node(node_id);
        if let Some(ref buffer) = node.buffer {
            Ok(buffer.clone())
        } else {
            bail!(
                "Buffer for node {:?} not available (op: {:?})",
                node_id,
                node.op
            );
        }
    }
}
