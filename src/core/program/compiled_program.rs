//! Compiled program representation and execution.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, bail};

use prettytable::{
    format::consts::FORMAT_BOX_CHARS,
    {Cell, Row, Table},
};

use crate::core::dtype::Buffer;
use crate::core::exec;
use crate::core::graph::{Graph, NodeId, Op};
use crate::core::kernel::ExecutableKernel;

use super::buffer_plan::StaticBufferPlan;
use super::labels::{node_label, op_label};
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

/// Output from program execution, including metadata for display.
pub struct Output {
    /// Output buffers
    pub buffers: Vec<Buffer>,
    /// Output shapes (corresponding to program.output_specs)
    pub shapes: Vec<Vec<usize>>,
}

impl Output {
    /// Print the tensor at the given output index.
    pub fn print_tensor(&self, index: usize) {
        let buffer = &self.buffers[index];
        let shape = &self.shapes[index];
        print_buffer(buffer, shape);
    }
}

fn print_buffer(buffer: &Buffer, shape: &[usize]) {
    let n = shape.len();

    if (1..=8).contains(&n) {
        let table = if n % 2 == 1 {
            let row = buffer_odd_dimensions(buffer, shape, n, 0);
            let table = Table::init(vec![row]);
            set_table_style(table)
        } else {
            buffer_even_dimensions(buffer, shape, n, 0)
        };

        println!("{}", table);
    }

    println!(
        "Tensor {{ dtype: {:?}, dims: {}, elems: {}, shape: {:?} }}",
        buffer.dtype(),
        n,
        buffer.len(),
        shape,
    )
}

fn format_buffer_element(buffer: &Buffer, index: usize) -> String {
    match buffer {
        Buffer::F32(v) => format!("{}", v[index]),
        Buffer::F64(v) => format!("{}", v[index]),
        Buffer::I32(v) => format!("{}", v[index]),
        Buffer::I64(v) => format!("{}", v[index]),
    }
}

fn buffer_odd_dimensions(buffer: &Buffer, sizes: &[usize], n: usize, flat_offset: usize) -> Row {
    let rank = sizes.len();
    let dim = rank - n;
    let size = sizes[dim];

    if n == 1 {
        Row::from((0..size).map(|i| {
            let s = format_buffer_element(buffer, flat_offset + i);
            Cell::new(&s)
        }))
    } else {
        let inner_numel: usize = sizes[dim + 1..].iter().product();
        Row::from((0..size).map(|i| {
            let offset = flat_offset + i * inner_numel;
            buffer_even_dimensions(buffer, sizes, n - 1, offset)
        }))
    }
}

fn buffer_even_dimensions(buffer: &Buffer, sizes: &[usize], n: usize, flat_offset: usize) -> Table {
    let rank = sizes.len();
    let dim = rank - n;
    let size = sizes[dim];
    let inner_numel: usize = sizes[dim + 1..].iter().product();

    let rows = (0..size)
        .map(|i| {
            let offset = flat_offset + i * inner_numel;
            buffer_odd_dimensions(buffer, sizes, n - 1, offset)
        })
        .collect();

    let table = Table::init(rows);
    set_table_style(table)
}

fn set_table_style(mut table: Table) -> Table {
    table.set_format(*FORMAT_BOX_CHARS);
    table
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
                if let Op::Load = node.op
                    && node.buffer.is_none()
                {
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

    /// Execute the program and return outputs with shape metadata.
    pub fn execute_with_metadata(&self, inputs: &[&Buffer]) -> Result<Output> {
        let buffers = self.execute(inputs)?;
        let shapes: Vec<Vec<usize>> = self
            .output_specs
            .iter()
            .map(|spec| spec.shape.clone())
            .collect();
        Ok(Output { buffers, shapes })
    }

    /// Number of compiled kernels.
    pub fn num_kernels(&self) -> usize {
        self.kernels.len()
    }

    /// Number of execution steps.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Render the compiled program's execution schedule as Mermaid flowchart code.
    ///
    /// Shows the optimized computation graph with fused kernels, shape operations,
    /// and reduce operations grouped by execution step.
    pub fn render_compiled_graph(&self) -> String {
        let mut lines = vec!["flowchart BT".to_string()];

        // Track which nodes belong to which execution step
        let mut node_to_step: HashMap<usize, usize> = HashMap::new();

        for (step_idx, step) in self.steps.iter().enumerate() {
            match step {
                ExecutionStep::Kernel {
                    output_node,
                    input_nodes,
                    ..
                } => {
                    // Mark the output and collect all nodes reachable from output
                    // without crossing input boundaries
                    node_to_step.insert(output_node.0, step_idx);
                    collect_kernel_nodes(
                        &self.graph,
                        *output_node,
                        input_nodes,
                        &mut node_to_step,
                        step_idx,
                    );
                }
                ExecutionStep::Shape { output_node, .. } => {
                    node_to_step.insert(output_node.0, step_idx);
                }
                ExecutionStep::Reduce { output_node, .. } => {
                    node_to_step.insert(output_node.0, step_idx);
                }
            }
        }

        // Render each execution step as a subgraph
        let mut kernel_idx = 0;
        let mut shape_idx = 0;
        let mut reduce_idx = 0;

        for (step_idx, step) in self.steps.iter().enumerate() {
            match step {
                ExecutionStep::Kernel { .. } => {
                    let members: Vec<usize> = node_to_step
                        .iter()
                        .filter(|&(_, &si)| si == step_idx)
                        .map(|(&nid, _)| nid)
                        .collect();

                    if members.is_empty() {
                        continue;
                    }

                    lines.push(format!(
                        "    subgraph Kernel_{} [\"Fused Kernel {}\"]",
                        kernel_idx, kernel_idx
                    ));
                    kernel_idx += 1;

                    for &nid in &members {
                        let label = node_label(&self.graph, NodeId(nid));
                        lines.push(format!("        N{}[\"{}\"]", nid, label));
                    }
                    lines.push("    end".to_string());
                }
                ExecutionStep::Shape {
                    op, output_node, ..
                } => {
                    lines.push(format!(
                        "    subgraph Shape_{} [\"Shape Op {}\"]",
                        shape_idx,
                        op_label(op)
                    ));
                    shape_idx += 1;

                    let nid = output_node.0;
                    let label = node_label(&self.graph, NodeId(nid));
                    lines.push(format!("        N{}[\"{}\"]", nid, label));
                    lines.push("    end".to_string());
                }
                ExecutionStep::Reduce {
                    op, output_node, ..
                } => {
                    lines.push(format!(
                        "    subgraph Reduce_{} [\"Reduce Op {}\"]",
                        reduce_idx,
                        op_label(op)
                    ));
                    reduce_idx += 1;

                    let nid = output_node.0;
                    let label = node_label(&self.graph, NodeId(nid));
                    lines.push(format!("        N{}[\"{}\"]", nid, label));
                    lines.push("    end".to_string());
                }
            }
        }

        // Render leaf nodes that aren't part of any execution step
        let scheduled_nodes: HashSet<usize> = node_to_step.keys().copied().collect();
        let mut visited = HashSet::new();

        for &output_node in &self.output_nodes {
            render_leaf_nodes(
                &self.graph,
                output_node,
                &scheduled_nodes,
                &mut visited,
                &mut lines,
            );
        }

        // Render edges
        let mut edge_visited = HashSet::new();
        for &output_node in &self.output_nodes {
            render_edges(&self.graph, output_node, &mut edge_visited, &mut lines);
        }

        lines.join("\n")
    }
}

/// Recursively collect all nodes that are part of a fused kernel.
fn collect_kernel_nodes(
    graph: &Graph,
    node_id: NodeId,
    input_boundaries: &[NodeId],
    node_to_step: &mut HashMap<usize, usize>,
    step_idx: usize,
) {
    let input_set: std::collections::HashSet<NodeId> = input_boundaries.iter().copied().collect();

    fn dfs(
        graph: &Graph,
        node_id: NodeId,
        input_set: &std::collections::HashSet<NodeId>,
        node_to_step: &mut HashMap<usize, usize>,
        step_idx: usize,
    ) {
        let node = graph.node(node_id);
        for &input_id in &node.inputs {
            if input_set.contains(&input_id) {
                continue;
            }
            node_to_step.insert(input_id.0, step_idx);
            dfs(graph, input_id, input_set, node_to_step, step_idx);
        }
    }

    dfs(graph, node_id, &input_set, node_to_step, step_idx);
}

/// Render nodes that aren't part of any execution step (typically inputs/constants).
fn render_leaf_nodes(
    graph: &Graph,
    node_id: NodeId,
    scheduled_nodes: &std::collections::HashSet<usize>,
    visited: &mut std::collections::HashSet<usize>,
    lines: &mut Vec<String>,
) {
    if !visited.insert(node_id.0) {
        return;
    }

    if !scheduled_nodes.contains(&node_id.0) {
        let label = node_label(graph, node_id);
        lines.push(format!("    N{}[\"{}\"]", node_id.0, label));
    }

    let node = graph.node(node_id);
    for &input_id in &node.inputs {
        render_leaf_nodes(graph, input_id, scheduled_nodes, visited, lines);
    }
}

/// Render edges between all nodes in the graph.
fn render_edges(
    graph: &Graph,
    node_id: NodeId,
    visited: &mut std::collections::HashSet<usize>,
    lines: &mut Vec<String>,
) {
    if !visited.insert(node_id.0) {
        return;
    }

    let node = graph.node(node_id);
    for &input_id in &node.inputs {
        render_edges(graph, input_id, visited, lines);
        lines.push(format!("    N{} --> N{}", input_id.0, node_id.0));
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
