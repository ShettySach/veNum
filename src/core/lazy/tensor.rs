use anyhow::Result;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

use crate::core::eager::ETensor;

use super::graph::{Graph, NodeId, Op};
use super::jit::compile_kernel;
use super::optimize;
use super::render;
use super::schedule::{build_schedule, ScheduleItem};

/// A lazy tensor that builds a computation graph instead of computing eagerly.
///
/// Operations on `LTensor` record nodes in a shared `Graph`. No computation happens
/// until `.realize()` is called, which schedules, fuses, JIT-compiles, and executes.
pub struct LTensor {
    graph: Arc<std::sync::Mutex<Graph>>,
    id: NodeId,
    shape: Vec<usize>,
}

impl LTensor {
    /// Create a lazy tensor from an existing eager `Tensor<f32>`.
    pub fn from_tensor(tensor: &ETensor<f32>) -> Self {
        let data = Arc::new(tensor.data().to_vec());
        let shape = tensor.sizes().to_vec();

        let mut graph = Graph::new();
        let id = graph.load(data, shape.clone());

        LTensor {
            graph: Arc::new(std::sync::Mutex::new(graph)),
            id,
            shape,
        }
    }

    /// Create a lazy tensor from raw f32 data.
    pub fn from_slice(data: &[f32], shape: Vec<usize>) -> Self {
        let data = Arc::new(data.to_vec());

        let mut graph = Graph::new();
        let id = graph.load(data, shape.clone());

        LTensor {
            graph: Arc::new(std::sync::Mutex::new(graph)),
            id,
            shape,
        }
    }

    /// Create a constant lazy tensor (broadcasts a scalar).
    pub fn constant(value: f32, shape: Vec<usize>) -> Self {
        let mut graph = Graph::new();
        let id = graph.constant(value, shape.clone());

        LTensor {
            graph: Arc::new(std::sync::Mutex::new(graph)),
            id,
            shape,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    // --- lazy ops ---

    fn binary_op(&self, rhs: &LTensor, op: Op) -> LTensor {
        let same_graph = Arc::ptr_eq(&self.graph, &rhs.graph);
        let mut graph = self.graph.lock().unwrap();

        let rhs_id = if same_graph {
            rhs.id
        } else {
            let rhs_graph = rhs.graph.lock().unwrap();
            import_subgraph(&rhs_graph, rhs.id, &mut graph)
        };

        let id = graph.binary(op, self.id, rhs_id);
        let shape = graph.node(id).shape.clone();
        drop(graph);

        LTensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        }
    }

    fn unary_op(&self, op: Op) -> LTensor {
        let mut graph = self.graph.lock().unwrap();
        let id = graph.unary(op, self.id);
        let shape = graph.node(id).shape.clone();
        drop(graph);

        LTensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        }
    }

    pub fn exp(&self) -> LTensor {
        self.unary_op(Op::Exp)
    }

    pub fn ln(&self) -> LTensor {
        self.unary_op(Op::Ln)
    }

    pub fn sqrt(&self) -> LTensor {
        self.unary_op(Op::Sqrt)
    }

    pub fn neg(&self) -> LTensor {
        self.unary_op(Op::Neg)
    }

    // --- Visualization ---

    /// Render the raw DAG (before fusion) as Mermaid flowchart code.
    pub fn render_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        render::render_dag(&graph, self.id)
    }

    /// Render the DAG after fusion, with fused kernels grouped in subgraphs.
    pub fn render_fused_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        render::render_fused_dag(&graph, self.id)
    }

    /// Run egglog optimization and render the optimized DAG.
    pub fn render_optimized_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
        render::render_dag(&opt_graph, opt_root)
    }

    /// Run egglog optimization + fusion and render the result.
    pub fn render_optimized_fused_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
        render::render_fused_dag(&opt_graph, opt_root)
    }

    // --- Realize ---

    /// Schedule, compile, and execute the computation graph, returning an eager `Tensor<f32>`.
    pub fn realize(&self) -> Result<ETensor<f32>> {
        let graph = self.graph.lock().unwrap();

        // If the root is already a realized leaf buffer, just return it.
        let root_node = graph.node(self.id);
        if let Some(ref buffer) = root_node.buffer {
            let data = buffer.as_f32().to_vec();
            let shape = root_node.shape.clone();
            drop(graph);
            return ETensor::new(&data, &shape);
        }

        // Run egglog optimization on the graph before scheduling.
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
        drop(graph); // Release the mutex before working with opt_graph.
        let graph = &opt_graph;

        // After optimization, the root may have become a realized leaf.
        let opt_root_node = graph.node(opt_root);
        if let Some(ref buffer) = opt_root_node.buffer {
            let data = buffer.as_f32().to_vec();
            return ETensor::new(&data, &self.shape);
        }
        // The root may also be a Const (scalar broadcast).
        if let Op::Const(val) = opt_root_node.op {
            let numel = self.shape.iter().product();
            let data = vec![val; numel];
            return ETensor::new(&data, &self.shape);
        }

        let schedule = build_schedule(graph, opt_root);

        // Execute schedule items. For now we only have fused elementwise kernels.
        // Buffers realized along the way are stored here.
        let mut realized: std::collections::HashMap<NodeId, Vec<f32>> =
            std::collections::HashMap::new();

        for item in &schedule {
            match item {
                ScheduleItem::Fused(kernel) => {
                    let compiled = compile_kernel(&graph, kernel)?;

                    // Gather input pointers.
                    let input_ptrs: Vec<*const f32> = kernel
                        .input_buffers
                        .iter()
                        .map(|&buf_id| {
                            let node = graph.node(buf_id);
                            if let Some(ref buffer) = node.buffer {
                                buffer.as_f32_ptr()
                            } else if let Some(data) = realized.get(&buf_id) {
                                data.as_ptr()
                            } else {
                                panic!("Input buffer {:?} not realized and has no data", buf_id);
                            }
                        })
                        .collect();

                    // Allocate output.
                    let mut output = vec![0.0f32; kernel.numel];

                    unsafe {
                        compiled.execute(&input_ptrs, output.as_mut_ptr(), kernel.numel);
                    }

                    realized.insert(kernel.root, output);
                }
            }
        }

        // The final result is for opt_root.
        let data = realized
            .remove(&opt_root)
            .ok_or_else(|| anyhow::anyhow!("Root node was not realized"))?;

        ETensor::new(&data, &self.shape)
    }
}

/// Import a subgraph from `src_graph` into `dst_graph`, returning the new NodeId
/// corresponding to `src_id`. Load nodes are deduplicated by buffer identity
/// (`Arc::ptr_eq`), so importing the same buffer twice reuses the existing node.
fn import_subgraph(src_graph: &Graph, src_id: NodeId, dst_graph: &mut Graph) -> NodeId {
    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();

    // Build a map from buffer Arc pointer to existing NodeId in dst_graph,
    // so we can deduplicate Load nodes that reference the same underlying data.
    let mut buffer_map: std::collections::HashMap<*const Vec<f32>, NodeId> =
        std::collections::HashMap::new();
    for (i, node) in dst_graph.nodes.iter().enumerate() {
        if let (Op::Load, Some(super::dtype::Buffer::F32(ref arc))) = (&node.op, &node.buffer) {
            buffer_map.insert(Arc::as_ptr(arc), NodeId(i));
        }
    }

    import_node(src_graph, src_id, dst_graph, &mut id_map, &buffer_map)
}

fn import_node(
    src_graph: &Graph,
    src_id: NodeId,
    dst_graph: &mut Graph,
    id_map: &mut std::collections::HashMap<NodeId, NodeId>,
    buffer_map: &std::collections::HashMap<*const Vec<f32>, NodeId>,
) -> NodeId {
    if let Some(&mapped) = id_map.get(&src_id) {
        return mapped;
    }

    let node = src_graph.node(src_id);

    // Deduplicate Load nodes: if dst_graph already has a Load with the same buffer, reuse it.
    if let (Op::Load, Some(super::dtype::Buffer::F32(ref arc))) = (&node.op, &node.buffer) {
        if let Some(&existing_id) = buffer_map.get(&Arc::as_ptr(arc)) {
            id_map.insert(src_id, existing_id);
            return existing_id;
        }
    }

    // Recursively import inputs first.
    let new_inputs: Vec<NodeId> = node
        .inputs
        .iter()
        .map(|&input_id| import_node(src_graph, input_id, dst_graph, id_map, buffer_map))
        .collect();

    let new_id = dst_graph.add_node(super::graph::Node {
        op: node.op.clone(),
        inputs: new_inputs,
        shape: node.shape.clone(),
        dtype: node.dtype,
        buffer: node.buffer.clone(),
    });

    id_map.insert(src_id, new_id);
    new_id
}

// --- Operator overloading ---

impl Add for &LTensor {
    type Output = LTensor;
    fn add(self, rhs: &LTensor) -> LTensor {
        self.binary_op(rhs, Op::Add)
    }
}

impl Sub for &LTensor {
    type Output = LTensor;
    fn sub(self, rhs: &LTensor) -> LTensor {
        self.binary_op(rhs, Op::Sub)
    }
}

impl Mul for &LTensor {
    type Output = LTensor;
    fn mul(self, rhs: &LTensor) -> LTensor {
        self.binary_op(rhs, Op::Mul)
    }
}

impl Div for &LTensor {
    type Output = LTensor;
    fn div(self, rhs: &LTensor) -> LTensor {
        self.binary_op(rhs, Op::Div)
    }
}

impl Add for LTensor {
    type Output = LTensor;
    fn add(self, rhs: LTensor) -> LTensor {
        (self).binary_op(&rhs, Op::Add)
    }
}

impl Sub for LTensor {
    type Output = LTensor;
    fn sub(self, rhs: LTensor) -> LTensor {
        (self).binary_op(&rhs, Op::Sub)
    }
}

impl Mul for LTensor {
    type Output = LTensor;
    fn mul(self, rhs: LTensor) -> LTensor {
        (self).binary_op(&rhs, Op::Mul)
    }
}

impl Div for LTensor {
    type Output = LTensor;
    fn div(self, rhs: LTensor) -> LTensor {
        (self).binary_op(&rhs, Op::Div)
    }
}
