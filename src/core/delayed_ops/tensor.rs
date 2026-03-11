use anyhow::Result;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

use crate::Tensor;

use super::graph::{Graph, NodeId, Op};
use super::jit::compile_kernel;
use super::schedule::{build_schedule, ScheduleItem};

/// A delayed (lazy) tensor that builds a computation graph instead of computing eagerly.
///
/// Operations on `DTensor` record nodes in a shared `Graph`. No computation happens
/// until `.realize()` is called, which schedules, fuses, JIT-compiles, and executes.
pub struct DTensor {
    graph: Arc<std::sync::Mutex<Graph>>,
    id: NodeId,
    shape: Vec<usize>,
}

impl DTensor {
    /// Create a delayed tensor from an existing eager `Tensor<f32>`.
    pub fn from_tensor(tensor: &Tensor<f32>) -> Self {
        let data = Arc::new(tensor.data().to_vec());
        let shape = tensor.sizes().to_vec();

        let mut graph = Graph::new();
        let id = graph.load(data, shape.clone());

        DTensor {
            graph: Arc::new(std::sync::Mutex::new(graph)),
            id,
            shape,
        }
    }

    /// Create a delayed tensor from raw f32 data.
    pub fn from_slice(data: &[f32], shape: Vec<usize>) -> Self {
        let data = Arc::new(data.to_vec());

        let mut graph = Graph::new();
        let id = graph.load(data, shape.clone());

        DTensor {
            graph: Arc::new(std::sync::Mutex::new(graph)),
            id,
            shape,
        }
    }

    /// Create a constant delayed tensor (broadcasts a scalar).
    pub fn constant(value: f32, shape: Vec<usize>) -> Self {
        let mut graph = Graph::new();
        let id = graph.constant(value, shape.clone());

        DTensor {
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

    // --- Delayed ops ---

    fn binary_op(&self, rhs: &DTensor, op: Op) -> DTensor {
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

        DTensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        }
    }

    fn unary_op(&self, op: Op) -> DTensor {
        let mut graph = self.graph.lock().unwrap();
        let id = graph.unary(op, self.id);
        let shape = graph.node(id).shape.clone();
        drop(graph);

        DTensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        }
    }

    pub fn exp(&self) -> DTensor {
        self.unary_op(Op::Exp)
    }

    pub fn ln(&self) -> DTensor {
        self.unary_op(Op::Ln)
    }

    pub fn sqrt(&self) -> DTensor {
        self.unary_op(Op::Sqrt)
    }

    pub fn neg(&self) -> DTensor {
        self.unary_op(Op::Neg)
    }

    // --- Realize ---

    /// Schedule, compile, and execute the computation graph, returning an eager `Tensor<f32>`.
    pub fn realize(&self) -> Result<Tensor<f32>> {
        let graph = self.graph.lock().unwrap();

        // If the root is already a realized leaf buffer, just return it.
        let root_node = graph.node(self.id);
        if let Some(ref buffer) = root_node.buffer {
            let data = buffer.as_f32().to_vec();
            let shape = root_node.shape.clone();
            drop(graph);
            return Tensor::new(&data, &shape);
        }

        let schedule = build_schedule(&graph, self.id);

        // Execute schedule items. For now we only have fused elementwise kernels.
        // Buffers realized along the way are stored here.
        let mut realized: std::collections::HashMap<NodeId, Vec<f32>> = std::collections::HashMap::new();

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
                            } else if let Some(ref data) = realized.get(&buf_id) {
                                data.as_ptr()
                            } else {
                                panic!(
                                    "Input buffer {:?} not realized and has no data",
                                    buf_id
                                );
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

        // The final result is for self.id.
        let data = realized
            .remove(&self.id)
            .ok_or_else(|| anyhow::anyhow!("Root node was not realized"))?;

        drop(graph);
        Tensor::new(&data, &self.shape)
    }
}

/// Import a subgraph from `src_graph` into `dst_graph`, returning the new NodeId
/// corresponding to `src_id`.
fn import_subgraph(
    src_graph: &Graph,
    src_id: NodeId,
    dst_graph: &mut Graph,
) -> NodeId {
    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    import_node(src_graph, src_id, dst_graph, &mut id_map)
}

fn import_node(
    src_graph: &Graph,
    src_id: NodeId,
    dst_graph: &mut Graph,
    id_map: &mut std::collections::HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&mapped) = id_map.get(&src_id) {
        return mapped;
    }

    let node = src_graph.node(src_id);

    // Recursively import inputs first.
    let new_inputs: Vec<NodeId> = node
        .inputs
        .iter()
        .map(|&input_id| import_node(src_graph, input_id, dst_graph, id_map))
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

impl Add for &DTensor {
    type Output = DTensor;
    fn add(self, rhs: &DTensor) -> DTensor {
        self.binary_op(rhs, Op::Add)
    }
}

impl Sub for &DTensor {
    type Output = DTensor;
    fn sub(self, rhs: &DTensor) -> DTensor {
        self.binary_op(rhs, Op::Sub)
    }
}

impl Mul for &DTensor {
    type Output = DTensor;
    fn mul(self, rhs: &DTensor) -> DTensor {
        self.binary_op(rhs, Op::Mul)
    }
}

impl Div for &DTensor {
    type Output = DTensor;
    fn div(self, rhs: &DTensor) -> DTensor {
        self.binary_op(rhs, Op::Div)
    }
}

impl Add for DTensor {
    type Output = DTensor;
    fn add(self, rhs: DTensor) -> DTensor {
        (&self).binary_op(&rhs, Op::Add)
    }
}

impl Sub for DTensor {
    type Output = DTensor;
    fn sub(self, rhs: DTensor) -> DTensor {
        (&self).binary_op(&rhs, Op::Sub)
    }
}

impl Mul for DTensor {
    type Output = DTensor;
    fn mul(self, rhs: DTensor) -> DTensor {
        (&self).binary_op(&rhs, Op::Mul)
    }
}

impl Div for DTensor {
    type Output = DTensor;
    fn div(self, rhs: DTensor) -> DTensor {
        (&self).binary_op(&rhs, Op::Div)
    }
}
