use std::sync::Arc;

use super::dtype::{Buffer, DType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug)]
pub enum Op {
    // Leaf
    Const(f32),
    Load,

    // Binary elementwise
    Add,
    Sub,
    Mul,
    Div,

    // Unary elementwise
    Exp,
    Ln,
    Sqrt,
    Neg,
}

impl Op {
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Exp | Op::Ln | Op::Sqrt | Op::Neg
        )
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, Op::Add | Op::Sub | Op::Mul | Op::Div)
    }

    pub fn is_unary(&self) -> bool {
        matches!(self, Op::Exp | Op::Ln | Op::Sqrt | Op::Neg)
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub buffer: Option<Buffer>,
}

impl Node {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct Graph {
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.0]
    }

    pub fn load(&mut self, data: Arc<Vec<f32>>, shape: Vec<usize>) -> NodeId {
        let buffer = Buffer::F32(data);
        self.add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape,
            dtype: DType::F32,
            buffer: Some(buffer),
        })
    }

    pub fn constant(&mut self, value: f32, shape: Vec<usize>) -> NodeId {
        self.add_node(Node {
            op: Op::Const(value),
            inputs: vec![],
            shape,
            dtype: DType::F32,
            buffer: None,
        })
    }

    pub fn binary(&mut self, op: Op, lhs: NodeId, rhs: NodeId) -> NodeId {
        let shape = self.node(lhs).shape.clone();
        self.add_node(Node {
            op,
            inputs: vec![lhs, rhs],
            shape,
            dtype: DType::F32,
            buffer: None,
        })
    }

    pub fn unary(&mut self, op: Op, input: NodeId) -> NodeId {
        let shape = self.node(input).shape.clone();
        self.add_node(Node {
            op,
            inputs: vec![input],
            shape,
            dtype: DType::F32,
            buffer: None,
        })
    }

    /// Count how many nodes reference this node as an input.
    pub fn consumer_count(&self, id: NodeId) -> usize {
        self.nodes.iter().filter(|n| n.inputs.contains(&id)).count()
    }
}
