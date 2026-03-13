use std::sync::Arc;

use super::dtype::{Buffer, DType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug)]
#[allow(dead_code)]
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

    // Shape ops (lazy graph nodes)
    Reshape(Vec<usize>),
    Permute(Vec<usize>),
    Transpose(usize, usize),
    Expand(Vec<usize>),
    Slice(Vec<(usize, usize)>),
    Flip(Vec<usize>),
    Squeeze,
    Unsqueeze(usize),
    Pad(f32, Vec<(usize, usize)>),

    // Reduce ops
    Sum(Vec<usize>, bool),
    Prod(Vec<usize>, bool),
    Max(Vec<usize>, bool),
    Min(Vec<usize>, bool),
}

impl Op {
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Exp | Op::Ln | Op::Sqrt | Op::Neg
        )
    }

    pub fn is_shape_op(&self) -> bool {
        matches!(
            self,
            Op::Reshape(_)
                | Op::Permute(_)
                | Op::Transpose(_, _)
                | Op::Expand(_)
                | Op::Slice(_)
                | Op::Flip(_)
                | Op::Squeeze
                | Op::Unsqueeze(_)
                | Op::Pad(_, _)
        )
    }

    pub fn is_reduce_op(&self) -> bool {
        matches!(
            self,
            Op::Sum(_, _) | Op::Prod(_, _) | Op::Max(_, _) | Op::Min(_, _)
        )
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

    // --- Shape node builders ---

    pub fn reshape(&mut self, input: NodeId, shape: Vec<usize>) -> NodeId {
        self.add_node(Node {
            op: Op::Reshape(shape.clone()),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn permute(&mut self, input: NodeId, permutation: Vec<usize>, shape: Vec<usize>) -> NodeId {
        self.add_node(Node {
            op: Op::Permute(permutation),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn transpose(
        &mut self,
        input: NodeId,
        dim_1: usize,
        dim_2: usize,
        shape: Vec<usize>,
    ) -> NodeId {
        self.add_node(Node {
            op: Op::Transpose(dim_1, dim_2),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn expand(&mut self, input: NodeId, expansions: Vec<usize>, shape: Vec<usize>) -> NodeId {
        self.add_node(Node {
            op: Op::Expand(expansions),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn slice(
        &mut self,
        input: NodeId,
        ranges: Vec<(usize, usize)>,
        shape: Vec<usize>,
    ) -> NodeId {
        self.add_node(Node {
            op: Op::Slice(ranges),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn flip(&mut self, input: NodeId, flips: Vec<usize>, shape: Vec<usize>) -> NodeId {
        self.add_node(Node {
            op: Op::Flip(flips),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn squeeze(&mut self, input: NodeId, shape: Vec<usize>) -> NodeId {
        self.add_node(Node {
            op: Op::Squeeze,
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn unsqueeze(&mut self, input: NodeId, unsqueezed: usize, shape: Vec<usize>) -> NodeId {
        self.add_node(Node {
            op: Op::Unsqueeze(unsqueezed),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn pad(
        &mut self,
        input: NodeId,
        constant: f32,
        padding: Vec<(usize, usize)>,
        shape: Vec<usize>,
    ) -> NodeId {
        self.add_node(Node {
            op: Op::Pad(constant, padding),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    // --- Reduce node builders ---

    pub fn sum(
        &mut self,
        input: NodeId,
        dimensions: Vec<usize>,
        keepdims: bool,
        shape: Vec<usize>,
    ) -> NodeId {
        self.add_node(Node {
            op: Op::Sum(dimensions, keepdims),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn prod(
        &mut self,
        input: NodeId,
        dimensions: Vec<usize>,
        keepdims: bool,
        shape: Vec<usize>,
    ) -> NodeId {
        self.add_node(Node {
            op: Op::Prod(dimensions, keepdims),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn max(
        &mut self,
        input: NodeId,
        dimensions: Vec<usize>,
        keepdims: bool,
        shape: Vec<usize>,
    ) -> NodeId {
        self.add_node(Node {
            op: Op::Max(dimensions, keepdims),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }

    pub fn min(
        &mut self,
        input: NodeId,
        dimensions: Vec<usize>,
        keepdims: bool,
        shape: Vec<usize>,
    ) -> NodeId {
        self.add_node(Node {
            op: Op::Min(dimensions, keepdims),
            inputs: vec![input],
            shape,
            dtype: self.node(input).dtype,
            buffer: None,
        })
    }
}
