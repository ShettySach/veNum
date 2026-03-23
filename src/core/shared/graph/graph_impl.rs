use crate::core::shared::dtype::{Buffer, Scalar};

use super::{Node, NodeId, Op};

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

    pub fn load(&mut self, buffer: Buffer, shape: Vec<usize>) -> NodeId {
        let dtype = buffer.dtype();
        self.add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape,
            dtype,
            buffer: Some(buffer),
        })
    }

    pub fn constant(&mut self, value: Scalar, shape: Vec<usize>) -> NodeId {
        let dtype = value.dtype();
        self.add_node(Node {
            op: Op::Const(value),
            inputs: vec![],
            shape,
            dtype,
            buffer: None,
        })
    }

    pub fn binary(&mut self, op: Op, lhs: NodeId, rhs: NodeId) -> NodeId {
        let shape = self.node(lhs).shape.clone();
        let dtype = self.node(lhs).dtype;
        self.add_node(Node {
            op,
            inputs: vec![lhs, rhs],
            shape,
            dtype,
            buffer: None,
        })
    }

    pub fn unary(&mut self, op: Op, input: NodeId) -> NodeId {
        let shape = self.node(input).shape.clone();
        let dtype = self.node(input).dtype;
        self.add_node(Node {
            op,
            inputs: vec![input],
            shape,
            dtype,
            buffer: None,
        })
    }

    // --- Shape node builders ---

    pub fn reshape(&mut self, input: NodeId, shape: Vec<usize>) -> NodeId {
        self.add_node(Node {
            op: Op::Reshape,
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

    pub fn expand(&mut self, input: NodeId, shape: Vec<usize>) -> NodeId {
        self.add_node(Node {
            op: Op::Expand,
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
        constant: Scalar,
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
