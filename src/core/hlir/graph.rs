use super::dim::Dim;
use super::op::{CmpOp, Op, Range, ReduceOp};
use super::types::{BufferId, DType, NodeId, Scalar, TensorType};

#[derive(Clone, Debug, PartialEq)]
pub struct HLIRNode {
    pub op: Op,
    pub ty: TensorType,
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct HLIRGraph {
    nodes: Vec<HLIRNode>,
}

impl HLIRGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, id: NodeId) -> &HLIRNode {
        &self.nodes[id.0]
    }

    pub fn ty(&self, id: NodeId) -> &TensorType {
        &self.node(id).ty
    }

    pub fn add_node(&mut self, op: Op, ty: TensorType) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(HLIRNode { op, ty });
        id
    }

    pub fn topo_iter(&self) -> impl Iterator<Item = (NodeId, &HLIRNode)> {
        self.nodes.iter().enumerate().map(|(i, n)| (NodeId(i), n))
    }

    pub fn constant(&mut self, value: Scalar, shape: Vec<Dim>, dtype: DType) -> NodeId {
        let op = Op::Const {
            value,
            shape: shape.clone(),
            dtype,
        };
        self.add_node(op, TensorType::contiguous(shape, dtype))
    }

    pub fn load(&mut self, buffer: BufferId, ty: TensorType) -> NodeId {
        self.add_node(Op::Load { buffer }, ty)
    }

    pub fn store(&mut self, buffer: BufferId, value: NodeId) -> NodeId {
        let ty = self.ty(value).clone();
        self.add_node(Op::Store { buffer, value }, ty)
    }

    pub fn unary(&mut self, input: NodeId, op: fn(NodeId) -> Op) -> NodeId {
        let ty = self.ty(input).clone();
        self.add_node(op(input), ty)
    }

    pub fn cast(&mut self, input: NodeId, to: DType) -> NodeId {
        let mut ty = self.ty(input).clone();
        ty.dtype = to;
        self.add_node(Op::Cast { input, to }, ty)
    }

    pub fn binary(&mut self, lhs: NodeId, rhs: NodeId, op: fn(NodeId, NodeId) -> Op) -> NodeId {
        let ty = self.ty(lhs).clone();
        self.add_node(op(lhs, rhs), ty)
    }

    pub fn cmp(&mut self, op: CmpOp, lhs: NodeId, rhs: NodeId) -> NodeId {
        let mut ty = self.ty(lhs).clone();
        ty.dtype = DType::Bool;
        self.add_node(Op::Cmp { op, lhs, rhs }, ty)
    }

    pub fn where_select(&mut self, cond: NodeId, then_val: NodeId, else_val: NodeId) -> NodeId {
        let ty = self.ty(then_val).clone();
        self.add_node(
            Op::Where {
                cond,
                then_val,
                else_val,
            },
            ty,
        )
    }

    pub fn reduce(
        &mut self,
        input: NodeId,
        axes: Vec<usize>,
        op: ReduceOp,
        keepdim: bool,
    ) -> NodeId {
        let in_ty = self.ty(input).clone();
        let out_shape = reduce_shape(&in_ty.shape, &axes, keepdim);
        let out_ty = TensorType::contiguous(out_shape, in_ty.dtype);
        self.add_node(
            Op::Reduce {
                input,
                axes,
                op,
                keepdim,
            },
            out_ty,
        )
    }

    pub fn reshape(&mut self, input: NodeId, shape: Vec<Dim>) -> NodeId {
        let mut out_ty = self.ty(input).clone();
        out_ty.shape = shape.clone();
        self.add_node(Op::Reshape { input, shape }, out_ty)
    }

    pub fn permute(&mut self, input: NodeId, axes: Vec<usize>) -> NodeId {
        let in_ty = self.ty(input).clone();
        let shape = axes.iter().map(|&i| in_ty.shape[i].clone()).collect();
        let out_ty = TensorType {
            shape,
            dtype: in_ty.dtype,
            layout: in_ty.layout.clone(),
        };
        self.add_node(Op::Permute { input, axes }, out_ty)
    }

    pub fn slice(&mut self, input: NodeId, ranges: Vec<Range>) -> NodeId {
        let in_ty = self.ty(input).clone();
        let shape = ranges
            .iter()
            .map(|r| match (&r.start, &r.end) {
                (Dim::Const(s), Dim::Const(e)) => Dim::constant(e - s),
                _ => Dim::add(r.end.clone(), Dim::mul(Dim::constant(-1), r.start.clone())),
            })
            .collect();
        let out_ty = TensorType {
            shape,
            dtype: in_ty.dtype,
            layout: in_ty.layout.clone(),
        };
        self.add_node(Op::Slice { input, ranges }, out_ty)
    }

    pub fn expand(&mut self, input: NodeId, shape: Vec<Dim>) -> NodeId {
        let mut out_ty = self.ty(input).clone();
        out_ty.shape = shape.clone();
        self.add_node(Op::Expand { input, shape }, out_ty)
    }

    pub fn concat(&mut self, inputs: Vec<NodeId>, axis: usize) -> NodeId {
        let first = inputs[0];
        let first_ty = self.ty(first).clone();
        let mut shape = first_ty.shape.clone();

        for &inp in inputs.iter().skip(1) {
            let ty = self.ty(inp);
            let a = &shape[axis];
            let b = &ty.shape[axis];
            shape[axis] = match (a.as_const(), b.as_const()) {
                (Some(av), Some(bv)) => Dim::constant(av + bv),
                _ => Dim::add(a.clone(), b.clone()),
            };
        }

        let out_ty = TensorType {
            shape,
            dtype: first_ty.dtype,
            layout: first_ty.layout.clone(),
        };
        self.add_node(Op::Concat { inputs, axis }, out_ty)
    }
}

fn reduce_shape(shape: &[Dim], axes: &[usize], keepdim: bool) -> Vec<Dim> {
    if keepdim {
        shape
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if axes.contains(&i) {
                    Dim::constant(1)
                } else {
                    d.clone()
                }
            })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                if axes.contains(&i) {
                    None
                } else {
                    Some(d.clone())
                }
            })
            .collect()
    }
}
