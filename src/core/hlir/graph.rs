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
        let in_ty = self.ty(input).clone();
        // Reshape requires contiguous input (or produces contiguous output via copy)
        // For strided inputs, the output is treated as contiguous (implicit copy)
        let out_ty = TensorType::contiguous(shape.clone(), in_ty.dtype);
        self.add_node(Op::Reshape { input, shape }, out_ty)
    }

    pub fn permute(&mut self, input: NodeId, axes: Vec<usize>) -> NodeId {
        let in_ty = self.ty(input).clone();
        let in_strides = in_ty.strides();
        let shape: Vec<Dim> = axes.iter().map(|&i| in_ty.shape[i].clone()).collect();
        let strides: Vec<Dim> = axes.iter().map(|&i| in_strides[i].clone()).collect();
        let out_ty = TensorType::strided(shape, in_ty.dtype, strides);
        self.add_node(Op::Permute { input, axes }, out_ty)
    }

    pub fn slice(&mut self, input: NodeId, ranges: Vec<Range>) -> NodeId {
        let in_ty = self.ty(input).clone();
        let in_strides = in_ty.strides();

        let shape: Vec<Dim> = ranges
            .iter()
            .map(|r| match (&r.start, &r.end) {
                (Dim::Const(s), Dim::Const(e)) => Dim::Const(e - s),
                _ => {
                    let neg_one = Dim::Const(-1);
                    let mul_result = &neg_one * &r.start;
                    &r.end + &mul_result
                }
            })
            .collect();

        // Slice preserves strides but changes the effective view
        let out_ty = TensorType::strided(shape, in_ty.dtype, in_strides);
        self.add_node(Op::Slice { input, ranges }, out_ty)
    }

    pub fn expand(&mut self, input: NodeId, shape: Vec<Dim>) -> NodeId {
        let in_ty = self.ty(input).clone();
        let in_strides = in_ty.strides();

        // Compute output strides: if input dim is 1 and output dim > 1, stride = 0
        let out_strides: Vec<Dim> = in_ty
            .shape
            .iter()
            .zip(shape.iter())
            .zip(in_strides.iter())
            .map(|((in_dim, out_dim), in_stride)| match (in_dim, out_dim) {
                (Dim::Const(1), Dim::Const(out_d)) if *out_d > 1 => Dim::Const(0),
                _ => in_stride.clone(),
            })
            .collect();

        let out_ty = TensorType::strided(shape.clone(), in_ty.dtype, out_strides);
        self.add_node(Op::Expand { input, shape }, out_ty)
    }

    pub fn broadcast(&mut self, input: NodeId, shape: Vec<Dim>) -> NodeId {
        let in_ty = self.ty(input).clone();
        let in_strides = in_ty.strides();
        let in_rank = in_ty.shape.len();
        let out_rank = shape.len();
        let rank_offset = out_rank.saturating_sub(in_rank);

        let out_strides: Vec<Dim> = shape
            .iter()
            .enumerate()
            .map(|(out_idx, _)| {
                if out_idx < rank_offset {
                    return Dim::Const(0);
                }

                let in_idx = out_idx - rank_offset;
                match &in_ty.shape[in_idx] {
                    Dim::Const(1) => Dim::Const(0),
                    _ => in_strides[in_idx].clone(),
                }
            })
            .collect();

        let out_ty = TensorType::strided(shape.clone(), in_ty.dtype, out_strides);
        self.add_node(Op::Broadcast { input, shape }, out_ty)
    }

    pub fn concat(&mut self, inputs: Vec<NodeId>, axis: usize) -> NodeId {
        let first = inputs[0];
        let first_ty = self.ty(first).clone();
        let mut shape = first_ty.shape.clone();

        for &inp in inputs.iter().skip(1) {
            let ty = self.ty(inp);
            let a = &shape[axis];
            let b = &ty.shape[axis];
            shape[axis] = match (a, b) {
                (Dim::Const(av), Dim::Const(bv)) => Dim::Const(av + bv),
                _ => a + b,
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
                    Dim::Const(1)
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
