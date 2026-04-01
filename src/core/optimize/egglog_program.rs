use egglog::ast::Expr;
use egglog::prelude::*;

use crate::core::hlir::{Dim, HLIRGraph, NodeId, Op, Scalar};

const SORT_TEXPR: &str = "TExpr";

pub(super) struct ProgramData {
    pub actions: Vec<egglog::ast::Command>,
}

pub(super) fn commands() -> Vec<egglog::ast::Command> {
    let mut cmds = Vec::with_capacity(64);

    cmds.push(egglog::ast::Command::Sort(
        span!(),
        SORT_TEXPR.to_owned(),
        None,
    ));
    for (name, arity) in [
        ("tInput", 1usize),
        ("tConst", 1),
        ("tAdd", 2),
        ("tMul", 2),
        ("tNeg", 1),
        ("tRecip", 1),
        ("tExp", 1),
        ("tLog", 1),
        ("tSqrt", 1),
        ("tSin", 1),
        ("tCos", 1),
        ("tReshape", 1),
        ("tPermute", 1),
        ("tExpand", 1),
    ] {
        let schema = if arity == 1 && (name == "tInput" || name == "tConst") {
            egglog::ast::Schema::new(vec!["i64".to_owned()], SORT_TEXPR.to_owned())
        } else {
            egglog::ast::Schema::new(vec![SORT_TEXPR.to_owned(); arity], SORT_TEXPR.to_owned())
        };
        cmds.push(egglog::ast::Command::Constructor {
            span: span!(),
            name: name.to_owned(),
            schema,
            cost: None,
            unextractable: false,
        });
    }

    cmds.extend(rewrites());
    cmds
}

fn rewrites() -> Vec<egglog::ast::Command> {
    fn rw(lhs: Expr, rhs: Expr) -> egglog::ast::Command {
        egglog::ast::Command::Rewrite(
            "".to_owned(),
            egglog::ast::GenericRewrite {
                span: span!(),
                lhs,
                rhs,
                conditions: vec![],
            },
            false,
        )
    }

    let a = exprs::var("a");
    let b = exprs::var("b");
    let zero = exprs::call("tConst", vec![exprs::int(0)]);
    let one = exprs::call("tConst", vec![exprs::int(1)]);

    vec![
        rw(
            exprs::call("tAdd", vec![a.clone(), zero.clone()]),
            a.clone(),
        ),
        rw(
            exprs::call("tAdd", vec![zero.clone(), a.clone()]),
            a.clone(),
        ),
        rw(exprs::call("tMul", vec![a.clone(), one.clone()]), a.clone()),
        rw(exprs::call("tMul", vec![one.clone(), a.clone()]), a.clone()),
        rw(
            exprs::call("tMul", vec![a.clone(), zero.clone()]),
            zero.clone(),
        ),
        rw(
            exprs::call("tMul", vec![zero.clone(), a.clone()]),
            zero.clone(),
        ),
        rw(
            exprs::call("tNeg", vec![exprs::call("tNeg", vec![a.clone()])]),
            a.clone(),
        ),
        rw(
            exprs::call("tExp", vec![exprs::call("tLog", vec![a.clone()])]),
            a.clone(),
        ),
        rw(
            exprs::call("tLog", vec![exprs::call("tExp", vec![a.clone()])]),
            a.clone(),
        ),
        rw(
            exprs::call("tAdd", vec![a.clone(), b.clone()]),
            exprs::call("tAdd", vec![b.clone(), a.clone()]),
        ),
        rw(
            exprs::call("tMul", vec![a.clone(), b.clone()]),
            exprs::call("tMul", vec![b.clone(), a.clone()]),
        ),
        rw(
            exprs::call("tReshape", vec![exprs::call("tReshape", vec![a.clone()])]),
            exprs::call("tReshape", vec![a.clone()]),
        ),
        rw(
            exprs::call("tExpand", vec![exprs::call("tExpand", vec![a.clone()])]),
            exprs::call("tExpand", vec![a]),
        ),
    ]
}

pub(super) fn graph_to_actions(graph: &HLIRGraph, root: NodeId) -> ProgramData {
    let mut memo: Vec<Option<Expr>> = vec![None; graph.len()];
    let root_expr = {
        let mut builder = NodeExprBuilder {
            graph,
            memo: &mut memo,
        };
        builder.build(root)
    };

    ProgramData {
        actions: vec![egglog::ast::Command::Action(egglog::ast::Action::Let(
            span!(),
            "root".to_owned(),
            root_expr,
        ))],
    }
}

struct NodeExprBuilder<'a> {
    graph: &'a HLIRGraph,
    memo: &'a mut [Option<Expr>],
}

impl NodeExprBuilder<'_> {
    fn build(&mut self, id: NodeId) -> Expr {
        if let Some(expr) = &self.memo[id.0] {
            return expr.clone();
        }

        let node = self.graph.node(id);
        let expr = match &node.op {
            Op::Load { .. } => exprs::call("tInput", vec![exprs::int(id.0 as i64)]),
            Op::Const { value, .. } => exprs::call("tConst", vec![exprs::int(scalar_int(value))]),
            Op::Add(a, b) => exprs::call("tAdd", vec![self.build(*a), self.build(*b)]),
            Op::Mul(a, b) => exprs::call("tMul", vec![self.build(*a), self.build(*b)]),
            Op::Neg(a) => exprs::call("tNeg", vec![self.build(*a)]),
            Op::Recip(a) => exprs::call("tRecip", vec![self.build(*a)]),
            Op::Exp(a) => exprs::call("tExp", vec![self.build(*a)]),
            Op::Log(a) => exprs::call("tLog", vec![self.build(*a)]),
            Op::Sqrt(a) => exprs::call("tSqrt", vec![self.build(*a)]),
            Op::Sin(a) => exprs::call("tSin", vec![self.build(*a)]),
            Op::Cos(a) => exprs::call("tCos", vec![self.build(*a)]),
            Op::Reshape { input, shape } => {
                let _ = shape;
                exprs::call("tReshape", vec![self.build(*input)])
            }
            Op::Permute { input, axes } => {
                let _ = axes;
                exprs::call("tPermute", vec![self.build(*input)])
            }
            Op::Expand { input, shape } => {
                let _ = shape;
                exprs::call("tExpand", vec![self.build(*input)])
            }

            Op::Store { .. }
            | Op::Max(_, _)
            | Op::Min(_, _)
            | Op::Cast { .. }
            | Op::Cmp { .. }
            | Op::Where { .. }
            | Op::Reduce { .. }
            | Op::Slice { .. }
            | Op::Concat { .. } => exprs::call("tInput", vec![exprs::int(id.0 as i64)]),
        };

        self.memo[id.0] = Some(expr.clone());
        expr
    }
}

fn scalar_int(v: &Scalar) -> i64 {
    match v {
        Scalar::F32(x) => *x as i64,
        Scalar::F16(x) => *x as i64,
        Scalar::BF16(x) => *x as i64,
        Scalar::F64(x) => *x as i64,
        Scalar::I8(x) => *x as i64,
        Scalar::I16(x) => *x as i64,
        Scalar::I32(x) => *x as i64,
        Scalar::I64(x) => *x,
        Scalar::U8(x) => *x as i64,
        Scalar::U16(x) => *x as i64,
        Scalar::U32(x) => *x as i64,
        Scalar::U64(x) => *x as i64,
        Scalar::Bool(x) => {
            if *x {
                1
            } else {
                0
            }
        }
    }
}

#[allow(dead_code)]
fn _is_const_shape(shape: &[Dim]) -> bool {
    shape.iter().all(|d| matches!(d, Dim::Const(_)))
}
