use egglog::ast::Expr;
use egglog::prelude::*;
use std::collections::HashMap;

use crate::core::graph::{Graph, NodeId, Op};

const SORT_TEXPR: &str = "TExpr";

pub(super) struct ProgramData {
    pub actions: Vec<egglog::ast::Command>,
    pub shapes: Vec<Vec<usize>>,
    pub perms: Vec<Vec<usize>>,
}

pub(super) fn commands() -> Vec<egglog::ast::Command> {
    let mut cmds = Vec::with_capacity(120);

    // Declare TExpr and its constructors.
    cmds.push(egglog::ast::Command::Sort(
        span!(),
        SORT_TEXPR.to_owned(),
        None,
    ));
    cmds.push(egglog::ast::Command::Constructor {
        span: span!(),
        name: "tLoad".to_owned(),
        schema: egglog::ast::Schema::new(vec!["i64".to_owned()], SORT_TEXPR.to_owned()),
        cost: None,
        unextractable: false,
    });
    cmds.push(egglog::ast::Command::Constructor {
        span: span!(),
        name: "tConst".to_owned(),
        schema: egglog::ast::Schema::new(vec!["f64".to_owned()], SORT_TEXPR.to_owned()),
        cost: None,
        unextractable: false,
    });
    for (name, arity) in [
        ("tAdd", 2usize),
        ("tSub", 2),
        ("tMul", 2),
        ("tDiv", 2),
        ("tExp", 1),
        ("tLn", 1),
        ("tSqrt", 1),
        ("tNeg", 1),
    ] {
        cmds.push(egglog::ast::Command::Constructor {
            span: span!(),
            name: name.to_owned(),
            schema: egglog::ast::Schema::new(
                vec![SORT_TEXPR.to_owned(); arity],
                SORT_TEXPR.to_owned(),
            ),
            cost: None,
            unextractable: false,
        });
    }
    cmds.push(egglog::ast::Command::Constructor {
        span: span!(),
        name: "tReshape".to_owned(),
        schema: egglog::ast::Schema::new(
            vec![SORT_TEXPR.to_owned(), "i64".to_owned()],
            SORT_TEXPR.to_owned(),
        ),
        cost: None,
        unextractable: false,
    });
    cmds.push(egglog::ast::Command::Constructor {
        span: span!(),
        name: "tPermute".to_owned(),
        schema: egglog::ast::Schema::new(
            vec![SORT_TEXPR.to_owned(), "i64".to_owned()],
            SORT_TEXPR.to_owned(),
        ),
        cost: None,
        unextractable: false,
    });
    cmds.push(egglog::ast::Command::Constructor {
        span: span!(),
        name: "tTranspose".to_owned(),
        schema: egglog::ast::Schema::new(
            vec![SORT_TEXPR.to_owned(), "i64".to_owned(), "i64".to_owned()],
            SORT_TEXPR.to_owned(),
        ),
        cost: None,
        unextractable: false,
    });
    cmds.push(egglog::ast::Command::Constructor {
        span: span!(),
        name: "tExpand".to_owned(),
        schema: egglog::ast::Schema::new(
            vec![SORT_TEXPR.to_owned(), "i64".to_owned()],
            SORT_TEXPR.to_owned(),
        ),
        cost: None,
        unextractable: false,
    });
    cmds.push(egglog::ast::Command::Constructor {
        span: span!(),
        name: "tSqueeze".to_owned(),
        schema: egglog::ast::Schema::new(vec![SORT_TEXPR.to_owned()], SORT_TEXPR.to_owned()),
        cost: None,
        unextractable: false,
    });
    cmds.push(egglog::ast::Command::Constructor {
        span: span!(),
        name: "tUnsqueeze".to_owned(),
        schema: egglog::ast::Schema::new(
            vec![SORT_TEXPR.to_owned(), "i64".to_owned()],
            SORT_TEXPR.to_owned(),
        ),
        cost: None,
        unextractable: false,
    });

    // Add rewrite rules directly as AST.
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

    // NOTE: egglog variables are plain identifiers; no '?' prefix.
    let a = exprs::var("a");
    let b = exprs::var("b");
    let s = exprs::var("s");
    let s1 = exprs::var("s1");
    let s2 = exprs::var("s2");
    let d1 = exprs::var("d1");
    let d2 = exprs::var("d2");
    let rank = exprs::var("rank");
    let zero = exprs::call("tConst", vec![exprs::float(0.0)]);
    let one = exprs::call("tConst", vec![exprs::float(1.0)]);
    let two = exprs::call("tConst", vec![exprs::float(2.0)]);

    vec![
        // --- Identity rules ---
        rw(
            exprs::call("tAdd", vec![a.clone(), zero.clone()]),
            a.clone(),
        ),
        rw(
            exprs::call("tAdd", vec![zero.clone(), a.clone()]),
            a.clone(),
        ),
        rw(
            exprs::call("tSub", vec![a.clone(), zero.clone()]),
            a.clone(),
        ),
        rw(exprs::call("tMul", vec![a.clone(), one.clone()]), a.clone()),
        rw(exprs::call("tMul", vec![one.clone(), a.clone()]), a.clone()),
        rw(exprs::call("tDiv", vec![a.clone(), one.clone()]), a.clone()),
        // --- Zero rules ---
        rw(
            exprs::call("tMul", vec![a.clone(), zero.clone()]),
            zero.clone(),
        ),
        rw(
            exprs::call("tMul", vec![zero.clone(), a.clone()]),
            zero.clone(),
        ),
        // --- Double negation ---
        rw(
            exprs::call("tNeg", vec![exprs::call("tNeg", vec![a.clone()])]),
            a.clone(),
        ),
        // --- Inverse ops ---
        rw(
            exprs::call("tExp", vec![exprs::call("tLn", vec![a.clone()])]),
            a.clone(),
        ),
        rw(
            exprs::call("tLn", vec![exprs::call("tExp", vec![a.clone()])]),
            a.clone(),
        ),
        rw(
            exprs::call(
                "tSqrt",
                vec![exprs::call("tMul", vec![a.clone(), a.clone()])],
            ),
            a.clone(),
        ),
        // --- Self-cancellation ---
        rw(
            exprs::call("tSub", vec![a.clone(), a.clone()]),
            zero.clone(),
        ),
        rw(exprs::call("tDiv", vec![a.clone(), a.clone()]), one.clone()),
        // --- Commutativity ---
        rw(
            exprs::call("tAdd", vec![a.clone(), b.clone()]),
            exprs::call("tAdd", vec![b.clone(), a.clone()]),
        ),
        rw(
            exprs::call("tMul", vec![a.clone(), b.clone()]),
            exprs::call("tMul", vec![b.clone(), a.clone()]),
        ),
        // --- Strength reduction ---
        rw(
            exprs::call("tAdd", vec![a.clone(), a.clone()]),
            exprs::call("tMul", vec![two.clone(), a.clone()]),
        ),
        // --- Shape simplifications ---
        rw(
            exprs::call(
                "tReshape",
                vec![
                    exprs::call("tReshape", vec![a.clone(), s1.clone()]),
                    s2.clone(),
                ],
            ),
            exprs::call("tReshape", vec![a.clone(), s2.clone()]),
        ),
        rw(
            exprs::call(
                "tExpand",
                vec![
                    exprs::call("tExpand", vec![a.clone(), s1.clone()]),
                    s2.clone(),
                ],
            ),
            exprs::call("tExpand", vec![a.clone(), s2.clone()]),
        ),
        rw(
            exprs::call(
                "tTranspose",
                vec![
                    exprs::call("tTranspose", vec![a.clone(), d1.clone(), d2.clone()]),
                    d1.clone(),
                    d2.clone(),
                ],
            ),
            a.clone(),
        ),
        rw(
            exprs::call(
                "tUnsqueeze",
                vec![
                    exprs::call("tUnsqueeze", vec![a.clone(), rank.clone()]),
                    rank.clone(),
                ],
            ),
            exprs::call("tUnsqueeze", vec![a.clone(), rank.clone()]),
        ),
        rw(
            exprs::call("tSqueeze", vec![exprs::call("tSqueeze", vec![a.clone()])]),
            exprs::call("tSqueeze", vec![a.clone()]),
        ),
        // --- Elementwise + shape canonicalization (same shape id on both sides) ---
        rw(
            exprs::call(
                "tAdd",
                vec![
                    exprs::call("tReshape", vec![a.clone(), s.clone()]),
                    exprs::call("tReshape", vec![b.clone(), s.clone()]),
                ],
            ),
            exprs::call(
                "tReshape",
                vec![exprs::call("tAdd", vec![a.clone(), b.clone()]), s.clone()],
            ),
        ),
        rw(
            exprs::call(
                "tSub",
                vec![
                    exprs::call("tReshape", vec![a.clone(), s.clone()]),
                    exprs::call("tReshape", vec![b.clone(), s.clone()]),
                ],
            ),
            exprs::call(
                "tReshape",
                vec![exprs::call("tSub", vec![a.clone(), b.clone()]), s.clone()],
            ),
        ),
        rw(
            exprs::call(
                "tMul",
                vec![
                    exprs::call("tReshape", vec![a.clone(), s.clone()]),
                    exprs::call("tReshape", vec![b.clone(), s.clone()]),
                ],
            ),
            exprs::call(
                "tReshape",
                vec![exprs::call("tMul", vec![a.clone(), b.clone()]), s.clone()],
            ),
        ),
        rw(
            exprs::call(
                "tDiv",
                vec![
                    exprs::call("tReshape", vec![a.clone(), s.clone()]),
                    exprs::call("tReshape", vec![b.clone(), s.clone()]),
                ],
            ),
            exprs::call(
                "tReshape",
                vec![exprs::call("tDiv", vec![a.clone(), b.clone()]), s.clone()],
            ),
        ),
        rw(
            exprs::call(
                "tNeg",
                vec![exprs::call("tReshape", vec![a.clone(), s.clone()])],
            ),
            exprs::call(
                "tReshape",
                vec![exprs::call("tNeg", vec![a.clone()]), s.clone()],
            ),
        ),
        rw(
            exprs::call(
                "tExp",
                vec![exprs::call("tReshape", vec![a.clone(), s.clone()])],
            ),
            exprs::call(
                "tReshape",
                vec![exprs::call("tExp", vec![a.clone()]), s.clone()],
            ),
        ),
        rw(
            exprs::call(
                "tLn",
                vec![exprs::call("tReshape", vec![a.clone(), s.clone()])],
            ),
            exprs::call(
                "tReshape",
                vec![exprs::call("tLn", vec![a.clone()]), s.clone()],
            ),
        ),
        rw(
            exprs::call(
                "tSqrt",
                vec![exprs::call("tReshape", vec![a.clone(), s.clone()])],
            ),
            exprs::call(
                "tReshape",
                vec![exprs::call("tSqrt", vec![a.clone()]), s.clone()],
            ),
        ),
    ]
}

pub(super) fn graph_to_actions(graph: &Graph, root: NodeId) -> ProgramData {
    let node_count = graph.nodes.len();
    let mut memo: Vec<Option<Expr>> = vec![None; node_count];
    let mut shape_ids: HashMap<Vec<usize>, i64> = HashMap::with_capacity(node_count / 4);
    let mut shape_table: Vec<Vec<usize>> = Vec::with_capacity(node_count / 4);
    let mut perm_ids: HashMap<Vec<usize>, i64> = HashMap::with_capacity(node_count / 8);
    let mut perm_table: Vec<Vec<usize>> = Vec::with_capacity(node_count / 8);

    let root_expr = {
        let mut builder = NodeExprBuilder {
            graph,
            memo: &mut memo,
            shape_ids: &mut shape_ids,
            shape_table: &mut shape_table,
            perm_ids: &mut perm_ids,
            perm_table: &mut perm_table,
        };
        builder.build(root)
    };
    let actions = vec![egglog::ast::Command::Action(egglog::ast::Action::Let(
        span!(),
        "root".to_owned(),
        root_expr,
    ))];

    ProgramData {
        actions,
        shapes: shape_table,
        perms: perm_table,
    }
}

fn intern_id(
    map: &mut HashMap<Vec<usize>, i64>,
    table: &mut Vec<Vec<usize>>,
    data: &[usize],
) -> i64 {
    if let Some(id) = map.get(data) {
        *id
    } else {
        let id = table.len() as i64;
        let key = data.to_vec();
        map.insert(key.clone(), id);
        table.push(key);
        id
    }
}

struct NodeExprBuilder<'a> {
    graph: &'a Graph,
    memo: &'a mut [Option<Expr>],
    shape_ids: &'a mut HashMap<Vec<usize>, i64>,
    shape_table: &'a mut Vec<Vec<usize>>,
    perm_ids: &'a mut HashMap<Vec<usize>, i64>,
    perm_table: &'a mut Vec<Vec<usize>>,
}

impl NodeExprBuilder<'_> {
    fn build(&mut self, id: NodeId) -> Expr {
        if let Some(expr) = &self.memo[id.0] {
            return expr.clone();
        }

        let node = self.graph.node(id);
        let expr = match &node.op {
            Op::Load => exprs::call("tLoad", vec![exprs::int(id.0 as i64)]),
            Op::Const(v) => exprs::call("tConst", vec![exprs::float(v.to_f64())]),
            Op::Add => self.binary("tAdd", node.inputs[0], node.inputs[1]),
            Op::Sub => self.binary("tSub", node.inputs[0], node.inputs[1]),
            Op::Mul => self.binary("tMul", node.inputs[0], node.inputs[1]),
            Op::Div => self.binary("tDiv", node.inputs[0], node.inputs[1]),
            Op::Exp => self.unary("tExp", node.inputs[0]),
            Op::Ln => self.unary("tLn", node.inputs[0]),
            Op::Sqrt => self.unary("tSqrt", node.inputs[0]),
            Op::Neg => self.unary("tNeg", node.inputs[0]),
            Op::Reshape => {
                let shape_id = self.intern_shape(&node.shape);
                exprs::call(
                    "tReshape",
                    vec![self.build(node.inputs[0]), exprs::int(shape_id)],
                )
            }
            Op::Permute(axes) => {
                let perm_id = self.intern_perm(axes);
                exprs::call(
                    "tPermute",
                    vec![self.build(node.inputs[0]), exprs::int(perm_id)],
                )
            }
            Op::Transpose(d1, d2) => exprs::call(
                "tTranspose",
                vec![
                    self.build(node.inputs[0]),
                    exprs::int(*d1 as i64),
                    exprs::int(*d2 as i64),
                ],
            ),
            Op::Expand => {
                let shape_id = self.intern_shape(&node.shape);
                exprs::call(
                    "tExpand",
                    vec![self.build(node.inputs[0]), exprs::int(shape_id)],
                )
            }
            Op::Squeeze => self.unary("tSqueeze", node.inputs[0]),
            Op::Unsqueeze(new_rank) => exprs::call(
                "tUnsqueeze",
                vec![self.build(node.inputs[0]), exprs::int(*new_rank as i64)],
            ),

            // Ops not modeled in egglog remain opaque leaves.
            Op::Slice(_)
            | Op::Flip(_)
            | Op::Pad(_, _)
            | Op::Sum(_, _)
            | Op::Prod(_, _)
            | Op::Max(_, _)
            | Op::Min(_, _) => exprs::call("tLoad", vec![exprs::int(id.0 as i64)]),
        };

        self.memo[id.0] = Some(expr.clone());
        expr
    }

    fn unary(&mut self, ctor: &str, input: NodeId) -> Expr {
        exprs::call(ctor, vec![self.build(input)])
    }

    fn binary(&mut self, ctor: &str, lhs: NodeId, rhs: NodeId) -> Expr {
        exprs::call(ctor, vec![self.build(lhs), self.build(rhs)])
    }

    fn intern_shape(&mut self, shape: &[usize]) -> i64 {
        intern_id(self.shape_ids, self.shape_table, shape)
    }

    fn intern_perm(&mut self, perm: &[usize]) -> i64 {
        intern_id(self.perm_ids, self.perm_table, perm)
    }
}
