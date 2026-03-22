use egglog::ast::Expr;
use egglog::prelude::*;
use std::collections::HashMap;

use crate::core::lazy::graph::{Graph, NodeId, Op};

const SORT_TEXPR: &str = "TExpr";

pub(super) struct ProgramData {
    pub actions: Vec<egglog::ast::Command>,
    pub shapes: Vec<Vec<usize>>,
    pub perms: Vec<Vec<usize>>,
}

pub(super) fn commands() -> Vec<egglog::ast::Command> {
    let mut cmds = Vec::new();

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
    let mut memo: Vec<Option<Expr>> = vec![None; graph.nodes.len()];
    let mut shape_ids: HashMap<Vec<usize>, i64> = HashMap::new();
    let mut shape_table: Vec<Vec<usize>> = Vec::new();
    let mut perm_ids: HashMap<Vec<usize>, i64> = HashMap::new();
    let mut perm_table: Vec<Vec<usize>> = Vec::new();

    let root_expr = node_expr(
        graph,
        root,
        &mut memo,
        &mut shape_ids,
        &mut shape_table,
        &mut perm_ids,
        &mut perm_table,
    );
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
        table.push(key.clone());
        map.insert(key, id);
        id
    }
}

fn node_expr(
    graph: &Graph,
    id: NodeId,
    memo: &mut [Option<Expr>],
    shape_ids: &mut HashMap<Vec<usize>, i64>,
    shape_table: &mut Vec<Vec<usize>>,
    perm_ids: &mut HashMap<Vec<usize>, i64>,
    perm_table: &mut Vec<Vec<usize>>,
) -> Expr {
    if let Some(expr) = memo[id.0].clone() {
        return expr;
    }

    let node = graph.node(id);
    let expr = match &node.op {
        Op::Load => exprs::call("tLoad", vec![exprs::int(id.0 as i64)]),
        Op::Const(v) => exprs::call("tConst", vec![exprs::float(v.to_f64())]),
        Op::Add => exprs::call(
            "tAdd",
            vec![
                node_expr(
                    graph,
                    node.inputs[0],
                    memo,
                    shape_ids,
                    shape_table,
                    perm_ids,
                    perm_table,
                ),
                node_expr(
                    graph,
                    node.inputs[1],
                    memo,
                    shape_ids,
                    shape_table,
                    perm_ids,
                    perm_table,
                ),
            ],
        ),
        Op::Sub => exprs::call(
            "tSub",
            vec![
                node_expr(
                    graph,
                    node.inputs[0],
                    memo,
                    shape_ids,
                    shape_table,
                    perm_ids,
                    perm_table,
                ),
                node_expr(
                    graph,
                    node.inputs[1],
                    memo,
                    shape_ids,
                    shape_table,
                    perm_ids,
                    perm_table,
                ),
            ],
        ),
        Op::Mul => exprs::call(
            "tMul",
            vec![
                node_expr(
                    graph,
                    node.inputs[0],
                    memo,
                    shape_ids,
                    shape_table,
                    perm_ids,
                    perm_table,
                ),
                node_expr(
                    graph,
                    node.inputs[1],
                    memo,
                    shape_ids,
                    shape_table,
                    perm_ids,
                    perm_table,
                ),
            ],
        ),
        Op::Div => exprs::call(
            "tDiv",
            vec![
                node_expr(
                    graph,
                    node.inputs[0],
                    memo,
                    shape_ids,
                    shape_table,
                    perm_ids,
                    perm_table,
                ),
                node_expr(
                    graph,
                    node.inputs[1],
                    memo,
                    shape_ids,
                    shape_table,
                    perm_ids,
                    perm_table,
                ),
            ],
        ),
        Op::Exp => exprs::call(
            "tExp",
            vec![node_expr(
                graph,
                node.inputs[0],
                memo,
                shape_ids,
                shape_table,
                perm_ids,
                perm_table,
            )],
        ),
        Op::Ln => exprs::call(
            "tLn",
            vec![node_expr(
                graph,
                node.inputs[0],
                memo,
                shape_ids,
                shape_table,
                perm_ids,
                perm_table,
            )],
        ),
        Op::Sqrt => exprs::call(
            "tSqrt",
            vec![node_expr(
                graph,
                node.inputs[0],
                memo,
                shape_ids,
                shape_table,
                perm_ids,
                perm_table,
            )],
        ),
        Op::Neg => exprs::call(
            "tNeg",
            vec![node_expr(
                graph,
                node.inputs[0],
                memo,
                shape_ids,
                shape_table,
                perm_ids,
                perm_table,
            )],
        ),
        Op::Reshape => {
            let shape_id = intern_id(shape_ids, shape_table, &node.shape);
            exprs::call(
                "tReshape",
                vec![
                    node_expr(
                        graph,
                        node.inputs[0],
                        memo,
                        shape_ids,
                        shape_table,
                        perm_ids,
                        perm_table,
                    ),
                    exprs::int(shape_id),
                ],
            )
        }
        Op::Permute(axes) => {
            let perm_id = intern_id(perm_ids, perm_table, axes);
            exprs::call(
                "tPermute",
                vec![
                    node_expr(
                        graph,
                        node.inputs[0],
                        memo,
                        shape_ids,
                        shape_table,
                        perm_ids,
                        perm_table,
                    ),
                    exprs::int(perm_id),
                ],
            )
        }
        Op::Transpose(d1, d2) => exprs::call(
            "tTranspose",
            vec![
                node_expr(
                    graph,
                    node.inputs[0],
                    memo,
                    shape_ids,
                    shape_table,
                    perm_ids,
                    perm_table,
                ),
                exprs::int(*d1 as i64),
                exprs::int(*d2 as i64),
            ],
        ),
        Op::Expand => {
            let shape_id = intern_id(shape_ids, shape_table, &node.shape);
            exprs::call(
                "tExpand",
                vec![
                    node_expr(
                        graph,
                        node.inputs[0],
                        memo,
                        shape_ids,
                        shape_table,
                        perm_ids,
                        perm_table,
                    ),
                    exprs::int(shape_id),
                ],
            )
        }
        Op::Squeeze => exprs::call(
            "tSqueeze",
            vec![node_expr(
                graph,
                node.inputs[0],
                memo,
                shape_ids,
                shape_table,
                perm_ids,
                perm_table,
            )],
        ),
        Op::Unsqueeze(new_rank) => exprs::call(
            "tUnsqueeze",
            vec![
                node_expr(
                    graph,
                    node.inputs[0],
                    memo,
                    shape_ids,
                    shape_table,
                    perm_ids,
                    perm_table,
                ),
                exprs::int(*new_rank as i64),
            ],
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

    memo[id.0] = Some(expr.clone());
    expr
}
