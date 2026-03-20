use egglog::ast::Expr;
use egglog::prelude::*;

use super::super::graph::{Graph, NodeId, Op};

const SORT_TEXPR: &str = "TExpr";

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
    ]
}

pub(super) fn graph_to_actions(graph: &Graph, root: NodeId) -> Vec<egglog::ast::Command> {
    let mut memo: Vec<Option<Expr>> = vec![None; graph.nodes.len()];
    let root_expr = node_expr(graph, root, &mut memo);
    vec![egglog::ast::Command::Action(egglog::ast::Action::Let(
        span!(),
        "root".to_owned(),
        root_expr,
    ))]
}

fn node_expr(graph: &Graph, id: NodeId, memo: &mut [Option<Expr>]) -> Expr {
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
                node_expr(graph, node.inputs[0], memo),
                node_expr(graph, node.inputs[1], memo),
            ],
        ),
        Op::Sub => exprs::call(
            "tSub",
            vec![
                node_expr(graph, node.inputs[0], memo),
                node_expr(graph, node.inputs[1], memo),
            ],
        ),
        Op::Mul => exprs::call(
            "tMul",
            vec![
                node_expr(graph, node.inputs[0], memo),
                node_expr(graph, node.inputs[1], memo),
            ],
        ),
        Op::Div => exprs::call(
            "tDiv",
            vec![
                node_expr(graph, node.inputs[0], memo),
                node_expr(graph, node.inputs[1], memo),
            ],
        ),
        Op::Exp => exprs::call("tExp", vec![node_expr(graph, node.inputs[0], memo)]),
        Op::Ln => exprs::call("tLn", vec![node_expr(graph, node.inputs[0], memo)]),
        Op::Sqrt => exprs::call("tSqrt", vec![node_expr(graph, node.inputs[0], memo)]),
        Op::Neg => exprs::call("tNeg", vec![node_expr(graph, node.inputs[0], memo)]),

        // Non-elementwise / shape ops are currently not modeled in egglog.
        // Keep optimizer safe by treating them as opaque leaves.
        Op::Reshape
        | Op::Permute(_)
        | Op::Transpose(_, _)
        | Op::Expand
        | Op::Slice(_)
        | Op::Flip(_)
        | Op::Squeeze
        | Op::Unsqueeze(_)
        | Op::Pad(_, _)
        | Op::Sum(_, _)
        | Op::Prod(_, _)
        | Op::Max(_, _)
        | Op::Min(_, _) => exprs::call("tLoad", vec![exprs::int(id.0 as i64)]),
    };

    memo[id.0] = Some(expr.clone());
    expr
}
