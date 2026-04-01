use anyhow::{bail, Context, Result};
use std::collections::HashMap;

use crate::core::hlir::{DType, HLIRGraph, NodeId, Op, Scalar, TensorType};

struct ParseContext<'a> {
    original: &'a HLIRGraph,
    termdag: &'a egglog::TermDag,
    memo: &'a mut HashMap<egglog::TermId, NodeId>,
}

pub(super) fn parse_extracted_term(
    original: &HLIRGraph,
    termdag: &egglog::TermDag,
    term: &egglog::Term,
    root_dtype: DType,
    _shape_table: &[Vec<usize>],
    _perm_table: &[Vec<usize>],
) -> Result<(HLIRGraph, NodeId)> {
    let mut graph = HLIRGraph::new();
    let mut memo: HashMap<egglog::TermId, NodeId> = HashMap::new();
    let mut context = ParseContext {
        original,
        termdag,
        memo: &mut memo,
    };

    let root = parse_term_direct(&mut context, term, &mut graph, root_dtype)?;
    Ok((graph, root))
}

fn parse_term_direct(
    context: &mut ParseContext<'_>,
    term: &egglog::Term,
    graph: &mut HLIRGraph,
    dtype: DType,
) -> Result<NodeId> {
    match term {
        egglog::Term::App(_, _) => parse_term_from_app(context, term, graph, dtype),
        egglog::Term::Lit(_) | egglog::Term::Var(_) => {
            bail!("Unexpected leaf term in extracted output")
        }
    }
}

fn parse_term(
    context: &mut ParseContext<'_>,
    term_id: egglog::TermId,
    graph: &mut HLIRGraph,
    dtype: DType,
) -> Result<NodeId> {
    if let Some(&node_id) = context.memo.get(&term_id) {
        return Ok(node_id);
    }

    let term = context.termdag.get(term_id).clone();
    let node_id = parse_term_from_app(context, &term, graph, dtype)?;
    context.memo.insert(term_id, node_id);
    Ok(node_id)
}

fn parse_term_from_app(
    context: &mut ParseContext<'_>,
    term: &egglog::Term,
    graph: &mut HLIRGraph,
    dtype: DType,
) -> Result<NodeId> {
    fn parse_idx(termdag: &egglog::TermDag, tid: egglog::TermId, what: &str) -> Result<usize> {
        let egglog::Term::Lit(egglog::ast::Literal::Int(raw)) = termdag.get(tid) else {
            bail!("{} expects int literal", what);
        };
        usize::try_from(*raw).context("id out of range")
    }

    match term {
        egglog::Term::App(head, args) => {
            let node_id = match (head.as_str(), args.as_slice()) {
                ("tInput", [id]) => {
                    let idx = parse_idx(context.termdag, *id, "tInput")?;
                    let old = context.original.node(NodeId(idx));
                    graph.add_node(old.op.clone(), old.ty.clone())
                }
                ("tConst", [val]) => {
                    let raw = parse_idx(context.termdag, *val, "tConst")? as i64;
                    let scalar = Scalar::from_f64(raw as f64, dtype);
                    graph.constant(scalar, vec![crate::core::hlir::Dim::Const(1)], dtype)
                }
                ("tAdd", [a, b]) => {
                    let lhs = parse_term(context, *a, graph, dtype)?;
                    let rhs = parse_term(context, *b, graph, dtype)?;
                    graph.binary(lhs, rhs, Op::Add)
                }
                ("tMul", [a, b]) => {
                    let lhs = parse_term(context, *a, graph, dtype)?;
                    let rhs = parse_term(context, *b, graph, dtype)?;
                    graph.binary(lhs, rhs, Op::Mul)
                }
                ("tNeg", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    graph.unary(arg, Op::Neg)
                }
                ("tRecip", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    graph.unary(arg, Op::Recip)
                }
                ("tExp", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    graph.unary(arg, Op::Exp)
                }
                ("tLog", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    graph.unary(arg, Op::Log)
                }
                ("tSqrt", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    graph.unary(arg, Op::Sqrt)
                }
                ("tSin", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    graph.unary(arg, Op::Sin)
                }
                ("tCos", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    graph.unary(arg, Op::Cos)
                }
                ("tReshape", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    let shape = graph.ty(arg).shape.clone();
                    graph.add_node(
                        Op::Reshape {
                            input: arg,
                            shape: shape.clone(),
                        },
                        TensorType::contiguous(shape, dtype),
                    )
                }
                ("tPermute", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    let rank = graph.ty(arg).shape.len();
                    graph.permute(arg, (0..rank).collect())
                }
                ("tExpand", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    let shape = graph.ty(arg).shape.clone();
                    graph.expand(arg, shape)
                }
                _ => bail!("Unknown term application: {}", head),
            };

            Ok(node_id)
        }
        egglog::Term::Lit(_) | egglog::Term::Var(_) => {
            bail!("Unexpected leaf term in extracted output")
        }
    }
}
