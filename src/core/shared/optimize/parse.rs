use anyhow::{bail, Context, Result};
use std::collections::HashMap;

use crate::core::shared::dtype::{DType, Scalar};
use crate::core::shared::graph::{Graph, Node, NodeId, Op};

struct ParseContext<'a> {
    original: &'a Graph,
    termdag: &'a egglog::TermDag,
    shape_table: &'a [Vec<usize>],
    perm_table: &'a [Vec<usize>],
    memo: &'a mut HashMap<egglog::TermId, NodeId>,
}

/// Parse an egglog extracted term back into a Graph.
/// Reuses Load buffers from the original graph.
/// Uses memoization to ensure identical subterms map to the same NodeId.
pub(super) fn parse_extracted_term(
    original: &Graph,
    termdag: &egglog::TermDag,
    term: &egglog::Term,
    root_dtype: DType,
    shape_table: &[Vec<usize>],
    perm_table: &[Vec<usize>],
) -> Result<(Graph, NodeId)> {
    let mut graph = Graph::new();
    let mut memo: HashMap<egglog::TermId, NodeId> = HashMap::new();
    let mut context = ParseContext {
        original,
        termdag,
        shape_table,
        perm_table,
        memo: &mut memo,
    };

    // The term returned by ExtractBest is already the root
    // We'll parse it and track TermIds for all subterms
    let root = parse_term_direct(&mut context, term, &mut graph, root_dtype)?;
    Ok((graph, root))
}

/// Helper to parse a Term when we don't have its TermId yet
fn parse_term_direct(
    context: &mut ParseContext<'_>,
    term: &egglog::Term,
    graph: &mut Graph,
    dtype: DType,
) -> Result<NodeId> {
    // For the root term, we don't have a TermId to check in memo
    // But for all child terms (via TermIds in args), we'll use memoization
    match term {
        egglog::Term::App(_head, _args) => {
            // For children, we have TermIds and can memoize
            // Parse using the TermId-based function
            // Since all children are TermIds, we can delegate to parse_term
            parse_term_from_app(context, term, graph, dtype)
        }
        egglog::Term::Lit(_) | egglog::Term::Var(_) => {
            bail!("Unexpected leaf term in extracted output")
        }
    }
}

fn parse_term(
    context: &mut ParseContext<'_>,
    term_id: egglog::TermId,
    graph: &mut Graph,
    dtype: DType,
) -> Result<NodeId> {
    // Check memo first - if we've already parsed this TermId, reuse the NodeId
    if let Some(&node_id) = context.memo.get(&term_id) {
        return Ok(node_id);
    }

    let term = context.termdag.get(term_id).clone();
    let node_id = parse_term_from_app(context, &term, graph, dtype)?;

    // Store in memo before returning
    context.memo.insert(term_id, node_id);
    Ok(node_id)
}

fn parse_term_from_app(
    context: &mut ParseContext<'_>,
    term: &egglog::Term,
    graph: &mut Graph,
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
                ("tLoad", [id]) => {
                    let egglog::Term::Lit(egglog::ast::Literal::Int(raw)) =
                        context.termdag.get(*id)
                    else {
                        bail!("tLoad expects int literal id");
                    };
                    let id = usize::try_from(*raw).context("tLoad id out of range")?;
                    let orig_node = context.original.node(NodeId(id));
                    graph.add_node(Node {
                        op: Op::Load,
                        inputs: vec![],
                        shape: orig_node.shape.clone(),
                        dtype: orig_node.dtype,
                        buffer: orig_node.buffer.clone(),
                    })
                }
                ("tConst", [val]) => {
                    let egglog::Term::Lit(egglog::ast::Literal::Float(raw)) =
                        context.termdag.get(*val)
                    else {
                        bail!("tConst expects float literal");
                    };
                    let scalar = Scalar::from_f64(raw.into_inner(), dtype);
                    graph.add_node(Node {
                        op: Op::Const(scalar),
                        inputs: vec![],
                        shape: vec![1],
                        dtype,
                        buffer: None,
                    })
                }
                ("tAdd", [a, b]) | ("tSub", [a, b]) | ("tMul", [a, b]) | ("tDiv", [a, b]) => {
                    let lhs = parse_term(context, *a, graph, dtype)?;
                    let rhs = parse_term(context, *b, graph, dtype)?;
                    let op = match head.as_str() {
                        "tAdd" => Op::Add,
                        "tSub" => Op::Sub,
                        "tMul" => Op::Mul,
                        "tDiv" => Op::Div,
                        _ => unreachable!(),
                    };
                    let shape = graph.node(lhs).shape.clone();
                    graph.add_node(Node {
                        op,
                        inputs: vec![lhs, rhs],
                        shape,
                        dtype,
                        buffer: None,
                    })
                }
                ("tExp", [a]) | ("tLn", [a]) | ("tSqrt", [a]) | ("tNeg", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    let op = match head.as_str() {
                        "tExp" => Op::Exp,
                        "tLn" => Op::Ln,
                        "tSqrt" => Op::Sqrt,
                        "tNeg" => Op::Neg,
                        _ => unreachable!(),
                    };
                    let shape = graph.node(arg).shape.clone();
                    graph.add_node(Node {
                        op,
                        inputs: vec![arg],
                        shape,
                        dtype,
                        buffer: None,
                    })
                }
                ("tReshape", [a, sid]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    let shape_idx = parse_idx(context.termdag, *sid, "tReshape")?;
                    let shape = context
                        .shape_table
                        .get(shape_idx)
                        .cloned()
                        .context("tReshape shape id out of range")?;
                    graph.add_node(Node {
                        op: Op::Reshape,
                        inputs: vec![arg],
                        shape,
                        dtype,
                        buffer: None,
                    })
                }
                ("tPermute", [a, pid]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    let perm_idx = parse_idx(context.termdag, *pid, "tPermute")?;
                    let perm = context
                        .perm_table
                        .get(perm_idx)
                        .cloned()
                        .context("tPermute permutation id out of range")?;
                    let in_shape = graph.node(arg).shape.clone();
                    let shape: Vec<usize> = perm.iter().map(|&i| in_shape[i]).collect();
                    graph.add_node(Node {
                        op: Op::Permute(perm),
                        inputs: vec![arg],
                        shape,
                        dtype,
                        buffer: None,
                    })
                }
                ("tTranspose", [a, d1, d2]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    let dim_1 = parse_idx(context.termdag, *d1, "tTranspose")?;
                    let dim_2 = parse_idx(context.termdag, *d2, "tTranspose")?;
                    let mut shape = graph.node(arg).shape.clone();
                    shape.swap(dim_1, dim_2);
                    graph.add_node(Node {
                        op: Op::Transpose(dim_1, dim_2),
                        inputs: vec![arg],
                        shape,
                        dtype,
                        buffer: None,
                    })
                }
                ("tExpand", [a, sid]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    let shape_idx = parse_idx(context.termdag, *sid, "tExpand")?;
                    let shape = context
                        .shape_table
                        .get(shape_idx)
                        .cloned()
                        .context("tExpand shape id out of range")?;
                    graph.add_node(Node {
                        op: Op::Expand,
                        inputs: vec![arg],
                        shape,
                        dtype,
                        buffer: None,
                    })
                }
                ("tSqueeze", [a]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    let mut shape: Vec<usize> = graph
                        .node(arg)
                        .shape
                        .iter()
                        .copied()
                        .filter(|&s| s != 1)
                        .collect();
                    if shape.is_empty() {
                        shape.push(1);
                    }
                    graph.add_node(Node {
                        op: Op::Squeeze,
                        inputs: vec![arg],
                        shape,
                        dtype,
                        buffer: None,
                    })
                }
                ("tUnsqueeze", [a, rank]) => {
                    let arg = parse_term(context, *a, graph, dtype)?;
                    let new_rank = parse_idx(context.termdag, *rank, "tUnsqueeze")?;
                    let in_shape = graph.node(arg).shape.clone();
                    if new_rank < in_shape.len() {
                        bail!(
                            "tUnsqueeze new rank {} smaller than {}",
                            new_rank,
                            in_shape.len()
                        );
                    }
                    let mut shape = vec![1; new_rank - in_shape.len()];
                    shape.extend_from_slice(&in_shape);
                    graph.add_node(Node {
                        op: Op::Unsqueeze(new_rank),
                        inputs: vec![arg],
                        shape,
                        dtype,
                        buffer: None,
                    })
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
