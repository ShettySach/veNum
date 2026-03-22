use anyhow::{bail, Context, Result};
use std::collections::HashMap;

use super::super::dtype::{DType, Scalar};
use super::super::graph::{Graph, Node, NodeId, Op};

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

    // The term returned by ExtractBest is already the root
    // We'll parse it and track TermIds for all subterms
    let root = parse_term_direct(
        original,
        termdag,
        term,
        &mut graph,
        root_dtype,
        shape_table,
        perm_table,
        &mut memo,
    )?;
    Ok((graph, root))
}

/// Helper to parse a Term when we don't have its TermId yet
fn parse_term_direct(
    original: &Graph,
    termdag: &egglog::TermDag,
    term: &egglog::Term,
    graph: &mut Graph,
    dtype: DType,
    shape_table: &[Vec<usize>],
    perm_table: &[Vec<usize>],
    memo: &mut HashMap<egglog::TermId, NodeId>,
) -> Result<NodeId> {
    // For the root term, we don't have a TermId to check in memo
    // But for all child terms (via TermIds in args), we'll use memoization
    match term {
        egglog::Term::App(_head, args) => {
            // For children, we have TermIds and can memoize
            // Parse using the TermId-based function
            // Since all children are TermIds, we can delegate to parse_term
            parse_term_from_app(
                original,
                termdag,
                term,
                args,
                graph,
                dtype,
                shape_table,
                perm_table,
                memo,
            )
        }
        egglog::Term::Lit(_) | egglog::Term::Var(_) => {
            bail!("Unexpected leaf term in extracted output")
        }
    }
}

fn parse_term(
    original: &Graph,
    termdag: &egglog::TermDag,
    term_id: egglog::TermId,
    graph: &mut Graph,
    dtype: DType,
    shape_table: &[Vec<usize>],
    perm_table: &[Vec<usize>],
    memo: &mut HashMap<egglog::TermId, NodeId>,
) -> Result<NodeId> {
    // Check memo first - if we've already parsed this TermId, reuse the NodeId
    if let Some(&node_id) = memo.get(&term_id) {
        return Ok(node_id);
    }

    let term = termdag.get(term_id);
    let node_id = parse_term_from_app(
        original,
        termdag,
        term,
        &[],
        graph,
        dtype,
        shape_table,
        perm_table,
        memo,
    )?;

    // Store in memo before returning
    memo.insert(term_id, node_id);
    Ok(node_id)
}

fn parse_term_from_app(
    original: &Graph,
    termdag: &egglog::TermDag,
    term: &egglog::Term,
    _args_hint: &[egglog::TermId], // Not used, but kept for signature compatibility
    graph: &mut Graph,
    dtype: DType,
    shape_table: &[Vec<usize>],
    perm_table: &[Vec<usize>],
    memo: &mut HashMap<egglog::TermId, NodeId>,
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
                    let egglog::Term::Lit(egglog::ast::Literal::Int(raw)) = termdag.get(*id) else {
                        bail!("tLoad expects int literal id");
                    };
                    let id = usize::try_from(*raw).context("tLoad id out of range")?;
                    let orig_node = original.node(NodeId(id));
                    graph.add_node(Node {
                        op: Op::Load,
                        inputs: vec![],
                        shape: orig_node.shape.clone(),
                        dtype: orig_node.dtype,
                        buffer: orig_node.buffer.clone(),
                    })
                }
                ("tConst", [val]) => {
                    let egglog::Term::Lit(egglog::ast::Literal::Float(raw)) = termdag.get(*val)
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
                    let lhs = parse_term(
                        original,
                        termdag,
                        *a,
                        graph,
                        dtype,
                        shape_table,
                        perm_table,
                        memo,
                    )?;
                    let rhs = parse_term(
                        original,
                        termdag,
                        *b,
                        graph,
                        dtype,
                        shape_table,
                        perm_table,
                        memo,
                    )?;
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
                    let arg = parse_term(
                        original,
                        termdag,
                        *a,
                        graph,
                        dtype,
                        shape_table,
                        perm_table,
                        memo,
                    )?;
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
                    let arg = parse_term(
                        original,
                        termdag,
                        *a,
                        graph,
                        dtype,
                        shape_table,
                        perm_table,
                        memo,
                    )?;
                    let shape_idx = parse_idx(termdag, *sid, "tReshape")?;
                    let shape = shape_table
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
                    let arg = parse_term(
                        original,
                        termdag,
                        *a,
                        graph,
                        dtype,
                        shape_table,
                        perm_table,
                        memo,
                    )?;
                    let perm_idx = parse_idx(termdag, *pid, "tPermute")?;
                    let perm = perm_table
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
                    let arg = parse_term(
                        original,
                        termdag,
                        *a,
                        graph,
                        dtype,
                        shape_table,
                        perm_table,
                        memo,
                    )?;
                    let dim_1 = parse_idx(termdag, *d1, "tTranspose")?;
                    let dim_2 = parse_idx(termdag, *d2, "tTranspose")?;
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
                    let arg = parse_term(
                        original,
                        termdag,
                        *a,
                        graph,
                        dtype,
                        shape_table,
                        perm_table,
                        memo,
                    )?;
                    let shape_idx = parse_idx(termdag, *sid, "tExpand")?;
                    let shape = shape_table
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
                    let arg = parse_term(
                        original,
                        termdag,
                        *a,
                        graph,
                        dtype,
                        shape_table,
                        perm_table,
                        memo,
                    )?;
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
                    let arg = parse_term(
                        original,
                        termdag,
                        *a,
                        graph,
                        dtype,
                        shape_table,
                        perm_table,
                        memo,
                    )?;
                    let new_rank = parse_idx(termdag, *rank, "tUnsqueeze")?;
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
