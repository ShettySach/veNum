use anyhow::{bail, Context, Result};

use super::super::dtype::{DType, Scalar};
use super::super::graph::{Graph, Node, NodeId, Op};

/// Parse an egglog extracted term back into a Graph.
/// Reuses Load buffers from the original graph.
pub(super) fn parse_extracted_term(
    original: &Graph,
    termdag: &egglog::TermDag,
    term: &egglog::Term,
    root_dtype: DType,
) -> Result<(Graph, NodeId)> {
    let mut graph = Graph::new();
    let root = parse_term(original, termdag, term, &mut graph, root_dtype)?;
    Ok((graph, root))
}

fn parse_term(
    original: &Graph,
    termdag: &egglog::TermDag,
    term: &egglog::Term,
    graph: &mut Graph,
    dtype: DType,
) -> Result<NodeId> {
    match term {
        egglog::Term::App(head, args) => match (head.as_str(), args.as_slice()) {
            ("tLoad", [id]) => {
                let egglog::Term::Lit(egglog::ast::Literal::Int(raw)) = termdag.get(*id) else {
                    bail!("tLoad expects int literal id");
                };
                let id = usize::try_from(*raw).context("tLoad id out of range")?;
                let orig_node = original.node(NodeId(id));
                Ok(graph.add_node(Node {
                    op: Op::Load,
                    inputs: vec![],
                    shape: orig_node.shape.clone(),
                    dtype: orig_node.dtype,
                    buffer: orig_node.buffer.clone(),
                }))
            }
            ("tConst", [val]) => {
                let egglog::Term::Lit(egglog::ast::Literal::Float(raw)) = termdag.get(*val) else {
                    bail!("tConst expects float literal");
                };
                let scalar = Scalar::from_f64(raw.into_inner(), dtype);
                Ok(graph.add_node(Node {
                    op: Op::Const(scalar),
                    inputs: vec![],
                    shape: vec![1],
                    dtype,
                    buffer: None,
                }))
            }
            ("tAdd", [a, b]) | ("tSub", [a, b]) | ("tMul", [a, b]) | ("tDiv", [a, b]) => {
                let lhs = parse_term(original, termdag, termdag.get(*a), graph, dtype)?;
                let rhs = parse_term(original, termdag, termdag.get(*b), graph, dtype)?;
                let op = match head.as_str() {
                    "tAdd" => Op::Add,
                    "tSub" => Op::Sub,
                    "tMul" => Op::Mul,
                    "tDiv" => Op::Div,
                    _ => unreachable!(),
                };
                let shape = graph.node(lhs).shape.clone();
                Ok(graph.add_node(Node {
                    op,
                    inputs: vec![lhs, rhs],
                    shape,
                    dtype,
                    buffer: None,
                }))
            }
            ("tExp", [a]) | ("tLn", [a]) | ("tSqrt", [a]) | ("tNeg", [a]) => {
                let arg = parse_term(original, termdag, termdag.get(*a), graph, dtype)?;
                let op = match head.as_str() {
                    "tExp" => Op::Exp,
                    "tLn" => Op::Ln,
                    "tSqrt" => Op::Sqrt,
                    "tNeg" => Op::Neg,
                    _ => unreachable!(),
                };
                let shape = graph.node(arg).shape.clone();
                Ok(graph.add_node(Node {
                    op,
                    inputs: vec![arg],
                    shape,
                    dtype,
                    buffer: None,
                }))
            }
            _ => bail!("Unknown term application: {}", head),
        },
        egglog::Term::Lit(_) | egglog::Term::Var(_) => {
            bail!("Unexpected leaf term in extracted output")
        }
    }
}
