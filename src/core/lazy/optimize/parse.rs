use anyhow::{bail, Context, Result};

use super::super::dtype::{DType, Scalar};
use super::super::graph::{Graph, Node, NodeId, Op};

/// Parse an egglog extracted term back into a Graph.
/// Reuses Load buffers from the original graph.
pub(super) fn parse_egglog_term(
    original: &Graph,
    term: &str,
    root_dtype: DType,
) -> Result<(Graph, NodeId)> {
    let term = term.trim();
    let mut graph = Graph::new();
    let root = parse_sexpr(original, term, &mut graph, root_dtype)?;
    Ok((graph, root))
}

/// Recursive s-expression parser for egglog output.
fn parse_sexpr(original: &Graph, s: &str, graph: &mut Graph, dtype: DType) -> Result<NodeId> {
    let s = s.trim();

    if !s.starts_with('(') {
        bail!("Expected s-expression, got: {}", s);
    }

    let inner = &s[1..s.len() - 1];
    let (head, rest) = split_head(inner);

    match head {
        "tLoad" => {
            let id: usize = rest
                .trim()
                .parse()
                .context("Failed to parse tLoad node id")?;
            let orig_node = original.node(NodeId(id));
            Ok(graph.add_node(Node {
                op: Op::Load,
                inputs: vec![],
                shape: orig_node.shape.clone(),
                dtype: orig_node.dtype,
                buffer: orig_node.buffer.clone(),
            }))
        }
        "tConst" => {
            let val: f64 = rest
                .trim()
                .parse()
                .context("Failed to parse tConst value")?;
            let scalar = Scalar::from_f64(val, dtype);
            Ok(graph.add_node(Node {
                op: Op::Const(scalar),
                inputs: vec![],
                shape: vec![1],
                dtype,
                buffer: None,
            }))
        }
        "tAdd" | "tSub" | "tMul" | "tDiv" => {
            let (arg1_str, arg2_str) = split_two_args(rest);
            let lhs = parse_sexpr(original, arg1_str, graph, dtype)?;
            let rhs = parse_sexpr(original, arg2_str, graph, dtype)?;
            let op = match head {
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
        "tExp" | "tLn" | "tSqrt" | "tNeg" => {
            let arg = parse_sexpr(original, rest.trim(), graph, dtype)?;
            let op = match head {
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
        other => {
            bail!("Unknown s-expression head: {}", other);
        }
    }
}

/// Split "head rest..." from an s-expression body.
fn split_head(s: &str) -> (&str, &str) {
    let s = s.trim();
    if let Some(idx) = s.find(|c: char| c.is_whitespace()) {
        (&s[..idx], &s[idx..])
    } else {
        (s, "")
    }
}

/// Split two s-expression arguments from a string like " (tAdd ...) (tMul ...)".
fn split_two_args(s: &str) -> (&str, &str) {
    let s = s.trim();
    let end_of_first = find_sexpr_end(s);
    let first = &s[..end_of_first];
    let second = s[end_of_first..].trim();
    (first, second)
}

/// Find the end index of the first s-expression in the string.
fn find_sexpr_end(s: &str) -> usize {
    let s = s.trim();
    if s.starts_with('(') {
        let mut depth = 0;
        for (i, c) in s.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return i + 1;
                    }
                }
                _ => {}
            }
        }
        s.len()
    } else {
        // Bare token - find next whitespace or end.
        s.find(|c: char| c.is_whitespace()).unwrap_or(s.len())
    }
}
