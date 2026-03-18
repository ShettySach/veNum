use super::dtype::{DType, Scalar};
use super::graph::{Graph, Node, NodeId, Op};

/// Run the egglog optimizer on the graph rooted at `root`.
/// Returns a new, optimized Graph and the new root NodeId.
pub fn optimize(graph: &Graph, root: NodeId) -> (Graph, NodeId) {
    // 1. Convert our Graph into egglog terms.
    let mut egraph = egglog::EGraph::default();

    // Define the tensor expression sort.
    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype TExpr
                (tLoad i64)
                (tConst f64)
                (tAdd TExpr TExpr)
                (tSub TExpr TExpr)
                (tMul TExpr TExpr)
                (tDiv TExpr TExpr)
                (tExp TExpr)
                (tLn  TExpr)
                (tSqrt TExpr)
                (tNeg TExpr)
            )
            "#,
        )
        .unwrap();

    // Algebraic simplification rules.
    egraph
        .parse_and_run_program(
            None,
            r#"
            ;; --- Identity rules ---
            (rewrite (tAdd ?a (tConst 0.0)) ?a)
            (rewrite (tAdd (tConst 0.0) ?a) ?a)
            (rewrite (tSub ?a (tConst 0.0)) ?a)
            (rewrite (tMul ?a (tConst 1.0)) ?a)
            (rewrite (tMul (tConst 1.0) ?a) ?a)
            (rewrite (tDiv ?a (tConst 1.0)) ?a)

            ;; --- Zero rules ---
            (rewrite (tMul ?a (tConst 0.0)) (tConst 0.0))
            (rewrite (tMul (tConst 0.0) ?a) (tConst 0.0))

            ;; --- Double negation ---
            (rewrite (tNeg (tNeg ?a)) ?a)

            ;; --- Inverse ops ---
            (rewrite (tExp (tLn ?a)) ?a)
            (rewrite (tLn (tExp ?a)) ?a)
            (rewrite (tSqrt (tMul ?a ?a)) ?a)

            ;; --- Self-cancellation ---
            (rewrite (tSub ?a ?a) (tConst 0.0))
            (rewrite (tDiv ?a ?a) (tConst 1.0))

            ;; --- Commutativity ---
            (rewrite (tAdd ?a ?b) (tAdd ?b ?a))
            (rewrite (tMul ?a ?b) (tMul ?b ?a))

            ;; --- Strength reduction ---
            (rewrite (tAdd ?a ?a) (tMul (tConst 2.0) ?a))
            "#,
        )
        .unwrap();

    // 2. Insert our DAG as egglog terms.
    let term_str = node_to_egglog(graph, root);
    let insert_cmd = format!("(let root {})", term_str);
    egraph.parse_and_run_program(None, &insert_cmd).unwrap();

    // 3. Run equality saturation.
    egraph.parse_and_run_program(None, "(run 10)").unwrap();

    // 4. Extract the best term.
    let outputs = egraph
        .parse_and_run_program(None, "(extract root)")
        .unwrap();

    let extracted = outputs[0].to_string();

    // 5. Parse the extracted term back into a new Graph.
    let root_dtype = graph.node(root).dtype;
    let (new_graph, new_root) = parse_egglog_term(graph, &extracted, root_dtype);

    (new_graph, new_root)
}

/// Convert a graph node to an egglog s-expression string.
fn node_to_egglog(graph: &Graph, id: NodeId) -> String {
    let node = graph.node(id);
    match &node.op {
        Op::Load => format!("(tLoad {})", id.0),
        Op::Const(v) => format!("(tConst {:.1})", v.to_f64()),
        Op::Add => format!(
            "(tAdd {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Sub => format!(
            "(tSub {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Mul => format!(
            "(tMul {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Div => format!(
            "(tDiv {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Exp => format!("(tExp {})", node_to_egglog(graph, node.inputs[0])),
        Op::Ln => format!("(tLn {})", node_to_egglog(graph, node.inputs[0])),
        Op::Sqrt => format!("(tSqrt {})", node_to_egglog(graph, node.inputs[0])),
        Op::Neg => format!("(tNeg {})", node_to_egglog(graph, node.inputs[0])),

        // Non-elementwise / shape ops are currently not modeled in egglog.
        // Keep optimizer safe by treating them as opaque leaves.
        Op::Reshape(_)
        | Op::Permute(_)
        | Op::Transpose(_, _)
        | Op::Expand(_)
        | Op::Slice(_)
        | Op::Flip(_)
        | Op::Squeeze
        | Op::Unsqueeze(_)
        | Op::Pad(_, _)
        | Op::Sum(_, _)
        | Op::Prod(_, _)
        | Op::Max(_, _)
        | Op::Min(_, _) => format!("(tLoad {})", id.0),
    }
}

/// Parse an egglog extracted term back into a Graph.
/// Reuses Load buffers from the original graph.
fn parse_egglog_term(original: &Graph, term: &str, root_dtype: DType) -> (Graph, NodeId) {
    let term = term.trim();
    let mut graph = Graph::new();
    let root = parse_sexpr(original, term, &mut graph, root_dtype);
    (graph, root)
}

/// Recursive s-expression parser for egglog output.
fn parse_sexpr(original: &Graph, s: &str, graph: &mut Graph, dtype: DType) -> NodeId {
    let s = s.trim();

    if !s.starts_with('(') {
        panic!("Expected s-expression, got: {}", s);
    }

    let inner = &s[1..s.len() - 1];
    let (head, rest) = split_head(inner);

    match head {
        "tLoad" => {
            let id: usize = rest.trim().parse().unwrap();
            let orig_node = original.node(NodeId(id));
            graph.add_node(Node {
                op: Op::Load,
                inputs: vec![],
                shape: orig_node.shape.clone(),
                dtype: orig_node.dtype,
                buffer: orig_node.buffer.clone(),
            })
        }
        "tConst" => {
            let val: f64 = rest.trim().parse().unwrap();
            let scalar = Scalar::from_f64(val, dtype);
            graph.add_node(Node {
                op: Op::Const(scalar),
                inputs: vec![],
                shape: vec![1],
                dtype,
                buffer: None,
            })
        }
        "tAdd" | "tSub" | "tMul" | "tDiv" => {
            let (arg1_str, arg2_str) = split_two_args(rest);
            let lhs = parse_sexpr(original, arg1_str, graph, dtype);
            let rhs = parse_sexpr(original, arg2_str, graph, dtype);
            let op = match head {
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
        "tExp" | "tLn" | "tSqrt" | "tNeg" => {
            let arg = parse_sexpr(original, rest.trim(), graph, dtype);
            let op = match head {
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
        _ => {
            let orig_node = original.node(NodeId(0));
            graph.add_node(Node {
                op: Op::Load,
                inputs: vec![],
                shape: orig_node.shape.clone(),
                dtype: orig_node.dtype,
                buffer: orig_node.buffer.clone(),
            })
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
        // Bare token — find next whitespace or end.
        s.find(|c: char| c.is_whitespace()).unwrap_or(s.len())
    }
}
