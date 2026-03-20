mod egglog_program;
mod parse;

use anyhow::{Context, Result};

use egglog::prelude::*;

use super::graph::{Graph, NodeId};

/// Run the egglog optimizer on the graph rooted at `root`.
/// Returns a new, optimized Graph and the new root NodeId.
pub fn optimize(graph: &Graph, root: NodeId) -> Result<(Graph, NodeId)> {
    let mut egraph = egglog::EGraph::default();

    let span = span!();
    let mut program = egglog_program::commands();

    // Insert our DAG as egglog terms without any text parsing.
    program.extend(egglog_program::graph_to_actions(graph, root));

    // Equality saturation: 10 iterations over the default ruleset.
    program.push(egglog::ast::Command::RunSchedule(
        egglog::ast::Schedule::Repeat(
            span.clone(),
            10,
            Box::new(egglog::ast::Schedule::Run(
                span.clone(),
                egglog::ast::RunConfig {
                    ruleset: "".to_owned(),
                    until: None,
                },
            )),
        ),
    ));

    // Extract best term for `root`.
    program.push(egglog::ast::Command::Extract(
        span.clone(),
        egglog::ast::Expr::Var(span.clone(), "root".to_owned()),
        egglog::ast::Expr::Lit(span.clone(), egglog::ast::Literal::Int(0)),
    ));

    let outputs = egraph
        .run_program(program)
        .context("Failed to run egglog optimization program")?;

    let mut extracted = None;
    for output in outputs {
        if let egglog::CommandOutput::ExtractBest(termdag, _cost, term) = output {
            extracted = Some((termdag, term));
        }
    }
    let (termdag, term) = extracted.context("Egglog returned no extract output")?;

    // Parse extracted term back into a new Graph.
    let root_dtype = graph.node(root).dtype;
    let (new_graph, new_root) = parse::parse_extracted_term(graph, &termdag, &term, root_dtype)?;

    Ok((new_graph, new_root))
}
