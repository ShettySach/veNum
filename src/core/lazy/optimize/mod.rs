mod egglog_program;
mod parse;
mod serialize;

use anyhow::{Context, Result};

use super::graph::{Graph, NodeId};

/// Run the egglog optimizer on the graph rooted at `root`.
/// Returns a new, optimized Graph and the new root NodeId.
pub fn optimize(graph: &Graph, root: NodeId) -> Result<(Graph, NodeId)> {
    let mut egraph = egglog::EGraph::default();

    egglog_program::define_datatype(&mut egraph).context("Failed to define egglog datatype")?;
    egglog_program::define_rewrites(&mut egraph)
        .context("Failed to define egglog rewrite rules")?;

    // Insert our DAG as an egglog term.
    let term_str = serialize::node_to_egglog(graph, root);
    let insert_cmd = format!("(let root {term_str})");
    egraph
        .parse_and_run_program(None, &insert_cmd)
        .context("Failed to insert DAG into egglog")?;

    // Equality saturation.
    egraph
        .parse_and_run_program(None, "(run 10)")
        .context("Failed to run equality saturation")?;

    // Extract best term.
    let outputs = egraph
        .parse_and_run_program(None, "(extract root)")
        .context("Failed to extract optimized term")?;

    let extracted = outputs[0].to_string();

    // Parse extracted term back into a new Graph.
    let root_dtype = graph.node(root).dtype;
    let (new_graph, new_root) = parse::parse_egglog_term(graph, &extracted, root_dtype)?;

    Ok((new_graph, new_root))
}
