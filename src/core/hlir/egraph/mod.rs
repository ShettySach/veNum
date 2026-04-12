use std::collections::HashMap;

use super::{HLIRGraph, NodeId};
use decode::decode_with_extracted;
use extract::{extract_structural_roots, ExtractionConfig};
use region::plan_regions;

pub mod cost;
pub mod decode;
pub mod encode;
pub mod extract;
pub mod facts;
pub mod region;

#[cfg(test)]
mod tests;

pub fn canonicalize_algebraic(
    graph: &HLIRGraph,
    roots: &[NodeId],
) -> (HLIRGraph, HashMap<NodeId, NodeId>) {
    let plan = plan_regions(graph, roots);
    if !plan.barriers.is_empty() {
        return (
            graph.clone(),
            roots
                .iter()
                .map(|&root| (root, root))
                .collect::<HashMap<_, _>>(),
        );
    }

    let config = ExtractionConfig::default();
    extract_structural_roots(graph, roots, &config)
        .and_then(|(encoding, extracted)| {
            decode_with_extracted(graph, roots, &encoding, &extracted.extracted_terms)
                .map_err(|err| format!("decode failed: {err:?}"))
        })
        .unwrap_or_else(|_| {
            (
                graph.clone(),
                roots
                    .iter()
                    .map(|&root| (root, root))
                    .collect::<HashMap<_, _>>(),
            )
        })
}

#[allow(dead_code)]
pub fn egglog_algebraic(
    graph: &HLIRGraph,
    roots: &[NodeId],
) -> (HLIRGraph, HashMap<NodeId, NodeId>) {
    canonicalize_algebraic(graph, roots)
}
