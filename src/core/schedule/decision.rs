use std::collections::HashMap;

use crate::core::hlir::NodeId;

use super::opt::Opt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FusionGroupId(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionGroup {
    pub id: FusionGroupId,
    pub nodes: Vec<NodeId>,
    pub topology: FusionTopology,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FusionTopology {
    Chain,
    FanIn {
        consumer: NodeId,
        producers: Vec<NodeId>,
    },
    FanOut {
        producer: NodeId,
        consumers: Vec<NodeId>,
    },
    DAG {
        edges: Vec<(NodeId, NodeId)>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleDecision {
    pub fusion_groups: Vec<FusionGroup>,
    pub opts: HashMap<FusionGroupId, Vec<Opt>>,
}

impl ScheduleDecision {
    pub fn new(fusion_groups: Vec<FusionGroup>) -> Self {
        Self {
            fusion_groups,
            opts: HashMap::new(),
        }
    }
}
