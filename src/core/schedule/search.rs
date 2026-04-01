use crate::core::hlir::{DType, HLIRGraph};

use super::decision::{FusionGroup, FusionGroupId, FusionTopology, ScheduleDecision};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendClass {
    Cpu,
    Gpu,
    Wgsl,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelContext {
    pub loop_bounds: Vec<i64>,
    pub reduce_axes: Vec<usize>,
    pub dtype: DType,
    pub shared_budget: usize,
    pub backend: BackendClass,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostEstimate {
    pub compute_cycles: f64,
    pub memory_cycles: f64,
    pub total_cycles: f64,
    pub working_set_bytes: usize,
    pub arithmetic_intensity: f64,
}

pub trait HardwareModel {
    fn estimate_cost(&self, _ctx: &KernelContext) -> CostEstimate;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrivialHardware;

impl HardwareModel for TrivialHardware {
    fn estimate_cost(&self, _ctx: &KernelContext) -> CostEstimate {
        CostEstimate::default()
    }
}

pub struct ScheduleSearcher<H: HardwareModel> {
    pub hardware: H,
    pub beam_width: usize,
    pub max_iterations: usize,
}

impl<H: HardwareModel> ScheduleSearcher<H> {
    pub fn new(hardware: H) -> Self {
        Self {
            hardware,
            beam_width: 1,
            max_iterations: 1,
        }
    }

    pub fn search(&self, hlir: &HLIRGraph) -> Vec<(ScheduleDecision, CostEstimate)> {
        let groups: Vec<FusionGroup> = hlir
            .topo_iter()
            .map(|(id, _)| FusionGroup {
                id: FusionGroupId(id.0),
                nodes: vec![id],
                topology: FusionTopology::Chain,
            })
            .collect();

        let mut decision = ScheduleDecision::new(groups);
        for g in &decision.fusion_groups {
            decision.opts.insert(g.id, Vec::new());
        }

        let cost = self.hardware.estimate_cost(&KernelContext {
            loop_bounds: Vec::new(),
            reduce_axes: Vec::new(),
            dtype: DType::F32,
            shared_budget: 0,
            backend: BackendClass::Cpu,
        });
        vec![(decision, cost)]
    }

    pub fn search_best(
        &self,
        hlir: &HLIRGraph,
    ) -> anyhow::Result<(ScheduleDecision, CostEstimate)> {
        self.search(hlir)
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("search returned no candidates"))
    }
}
