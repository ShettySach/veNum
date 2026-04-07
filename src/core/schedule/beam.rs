use std::cmp::Ordering;
use std::collections::HashSet;

use crate::core::hlir::{DType, HLIRGraph};
use crate::core::schedule::{CostEstimate, HardwareModel, KernelContext, ScheduleDecision};

pub fn run_beam_search<H: HardwareModel>(
    _hlir: &HLIRGraph,
    hardware: &H,
    seed: ScheduleDecision,
    beam_width: usize,
    max_iterations: usize,
) -> Vec<(ScheduleDecision, CostEstimate)> {
    let beam_width = beam_width.max(1);
    let mut beam: Vec<ScheduleDecision> = vec![seed];
    let mut seen = HashSet::new();
    let mut ranked = Vec::new();

    for _ in 0..max_iterations.max(1) {
        let mut expanded = Vec::new();
        for decision in &beam {
            expanded.push(decision.clone());
            for next in expand_decision(decision, hardware) {
                let key = schedule_key(&next);
                if seen.insert(key) {
                    expanded.push(next);
                }
            }
        }

        ranked = expanded
            .into_iter()
            .map(|decision| {
                let cost = estimate_decision_cost(&decision, hardware);
                (decision, cost)
            })
            .collect();

        ranked.sort_by(|a, b| {
            a.1.total_cycles
                .partial_cmp(&b.1.total_cycles)
                .unwrap_or(Ordering::Equal)
        });
        ranked.truncate(beam_width);
        beam = ranked.iter().map(|(d, _)| d.clone()).collect();
    }

    ranked
}

fn expand_decision<H: HardwareModel>(
    decision: &ScheduleDecision,
    hardware: &H,
) -> Vec<ScheduleDecision> {
    let mut out = Vec::new();
    for fg in &decision.fusion_groups {
        let mut next = decision.clone();
        let ctx = KernelContext {
            loop_bounds: vec![64; fg.nodes.len().max(1)],
            reduce_axes: vec![],
            carried_dep_axes: vec![],
            dtype: DType::F32,
            shared_budget: 48 * 1024,
            backend: crate::core::schedule::BackendClass::Cpu,
        };

        for opt in hardware.opt_candidates(&ctx).into_iter().take(8) {
            next.opts.entry(fg.id).or_default().push(opt.clone());
            out.push(next.clone());
            if let Some(v) = next.opts.get_mut(&fg.id) {
                v.pop();
            }
        }
    }
    out
}

fn estimate_decision_cost<H: HardwareModel>(
    decision: &ScheduleDecision,
    hardware: &H,
) -> CostEstimate {
    let mut total = CostEstimate::default();

    for fg in &decision.fusion_groups {
        let loop_rank = fg.nodes.len().max(1);
        let opts = decision.opts.get(&fg.id).cloned().unwrap_or_default();
        let reduce_axes = if opts
            .iter()
            .any(|o| matches!(o.op, crate::core::schedule::OptOp::GroupReduce))
        {
            vec![loop_rank - 1]
        } else {
            Vec::new()
        };

        let ctx = KernelContext {
            loop_bounds: vec![64; loop_rank],
            reduce_axes,
            carried_dep_axes: vec![],
            dtype: DType::F32,
            shared_budget: 48 * 1024,
            backend: crate::core::schedule::BackendClass::Cpu,
        };
        let mut c = hardware.estimate_cost(&ctx);
        apply_opt_cost_bias(&mut c, &opts);

        total.compute_cycles += c.compute_cycles;
        total.memory_cycles += c.memory_cycles;
        total.total_cycles += c.total_cycles;
        total.working_set_bytes += c.working_set_bytes;
        total.arithmetic_intensity += c.arithmetic_intensity;
    }

    total
}

fn apply_opt_cost_bias(cost: &mut CostEstimate, opts: &[crate::core::schedule::Opt]) {
    for opt in opts {
        match opt.op {
            crate::core::schedule::OptOp::Tile => {
                cost.memory_cycles *= 0.92;
                cost.total_cycles *= 0.94;
            }
            crate::core::schedule::OptOp::Vectorize => {
                cost.compute_cycles *= 0.75;
                cost.total_cycles *= 0.85;
                cost.arithmetic_intensity *= 1.1;
            }
            crate::core::schedule::OptOp::Unroll => {
                cost.compute_cycles *= 0.88;
                cost.total_cycles *= 0.92;
            }
            crate::core::schedule::OptOp::Parallelize => {
                cost.total_cycles *= 0.8;
            }
            crate::core::schedule::OptOp::GroupReduce => {
                cost.memory_cycles *= 0.9;
                cost.total_cycles *= 0.86;
            }
            crate::core::schedule::OptOp::PadTo => {
                cost.total_cycles *= 1.03;
            }
        }
    }
}

fn schedule_key(decision: &ScheduleDecision) -> String {
    let mut key = String::new();
    for fg in &decision.fusion_groups {
        key.push_str(&format!("g{}:", fg.id.0));
        if let Some(opts) = decision.opts.get(&fg.id) {
            for opt in opts {
                key.push_str(&format!("{:?}-{}-{};", opt.op, opt.axis, opt.amt));
            }
        }
        key.push('|');
    }
    key
}
