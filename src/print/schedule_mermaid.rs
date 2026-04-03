use std::fmt::Write;

use crate::core::schedule::{FusionTopology, ScheduleDecision};

pub fn to_schedule_mermaid(decision: &ScheduleDecision) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "flowchart LR");

    for fg in &decision.fusion_groups {
        let gid = format!("g{}", fg.id.0);
        let _ = writeln!(out, "  subgraph sg_{} [\"FusionGroup {}\"]", gid, fg.id.0);
        let _ = writeln!(
            out,
            "  {}[\"{}\"]",
            gid,
            sanitize_label(&format!("Topology\\n{}", topology_label(&fg.topology)))
        );

        for (idx, node) in fg.nodes.iter().enumerate() {
            let nid = format!("{}_n{}", gid, idx);
            let _ = writeln!(out, "  {}[\"Node n{}\"]", nid, node.0);
            let _ = writeln!(out, "  {} --> {}", gid, nid);
        }

        if let Some(opts) = decision.opts.get(&fg.id) {
            for (oidx, opt) in opts.iter().enumerate() {
                let oid = format!("{}_o{}", gid, oidx);
                let _ = writeln!(
                    out,
                    "  {}[\"Opt {:?}\\naxis={} amt={}\"]",
                    oid, opt.op, opt.axis, opt.amt
                );
                let _ = writeln!(out, "  {} --> {}", gid, oid);
            }
        }

        render_topology_edges(&mut out, &gid, &fg.topology);
        let _ = writeln!(out, "  end");
    }

    out
}

fn render_topology_edges(out: &mut String, gid: &str, topology: &FusionTopology) {
    match topology {
        FusionTopology::Chain => {}
        FusionTopology::FanIn {
            consumer,
            producers,
        } => {
            let c = format!("{}_fanin_c_{}", gid, consumer.0);
            let _ = writeln!(out, "  {}[\"Consumer n{}\"]", c, consumer.0);
            for p in producers {
                let pid = format!("{}_fanin_p_{}", gid, p.0);
                let _ = writeln!(out, "  {}[\"Producer n{}\"]", pid, p.0);
                let _ = writeln!(out, "  {} --> {}", pid, c);
            }
        }
        FusionTopology::FanOut {
            producer,
            consumers,
        } => {
            let p = format!("{}_fanout_p_{}", gid, producer.0);
            let _ = writeln!(out, "  {}[\"Producer n{}\"]", p, producer.0);
            for c in consumers {
                let cid = format!("{}_fanout_c_{}", gid, c.0);
                let _ = writeln!(out, "  {}[\"Consumer n{}\"]", cid, c.0);
                let _ = writeln!(out, "  {} --> {}", p, cid);
            }
        }
        FusionTopology::DAG { edges } => {
            for (s, d) in edges {
                let sid = format!("{}_dag_s_{}", gid, s.0);
                let did = format!("{}_dag_d_{}", gid, d.0);
                let _ = writeln!(out, "  {}[\"n{}\"]", sid, s.0);
                let _ = writeln!(out, "  {}[\"n{}\"]", did, d.0);
                let _ = writeln!(out, "  {} --> {}", sid, did);
            }
        }
    }
}

fn topology_label(topology: &FusionTopology) -> &'static str {
    match topology {
        FusionTopology::Chain => "Chain",
        FusionTopology::FanIn { .. } => "FanIn",
        FusionTopology::FanOut { .. } => "FanOut",
        FusionTopology::DAG { .. } => "DAG",
    }
}

fn sanitize_label(s: &str) -> String {
    s.replace('"', "'")
}
