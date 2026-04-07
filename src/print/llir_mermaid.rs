use std::fmt::Write;

use crate::core::llir::stmt::Expr;
use crate::core::llir::{LLIRProgram, LoopKind, Stmt};

pub fn to_llir_mermaid(program: &LLIRProgram) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "flowchart TD");

    for kernel in &program.kernels {
        let kid = format!("k{}", kernel.id.0);
        let _ = writeln!(out, "  subgraph sg_{} [\"{}\"]", kid, kid);

        let meta = format!("{}_meta", kid);
        let loops = format!("{}_loops", kid);
        let body = format!("{}_body", kid);

        let _ = writeln!(
            out,
            "  {}[\"{}\"]",
            meta,
            sanitize_label(&format!("Kernel {}\\nroot=n{}", kernel.name, kernel.root.0))
        );

        let loop_chain = if kernel.loop_nest.loops.is_empty() {
            "(no loops)".to_owned()
        } else {
            kernel
                .loop_nest
                .loops
                .iter()
                .map(|lp| {
                    format!(
                        "{}:{}[{}..{}]",
                        lp.var,
                        loop_kind_label(&lp.kind),
                        lp.lower,
                        lp.upper
                    )
                })
                .collect::<Vec<_>>()
                .join(" -> ")
        };

        let _ = writeln!(
            out,
            "  {}[\"{}\"]",
            loops,
            sanitize_label(&format!("Loops\\n{}", loop_chain))
        );

        let body_summary = summarize_body(&kernel.loop_nest.body);
        let _ = writeln!(
            out,
            "  {}[\"{}\"]",
            body,
            sanitize_label(&format!("Body\\n{}", body_summary))
        );

        let _ = writeln!(out, "  {} --> {}", meta, loops);
        let _ = writeln!(out, "  {} --> {}", loops, body);
        let _ = writeln!(out, "  end");
    }

    out
}

fn summarize_body(body: &[Stmt]) -> String {
    let mut lines = Vec::new();
    for stmt in body.iter().take(6) {
        lines.push(stmt_label(stmt));
    }
    if body.len() > 6 {
        lines.push(format!("... +{} more", body.len() - 6));
    }
    lines.join("\\n")
}

fn stmt_label(stmt: &Stmt) -> String {
    match stmt {
        Stmt::Assign { dst, src } => format!("Assign b{} <- {}", dst.buffer.0, expr_label(src)),
        Stmt::Accumulate { dst, op, src } => {
            format!(
                "Accumulate {:?} b{} <- {}",
                op,
                dst.buffer.0,
                expr_label(src)
            )
        }
        Stmt::If { .. } => "If".to_owned(),
        Stmt::Loop(lp, _) => format!("Nested Loop {}:{}", lp.var, loop_kind_label(&lp.kind)),
        Stmt::Barrier => "Barrier".to_owned(),
        Stmt::Epilogue { main_loop_var, .. } => format!("Epilogue {}", main_loop_var),
    }
}

fn expr_label(expr: &Expr) -> String {
    match expr {
        Expr::Literal(_) => "Literal".to_owned(),
        Expr::Load(ma) => format!("Load b{}", ma.buffer.0),
        Expr::Unary { op, .. } => format!("Unary {:?}", op),
        Expr::Binary { op, .. } => format!("Binary {:?}", op),
        Expr::Ternary { .. } => "Ternary".to_owned(),
        Expr::Cast { to, .. } => format!("Cast {:?}", to),
        Expr::AbstractVector(_op) => "VecOp".to_owned(),
    }
}

fn loop_kind_label(kind: &LoopKind) -> &'static str {
    match kind {
        LoopKind::Sequential => "Seq",
        LoopKind::Parallel => "Par",
        LoopKind::Vectorized { .. } => "Vec",
        LoopKind::Unrolled { .. } => "Unr",
        LoopKind::Reduce { .. } => "Red",
    }
}

fn sanitize_label(s: &str) -> String {
    s.replace('"', "'").replace('{', "(").replace('}', ")")
}
