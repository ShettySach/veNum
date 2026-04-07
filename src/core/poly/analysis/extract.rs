use crate::core::hlir::Symbol;
use crate::core::llir::affine::{AffineExpr, Var};
use crate::core::llir::loop_nest::Loop;
use crate::core::llir::program::Kernel;
use crate::core::llir::stmt::{AbstractVectorOp, Expr, Stmt};

use crate::core::poly::access_map::{AccessMap, memory_access_to_access_map};
use crate::core::poly::domain::{Aff, Constraint, Domain};

/// A normalized polyhedral view of a single LLIR statement.
#[derive(Clone, Debug)]
pub struct StatementInstance {
    pub stmt_id: usize,
    pub domain: Domain,
    pub reads: Vec<AccessMap>,
    pub writes: Vec<AccessMap>,
}

/// Extract polyhedral statement instances from a kernel.
///
/// Walks the loop nest top-down, accumulating loop bounds into a domain,
/// and collects read/write access maps from each leaf statement.
pub fn extract_instances(kernel: &Kernel) -> Vec<StatementInstance> {
    let mut instances = Vec::new();
    let mut counter = 0usize;

    // Start with the top-level loops as the initial domain context.
    let ctx = build_domain_from_loops(&kernel.loop_nest.loops);

    extract_from_body(&kernel.loop_nest.body, &ctx, &mut instances, &mut counter);

    instances
}

// ---------------------------------------------------------------------------
// Domain context built from a slice of loops
// ---------------------------------------------------------------------------

struct DomainCtx {
    iters: Vec<String>,
    params: Vec<Symbol>,
    constraints: Vec<Constraint>,
}

impl DomainCtx {
    fn to_domain(&self) -> Domain {
        Domain {
            iters: self.iters.clone(),
            params: self.params.clone(),
            constraints: self.constraints.clone(),
        }
    }

    fn loop_vars(&self) -> Vec<String> {
        self.iters.clone()
    }

    fn with_loop(&self, lp: &Loop) -> Self {
        let mut ctx = self.clone_ctx();
        ctx.add_loop(lp);
        ctx
    }

    fn with_affine_guard(&self, constraint: Constraint) -> Self {
        let mut ctx = self.clone_ctx();
        ctx.constraints.push(constraint);
        ctx
    }

    fn clone_ctx(&self) -> Self {
        Self {
            iters: self.iters.clone(),
            params: self.params.clone(),
            constraints: self.constraints.clone(),
        }
    }

    fn add_loop(&mut self, lp: &Loop) {
        let var = &lp.var;
        self.iters.push(var.clone());

        // lower bound: var - lower >= 0
        let lb_aff = affine_expr_to_aff(&lp.lower);
        self.constraints
            .push(Constraint::Ineq(Aff::iter_var(var).sub(&lb_aff)));

        // upper bound: upper - 1 - var >= 0
        let ub_aff = affine_expr_to_aff(&lp.upper);
        self.constraints.push(Constraint::Ineq(
            ub_aff.sub(&Aff::iter_var(var)).add(&Aff::constant(-1)),
        ));

        // Collect params from bounds.
        collect_params_from_expr(&lp.lower, &mut self.params);
        collect_params_from_expr(&lp.upper, &mut self.params);
    }
}

fn build_domain_from_loops(loops: &[Loop]) -> DomainCtx {
    let mut ctx = DomainCtx {
        iters: Vec::new(),
        params: Vec::new(),
        constraints: Vec::new(),
    };
    for lp in loops {
        ctx.add_loop(lp);
    }
    ctx
}

// ---------------------------------------------------------------------------
// Recursive body extraction
// ---------------------------------------------------------------------------

fn extract_from_body(
    stmts: &[Stmt],
    ctx: &DomainCtx,
    out: &mut Vec<StatementInstance>,
    counter: &mut usize,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Assign { dst, src } => {
                let loop_vars = ctx.loop_vars();
                let mut reads = Vec::new();
                collect_expr_accesses(src, &loop_vars, &mut reads);
                let write = memory_access_to_access_map(dst, &loop_vars);

                out.push(StatementInstance {
                    stmt_id: *counter,
                    domain: ctx.to_domain(),
                    reads,
                    writes: vec![write],
                });
                *counter += 1;
            }
            Stmt::Accumulate { dst, src, .. } => {
                let loop_vars = ctx.loop_vars();
                let mut reads = Vec::new();
                // Accumulate reads the dst too.
                reads.push(memory_access_to_access_map(dst, &loop_vars));
                collect_expr_accesses(src, &loop_vars, &mut reads);
                let write = memory_access_to_access_map(dst, &loop_vars);

                out.push(StatementInstance {
                    stmt_id: *counter,
                    domain: ctx.to_domain(),
                    reads,
                    writes: vec![write],
                });
                *counter += 1;
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                // Try to extract an affine guard from the condition.
                // For now we only handle the trivial literal-true case
                // (produced by PadTo) and otherwise conservatively ignore
                // the predicate—the domain stays unchanged.
                if let Some(constraint) = try_affine_guard(cond) {
                    let then_ctx = ctx.with_affine_guard(constraint);
                    extract_from_body(then_body, &then_ctx, out, counter);
                } else {
                    extract_from_body(then_body, ctx, out, counter);
                }
                extract_from_body(else_body, ctx, out, counter);
            }
            Stmt::Loop(lp, body) => {
                let inner_ctx = ctx.with_loop(lp);
                extract_from_body(body, &inner_ctx, out, counter);
            }
            Stmt::Epilogue { remainder_body, .. } => {
                extract_from_body(remainder_body, ctx, out, counter);
            }
            Stmt::Barrier => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Expression access collection
// ---------------------------------------------------------------------------

fn collect_expr_accesses(expr: &Expr, loop_vars: &[String], reads: &mut Vec<AccessMap>) {
    match expr {
        Expr::Load(ma) => {
            reads.push(memory_access_to_access_map(ma, loop_vars));
        }
        Expr::Unary { arg, .. } | Expr::Cast { arg, .. } => {
            collect_expr_accesses(arg, loop_vars, reads);
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_expr_accesses(lhs, loop_vars, reads);
            collect_expr_accesses(rhs, loop_vars, reads);
        }
        Expr::Ternary {
            cond,
            then_val,
            else_val,
        } => {
            collect_expr_accesses(cond, loop_vars, reads);
            collect_expr_accesses(then_val, loop_vars, reads);
            collect_expr_accesses(else_val, loop_vars, reads);
        }
        Expr::AbstractVector(vop) => {
            collect_vector_accesses(vop, loop_vars, reads);
        }
        Expr::Literal(_) => {}
    }
}

fn collect_vector_accesses(
    op: &AbstractVectorOp,
    loop_vars: &[String],
    reads: &mut Vec<AccessMap>,
) {
    match op {
        AbstractVectorOp::Fma { acc, lhs, rhs, .. } => {
            collect_expr_accesses(acc, loop_vars, reads);
            collect_expr_accesses(lhs, loop_vars, reads);
            collect_expr_accesses(rhs, loop_vars, reads);
        }
        AbstractVectorOp::HorizontalReduce { arg, .. }
        | AbstractVectorOp::Broadcast { scalar: arg, .. }
        | AbstractVectorOp::VecCast { arg, .. } => {
            collect_expr_accesses(arg, loop_vars, reads);
        }
        AbstractVectorOp::Gather { .. } => {}
        AbstractVectorOp::Scatter { value, .. } => {
            collect_expr_accesses(value, loop_vars, reads);
        }
        AbstractVectorOp::VecBinary { lhs, rhs, .. } => {
            collect_expr_accesses(lhs, loop_vars, reads);
            collect_expr_accesses(rhs, loop_vars, reads);
        }
    }
}

// ---------------------------------------------------------------------------
// Affine guard extraction from If conditions
// ---------------------------------------------------------------------------

fn try_affine_guard(cond: &Expr) -> Option<Constraint> {
    // Currently only recognizes literal true (PadTo guard placeholder).
    // Future: parse comparisons like `i < N` into affine constraints.
    match cond {
        Expr::Literal(crate::core::hlir::Scalar::Bool(true)) => None,
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn affine_expr_to_aff(expr: &AffineExpr) -> Aff {
    Aff::from(expr)
}

fn collect_params_from_expr(expr: &AffineExpr, params: &mut Vec<Symbol>) {
    for (_, var) in &expr.terms {
        if let Var::Param(sym) = var
            && !params.contains(sym)
        {
            params.push(*sym);
        }
    }
}
