use crate::core::hlir::{BufferId, Scalar, Symbol};
use crate::core::llir::affine::{AffineExpr, Var};
use crate::core::llir::loop_nest::Loop;
use crate::core::llir::memory::AccessKind;
use crate::core::llir::MemoryAccess;
use crate::core::llir::program::Kernel;
use crate::core::llir::stmt::{AbstractVectorOp, BinaryOp, Expr, Stmt};

use crate::core::poly::access_map::{AccessMap, memory_access_to_access_map};
use crate::core::poly::domain::{Aff, Constraint, Domain, IterName};

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
    iters: Vec<IterName>,
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

    fn loop_vars(&self) -> Vec<IterName> {
        self.iters.clone()
    }

    fn with_loop(&self, lp: &Loop) -> Self {
        let mut ctx = self.clone_ctx();
        ctx.add_loop(lp);
        ctx
    }

    fn with_affine_guard(&self, constraint: Constraint) -> Self {
        let mut ctx = self.clone_ctx();
        collect_params_from_constraint(&constraint, &mut ctx.params);
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
        self.iters.push(var.clone().into());

        // lower bound: var - lower >= 0
        let lb_aff = affine_expr_to_aff(&lp.lower);
        self.constraints
            .push(Constraint::Ineq(Aff::iter_var(var.as_str()).sub(&lb_aff)));

        // upper bound: upper - 1 - var >= 0
        let ub_aff = affine_expr_to_aff(&lp.upper);
        self.constraints.push(Constraint::Ineq(
            ub_aff
                .sub(&Aff::iter_var(var.as_str()))
                .add(&Aff::constant(-1)),
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
                let mut writes = Vec::new();
                collect_expr_accesses(src, &loop_vars, &mut reads, &mut writes);
                let write = memory_access_to_access_map(dst, &loop_vars);
                writes.push(write);

                out.push(StatementInstance {
                    stmt_id: *counter,
                    domain: ctx.to_domain(),
                    reads,
                    writes,
                });
                *counter += 1;
            }
            Stmt::Accumulate { dst, src, .. } => {
                let loop_vars = ctx.loop_vars();
                let mut reads = Vec::new();
                let mut writes = Vec::new();
                // Accumulate reads the dst too.
                reads.push(memory_access_to_access_map(dst, &loop_vars));
                collect_expr_accesses(src, &loop_vars, &mut reads, &mut writes);
                let write = memory_access_to_access_map(dst, &loop_vars);
                writes.push(write);

                out.push(StatementInstance {
                    stmt_id: *counter,
                    domain: ctx.to_domain(),
                    reads,
                    writes,
                });
                *counter += 1;
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                if let Some(guard) = try_affine_guard(cond) {
                    let then_ctx = ctx.with_affine_guard(guard.clone());
                    extract_from_body(then_body, &then_ctx, out, counter);

                    if let Some(else_guard) = negate_constraint(&guard) {
                        let else_ctx = ctx.with_affine_guard(else_guard);
                        extract_from_body(else_body, &else_ctx, out, counter);
                    } else {
                        extract_from_body(else_body, ctx, out, counter);
                    }
                } else {
                    extract_from_body(then_body, ctx, out, counter);
                    extract_from_body(else_body, ctx, out, counter);
                }
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

fn collect_expr_accesses(
    expr: &Expr,
    loop_vars: &[IterName],
    reads: &mut Vec<AccessMap>,
    writes: &mut Vec<AccessMap>,
) {
    match expr {
        Expr::Load(ma) => {
            reads.push(memory_access_to_access_map(ma, loop_vars));
        }
        Expr::Unary { arg, .. } | Expr::Cast { arg, .. } => {
            collect_expr_accesses(arg, loop_vars, reads, writes);
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_expr_accesses(lhs, loop_vars, reads, writes);
            collect_expr_accesses(rhs, loop_vars, reads, writes);
        }
        Expr::Ternary {
            cond,
            then_val,
            else_val,
        } => {
            collect_expr_accesses(cond, loop_vars, reads, writes);
            collect_expr_accesses(then_val, loop_vars, reads, writes);
            collect_expr_accesses(else_val, loop_vars, reads, writes);
        }
        Expr::AbstractVector(vop) => {
            collect_vector_accesses(vop, loop_vars, reads, writes);
        }
        Expr::Literal(_) => {}
    }
}

fn collect_vector_accesses(
    op: &AbstractVectorOp,
    loop_vars: &[IterName],
    reads: &mut Vec<AccessMap>,
    writes: &mut Vec<AccessMap>,
) {
    match op {
        AbstractVectorOp::Fma { acc, lhs, rhs, .. } => {
            collect_expr_accesses(acc, loop_vars, reads, writes);
            collect_expr_accesses(lhs, loop_vars, reads, writes);
            collect_expr_accesses(rhs, loop_vars, reads, writes);
        }
        AbstractVectorOp::HorizontalReduce { arg, .. }
        | AbstractVectorOp::Broadcast { scalar: arg, .. }
        | AbstractVectorOp::VecCast { arg, .. } => {
            collect_expr_accesses(arg, loop_vars, reads, writes);
        }
        AbstractVectorOp::Gather { base, indices, .. } => {
            collect_expr_accesses(indices, loop_vars, reads, writes);
            reads.push(vector_memory_access(*base, indices, loop_vars, AccessKind::Read));
        }
        AbstractVectorOp::Scatter {
            base,
            indices,
            value,
            ..
        } => {
            collect_expr_accesses(indices, loop_vars, reads, writes);
            collect_expr_accesses(value, loop_vars, reads, writes);
            writes.push(vector_memory_access(*base, indices, loop_vars, AccessKind::Write));
        }
        AbstractVectorOp::VecBinary { lhs, rhs, .. } => {
            collect_expr_accesses(lhs, loop_vars, reads, writes);
            collect_expr_accesses(rhs, loop_vars, reads, writes);
        }
    }
}

// ---------------------------------------------------------------------------
// Affine guard extraction from If conditions
// ---------------------------------------------------------------------------

fn try_affine_guard(cond: &Expr) -> Option<Constraint> {
    match cond {
        Expr::Literal(crate::core::hlir::Scalar::Bool(true)) => None,
        Expr::Binary { op, lhs, rhs } => {
            let lhs = expr_to_aff(lhs)?;
            let rhs = expr_to_aff(rhs)?;
            match op {
                BinaryOp::Lt => Some(Constraint::Ineq(rhs.sub(&lhs).add(&Aff::constant(-1)))),
                BinaryOp::Le => Some(Constraint::Ineq(rhs.sub(&lhs))),
                BinaryOp::Gt => Some(Constraint::Ineq(lhs.sub(&rhs).add(&Aff::constant(-1)))),
                BinaryOp::Ge => Some(Constraint::Ineq(lhs.sub(&rhs))),
                BinaryOp::Eq => Some(Constraint::Eq(lhs.sub(&rhs))),
                BinaryOp::Ne => None,
                _ => None,
            }
        }
        _ => None,
    }
}

fn negate_constraint(c: &Constraint) -> Option<Constraint> {
    match c {
        Constraint::Ineq(a) => {
            // not(a >= 0)  =>  a <= -1  =>  -a - 1 >= 0
            Some(Constraint::Ineq(a.scale(-1).add(&Aff::constant(-1))))
        }
        Constraint::Eq(_) => None,
    }
}

fn expr_to_aff(expr: &Expr) -> Option<Aff> {
    match expr {
        Expr::Literal(s) => scalar_to_i64(s).map(Aff::constant),
        Expr::Unary {
            op: crate::core::llir::stmt::UnaryOp::Neg,
            arg,
        } => Some(expr_to_aff(arg)?.scale(-1)),
        Expr::Binary { op, lhs, rhs } => {
            let lhs = expr_to_aff(lhs)?;
            let rhs = expr_to_aff(rhs)?;
            match op {
                BinaryOp::Add => Some(lhs.add(&rhs)),
                BinaryOp::Mul => {
                    if lhs.is_constant() {
                        Some(rhs.scale(lhs.constant))
                    } else if rhs.is_constant() {
                        Some(lhs.scale(rhs.constant))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn scalar_to_i64(s: &Scalar) -> Option<i64> {
    match s {
        Scalar::Bool(v) => Some(if *v { 1 } else { 0 }),
        Scalar::I8(v) => Some(*v as i64),
        Scalar::I16(v) => Some(*v as i64),
        Scalar::I32(v) => Some(*v as i64),
        Scalar::I64(v) => Some(*v),
        Scalar::U8(v) => Some(*v as i64),
        Scalar::U16(v) => Some(*v as i64),
        Scalar::U32(v) => Some(*v as i64),
        Scalar::U64(v) => i64::try_from(*v).ok(),
        Scalar::F32(v) => {
            if v.fract() == 0.0 {
                Some(*v as i64)
            } else {
                None
            }
        }
        Scalar::F64(v) => {
            if v.fract() == 0.0 {
                Some(*v as i64)
            } else {
                None
            }
        }
        Scalar::F16(_) | Scalar::BF16(_) => None,
    }
}

fn vector_memory_access(
    base: BufferId,
    indices: &Expr,
    loop_vars: &[IterName],
    access_kind: AccessKind,
) -> AccessMap {
    let idx_aff = expr_to_aff(indices).unwrap_or_else(|| {
        loop_vars
            .first()
            .map(|v| Aff::iter_var(v.clone()))
            .unwrap_or_else(|| Aff::constant(0))
    });

    memory_access_to_access_map(
        &MemoryAccess {
            buffer: base,
            indices: vec![AffineExpr::from(&idx_aff)],
            access_kind,
        },
        loop_vars,
    )
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

fn collect_params_from_constraint(c: &Constraint, params: &mut Vec<Symbol>) {
    let aff = match c {
        Constraint::Eq(a) | Constraint::Ineq(a) => a,
    };
    for (_, var) in &aff.terms {
        if let super::super::domain::PolyVar::Param(sym) = var && !params.contains(sym) {
            params.push(*sym);
        }
    }
}
