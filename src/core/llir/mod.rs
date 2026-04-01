pub mod affine;
pub mod dependence;
pub mod loop_nest;
pub mod memory;
pub mod program;
pub mod stmt;

#[cfg(test)]
mod tests;

pub use affine::{AffineExpr, Var};
pub use dependence::{AffineConstraint, ConstraintKind, DepKind, Dependence, DependenceRelation};
pub use loop_nest::{Loop, LoopKind, LoopNest};
pub use memory::MemoryAccess;
pub use program::{Kernel, LLIRProgram};
pub use stmt::Stmt;
