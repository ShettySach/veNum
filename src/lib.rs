/*!
```console
               __
__   _____  /\ \ \_   _ _ __ ___
\ \ / / _ \/  \/ / | | | '_ ` _ \
 \ V /  __/ /\  /| |_| | | | | | |
  \_/ \___\_\ \/  \__,_|_| |_| |_|
```

Vectorized N-dimensional numericals
*/

mod core;
mod print;

pub use core::compile::{SearchConfig, compile};
pub use core::cpu::{Buffer, CpuCodeGenerator, CpuModule};
pub use core::dep::NoOpDependenceAnalyzer;
pub use core::hlir::{BufferId, DType, Dim, HLIRGraph, NodeId, Scalar, Symbol};
pub use core::llir::{
    AffineConstraint, AffineExpr, ConstraintKind, DepKind, Dependence, DependenceRelation,
    LLIRProgram, Loop, LoopKind, LoopNest, MemoryAccess, Stmt, Var,
};
pub use core::poly::{
    shape_to_domain, strides_to_access, AccessMap, Aff, Constraint, Domain,
    NativeDependenceAnalyzer, PolyVar,
};
pub use core::runner::run_context;
pub use core::schedule::{
    BackendClass, CostEstimate, FusionGroup, FusionGroupId, FusionTopology, HardwareModel,
    KernelContext, Opt, OptOp, ScheduleDecision, ScheduleSearcher, TrivialHardware,
};
pub use core::tensor::{Context, Tensor};
pub use core::traits::{CodeGenerator, DependenceAnalyzer, ScheduleTransform};
