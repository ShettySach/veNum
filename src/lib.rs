/*!
```console
               __
__   _____  /\ \ \_   _ _ __ ___
\ \ / / _ \/  \/ / | | | '_ ` _ \
 \ V /  __/ /\  /| |_| | | | | | |
  \_/ \___\_\ \/  \__,_|_| |_| |_|
```

Vectorized _N_-dimensional numericals
*/

mod core;
mod print;

pub use core::compile::{compile, SearchConfig};
pub use core::cpu::{Buffer, CpuCodeGenerator, CpuModule};
pub use core::dep::NoOpDependenceAnalyzer;
pub use core::hlir::{BufferId, DType, Dim, HLIRGraph, NodeId, Scalar, Symbol};
pub use core::llir::{
    AffineConstraint, AffineExpr, ConstraintKind, DepKind, Dependence, DependenceRelation,
    LLIRProgram, Loop, LoopKind, LoopNest, MemoryAccess, Stmt, Var,
};
pub use core::runner::run_context;
pub use core::schedule::{
    BackendClass, CostEstimate, FusionGroup, FusionGroupId, FusionTopology, HardwareModel,
    KernelContext, Opt, OptOp, ScheduleDecision, ScheduleSearcher, TrivialHardware,
};
pub use core::tensor::{Context, Tensor};
pub use core::traits::{CodeGenerator, DependenceAnalyzer, ScheduleTransform};
