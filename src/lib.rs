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

pub use core::dtype::{Buffer, DType, RealizedTensor, Scalar};
pub use core::tensor::{Context, Tensor};

pub use core::backend::{CpuSolidBackend, SolidBackend};
pub use core::compile::compile;
pub use core::fusion_policy::SolidFusionPolicy;
pub use core::pass::{FusionPass, GraphPass, MemoryPlanningPass, OptimizationPass, PassManager};
pub use core::program::{CompiledProgram, StaticBufferPlan, TensorSpec};
