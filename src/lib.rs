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

pub use core::liquid::{DType, LiquidContext, RealizedTensor, Scalar};
pub use core::naive::ops::conv;
pub use core::naive::NaiveTensor;
pub use core::shared::dtype::Buffer;
pub use core::shared::tensor::Tensor;
pub use core::solid::{
    compile, CompiledProgram, CpuSolidBackend, FusionPass, GraphPass, MemoryPlanningPass,
    OptimizationPass, PassManager, SolidBackend, SolidContext, SolidFusionPolicy, StaticBufferPlan,
    TensorSpec,
};
