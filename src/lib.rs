/*!
```console
               __
__   _____  /\ \ \_   _ _ __ ___
\ \ / / _ \/  \/ / | | | '_ ` _ \
 \ V /  __/ /\  /| |_| | | | | | |
  \_/ \___\_\ \/  \__,_|_| |_| |_|
```

Vectorized _N_-dimensional numerical arrays.
*/

mod core;

pub use core::liquid::{Context, DType, RealizedTensor, Scalar, Tensor};
pub use core::naive::ops::conv;
pub use core::naive::NaiveTensor;
pub use core::solid::{
    CompiledProgram, SolidContext, SolidFusionPolicy, StaticBufferPlan, Tensor as SolidTensor,
    TensorSpec,
};
