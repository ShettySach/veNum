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

pub use core::dtype::{Buffer, DType, Scalar};
pub use core::tensor::{Context, Tensor};

pub use core::compile::compile;
pub use core::program::{CompiledProgram, Output};
