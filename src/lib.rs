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
pub use core::eager::ops::conv;
pub use core::eager::ETensor;
pub use core::lazy::LTensor;
