___veNum___

- Stands for vectorized N-dimensional numerical arrays. Tensor / NdArray library.
- Currently capable of creating Naive CPU Tensors of type T and performing 
    - broadcasted algebraic operations
    - Nd matrix multiplication 
    - 1d and 2d convolution / cross-correlation with strides
    - reduce operations such as sum, product, max, min and pooling
    - transformations such as view/reshape, permute/transpose, flip, expand, pad, slice, squeeze, unsqueeze

- Clone the repo and run examples
```bash
cargo run -r --example <example_name>
```
- Use as library
```bash
cargo add --git https://github.com/shettysach/veNum
```
