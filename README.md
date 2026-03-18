<div align="center">
    <img src="assets/icon.png" width="25%">
</div>

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

##### credits

- [kurtschelfthout/tensorken](https://github.com/kurtschelfthout/tensorken)
- [huggingface/candle](https://github.com/huggingface/candle)
- [minitorch/minitorch](https://github.com/minitorch/minitorch)
- [nreHieW/r-nn](https://github.com/nreHieW/r-nn)
- [assets/venom.png](https://www.reddit.com/r/metalgearsolid/comments/2xn8f2/i_heard_yall_like_sprites/)

##### resources

- [pytorch-internals by ezyang](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [tensorken articles by Kurt Schelfthout](https://getcode.substack.com/p/fun-and-hackable-tensors-in-rust)
- [MiniTorch](https://minitorch.github.io/)
- [Convolutions](https://youtu.be/Lakz2MoHy6o)
