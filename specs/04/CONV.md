Express convolution as a sum of shifted pointwise correlations.
Do not build a giant “window tensor” with kernel axes.
Let search/lowering decide later whether to materialize windows, inline slices, or rediscover an im2col-like form.

For input [B, Cin, H, W], weight [Cout, Cin, Kh, Kw], output [B, Cout, Oh, Ow]:

acc = 0
for ky in 0..Kh:
  for kx in 0..Kw:
    patch = slice(input, [0:B, 0:Cin, ky:ky+Oh, kx:kx+Ow])        // [B, Cin, Oh, Ow]
    filt  = slice(weight, [0:Cout, 0:Cin, ky:ky+1, kx:kx+1])      // [Cout, Cin, 1, 1]

    patch5 = reshape(patch, [B, 1,    Cin, Oh, Ow])
    filt5  = reshape(filt,  [1, Cout, Cin, 1,  1 ])

    prod   = mul(
      expand(patch5, [B, Cout, Cin, Oh, Ow]),
      expand(filt5,  [B, Cout, Cin, Oh, Ow]),
    )

    term   = reduce_sum(prod, axis=Cin)                           // [B, Cout, Oh, Ow]
    acc    = add(acc, term)


Why I think this is better:

It is visibly a primitive decomposition, not a special implementation trick.
It keeps convolution in ordinary graph space: Slice, Reshape, Expand, Mul, Reduce, Add.
It exposes optimization choices later rather than forcing one window-materialization shape up front.
It matches your “patterns are data, not code” goal better.

Why it is worse than the current trick in some ways:

Graph size grows with Kh * Kw.
Large kernels create many sibling branches.
You lose the “single regular window tensor” shape that some downstream code may find convenient.

I still think it is the better semantic decomposition layer.

My recommendation would be:

Keep matmul as it is conceptually in src/core/hlir/decompose.rs.
Replace frontend conv2d with the shifted-slice decomposition.
Treat im2col/window-materialization as a schedule/lowering choice later, not as the meaning of convolution.


One important caveat: your current primitive set does not express strided sampling especially elegantly. Range has start and end, but no step. That means stride-1 conv is comfortable, while stride>1 conv is awkward without either:

Extending Slice to support stepped ranges.
Adding a view-like op with affine indexing semantics.
Accepting an uglier decomposition built from extra slices/concats.

If you want strided conv to remain “primitive-free” without becoming disgusting, adding step to Range is probably the smallest principled extension.

