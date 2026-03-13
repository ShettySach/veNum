# Kernel Fusion Plan

## Problem

The scheduler treats shape ops and reduce ops as hard barriers, so nothing
actually fuses. A matmul `Load → reshape → expand → mul → sum` produces
**8 separate schedule items** (6 shape, 1 elementwise, 1 reduce) instead of
a single fused kernel. The `expand` ops physically materialize expanded
buffers, and the `mul → sum` runs as two separate passes over memory.

```
Current schedule for matmul [5,5,2] @ [2,5]:
  Shape  Reshape [5,5,2]         ← no-op copy
  Shape  Reshape [5,5,2,1]       ← no-op copy
  Shape  Expand  [5,5,2,5]       ← materializes 100 floats from 50
  Shape  Reshape [2,5]           ← no-op copy
  Shape  Reshape [1,1,2,5]       ← no-op copy
  Shape  Expand  [5,5,2,5]       ← materializes 100 floats from 10
  Fused  Mul     [5,5,2,5]       ← JIT kernel, reads 2×100, writes 100
  Reduce Sum     [5,5,5]         ← interpreted, reads 100, writes 125
```

This is O(n) temp memory and multiple passes. Tinygrad compiles this into a
single loop nest with no intermediate buffers.

---

## Goal

Fuse `Load → shape_ops → elementwise → reduce` chains into **one kernel**
that reads from the original flat buffers, computes index expressions for
shape ops, does the elementwise work, and accumulates the reduction — all
in a single pass with no temp allocations.

For the matmul example above, the fused kernel would be:

```
for b0 in 0..5:        # batch dim 0
  for b1 in 0..5:      # batch dim 1
    for n in 0..5:      # output col
      acc = 0.0
      for k in 0..2:    # reduction axis
        a_idx = b0*10 + b1*2 + k       # flat index into a's [50] buffer
        b_idx = k*5 + n                  # flat index into b's [10] buffer
        acc += a_buf[a_idx] * b_buf[b_idx]
      out[b0*25 + b1*5 + n] = acc
```

One kernel, two input buffers, zero intermediates.

---

## Architecture

### Core Concept: ShapeTracker

A `ShapeTracker` represents a virtual view of a flat buffer. It tracks how a
sequence of shape ops (reshape, expand, permute, transpose, slice, pad, etc.)
transforms the logical multi-dimensional index → physical flat offset.

```rust
struct ShapeTracker {
    shape: Vec<usize>,      // current logical shape
    strides: Vec<isize>,    // stride per dimension (0 = broadcast/expand)
    offset: isize,          // base offset into the underlying buffer
    mask: Option<Vec<(usize, usize)>>,  // valid ranges per dim (for pad/slice)
    contiguous: bool,       // cached: is this a simple row-major layout?
}
```

Each shape op transforms the tracker without touching data:
- **Reshape**: recompute strides from new shape (only if contiguous)
- **Expand**: set stride to 0 on expanded dims
- **Permute/Transpose**: reorder strides
- **Slice**: adjust offset + shape, add mask
- **Pad**: expand shape, add mask, pad value for out-of-mask reads

The tracker provides `fn index(&self, logical_idx: &[usize]) -> Option<usize>`
which maps a multi-dim index to a flat buffer offset (or `None` for padded
regions).

### Fused Schedule Item

```rust
struct FusedKernel {
    /// The loop nest shape: output dims + reduction dims
    loop_shape: Vec<usize>,
    /// Which dims are reduction axes (summed over)
    reduce_axes: Vec<usize>,
    /// Reduce operation (Sum, Prod, Max, Min) or None for pure elementwise
    reduce_op: Option<ReduceKind>,
    /// The expression tree for the inner body
    body: ExprTree,
    /// Input buffers with their ShapeTrackers
    inputs: Vec<(NodeId, ShapeTracker)>,
    /// Output shape
    output_shape: Vec<usize>,
}
```

### ExprTree (replaces recursive graph walk)

```rust
enum Expr {
    Load(usize),            // load from input buffer #i at tracked index
    Const(f32),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Unary(UnaryOp, Box<Expr>),
}
```

---

## Phases

### Phase 1 — ShapeTracker

Implement `ShapeTracker` with support for:

- [x] `new(shape)` — row-major contiguous
- [ ] `reshape(new_shape)` — recompute strides (must be contiguous)
- [ ] `expand(new_shape)` — zero strides on broadcast dims
- [ ] `permute(axes)` — reorder shape and strides
- [ ] `transpose(d1, d2)` — swap two dims in shape and strides
- [ ] `slice(ranges)` — adjust offset, shape, mask
- [ ] `pad(padding)` — expand shape, add mask
- [ ] `stride_at(dim)` / `index(logical_idx) -> Option<usize>` — core query
- [ ] `is_contiguous()` — can this be reshaped?

Reference: tinygrad's `ShapeTracker` in `tinygrad/shape/shapetracker.py`.

### Phase 2 — Scheduler rewrite

Change `build_schedule` to walk the graph and **absorb shape ops into
ShapeTrackers** instead of emitting `ScheduleItem::Shape`:

1. Walk topo order.
2. When hitting a shape op, don't emit a schedule item. Instead, compose it
   into the ShapeTracker of its input.
3. When hitting an elementwise op, check if its inputs are shape-tracked
   loads (or other elementwise ops). Build an `ExprTree`.
4. When hitting a reduce op whose input is elementwise (or shape-tracked),
   fold the reduce into the same kernel as `reduce_axes` + `reduce_op`.
5. Emit a single `FusedKernel` with the merged loop shape.

Fusion rules:
- **Shape ops**: always absorbed (never a barrier)
- **Elementwise → Elementwise**: fuse if single consumer + same numel
  (already implemented)
- **Elementwise → Reduce**: fuse into one kernel (the reduce becomes the
  outer loop, elementwise is the inner body)
- **Reduce → Elementwise**: barrier (reduce must materialize first)
- **Reduce → Reduce**: barrier (two separate kernels)

### Phase 3 — Cranelift codegen for fused kernels

Rewrite `compile_kernel` to emit a proper **loop nest** instead of a flat
`for i in 0..n` loop:

```
entry:
  for d0 in 0..loop_shape[0]:
    for d1 in 0..loop_shape[1]:
      ...
      // if reduce_axes present, init accumulator
      acc = identity_value  // 0 for sum, 1 for prod, etc.
      for rk in 0..loop_shape[reduce_axis]:
        // compute flat index via ShapeTracker for each input
        // evaluate ExprTree
        acc = reduce(acc, expr_result)
      // store acc to output[output_flat_idx]
```

The ShapeTracker's index computation compiles to integer arithmetic in
Cranelift IR (multiply by stride, add offset, etc.).

For the matmul case this produces:
- 4-deep loop nest: `b0, b1, n, k`
- `k` is the reduction axis (sum)
- Two `Load` expressions with ShapeTrackers that map `(b0, b1, k)` and
  `(k, n)` to flat offsets in the original buffers
- One `Mul` expression
- Accumulate into `acc`, store to output

### Phase 4 — Remove interpreted shape/reduce execution

Once fused codegen works:
- Remove `execute_shape_op`, `execute_reduce_op` and all the helper functions
  (`execute_expand`, `execute_permute`, etc.)
- Remove `ScheduleItem::Shape` and `ScheduleItem::Reduce` variants
- Shape ops exist only in the graph for DAG visualization; they are never
  "executed" — only compiled into index math

---

## Matmul Before and After

**Before (current):**
```
8 schedule items, 4 temp buffers, 7 memory passes
```

**After (fused):**
```
1 schedule item, 0 temp buffers, 1 memory pass
FusedKernel {
    loop_shape: [5, 5, 5, 2],    // b0, b1, n, k
    reduce_axes: [3],             // k
    reduce_op: Some(Sum),
    inputs: [
        (Load_0, ShapeTracker { shape: [5,5,2], strides: [10,2,1] }),
        (Load_1, ShapeTracker { shape: [2,5],   strides: [5,1] }),
    ],
    body: Mul(Load(0), Load(1)),
    output_shape: [5, 5, 5],
}
```

---

## Non-goals (for now)

- Tiling / blocking for cache locality
- SIMD vectorization
- GPU codegen
- Multi-output kernels
- Fusing across reduce boundaries (reduce → elementwise chains)
