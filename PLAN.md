# Lazy-First Refactor Plan

## Problem

The codebase currently treats `ETensor` (eager) as the primary, feature-rich type
and `LTensor` (lazy) as a thin wrapper for elementwise ops only. The goal is to
invert this: lazy evaluation should be the default, with eager kept only for niche
use cases.

Additionally, every tensor constructor creates its own `Graph`, causing redundant
node copying via `import_subgraph` whenever tensors from different graphs interact.

---

## Phase 0 — Fix Graph Allocation

**Problem:** `from_slice`, `from_tensor`, and `constant` each allocate a new
`Graph`. A binary op between tensors from different graphs triggers
`import_subgraph`, which deep-clones every node. For `a + b + c + d` from 4
independent tensors, this creates 4 graphs and copies nodes from 3 of them.

**Solution:** Introduce a shared `Graph` that tensors are created within.

- Add a `Session` (or similar) that owns a single `Arc<Mutex<Graph>>`.
- `Tensor::new`, `from_slice`, `constant` take a `&Session` and insert nodes into
  the shared graph — no import needed.
- Keep `import_subgraph` only for the rare case of merging tensors across sessions.
- Alternatively, use a thread-local default graph (like PyTorch's approach) so the
  API stays ergonomic without passing a session everywhere.

---

## Phase 1 — Rename and Restructure

- [ ] Rename `LTensor` → `Tensor` (the primary public type).
- [ ] Rename `ETensor` → `EagerTensor`.
- [ ] Update `lib.rs` to export `Tensor` as the default, `EagerTensor` secondary.
- [ ] Update all examples and tests.

---

## Phase 2 — Decouple `realize()` from `EagerTensor`

- [ ] Introduce `RealizedTensor` — a simple struct holding `Vec<f32>` + `Vec<usize>`
      (data + shape, no strides/offset tricks).
- [ ] `Tensor::realize()` returns `RealizedTensor` instead of `EagerTensor`.
- [ ] Add `RealizedTensor::to_eager()` for users who need `EagerTensor` features.
- [ ] This removes the lazy → eager hard dependency.

---

## Phase 3 — Add Shape Ops to the Graph

Add new `Op` variants so shape operations are lazy graph nodes:

- [ ] `Op::Reshape(Vec<usize>)`
- [ ] `Op::Permute(Vec<usize>)`
- [ ] `Op::Transpose(usize, usize)`
- [ ] `Op::Expand(Vec<usize>)` (broadcasting)
- [ ] `Op::Slice(Vec<(usize, usize)>)`
- [ ] `Op::Flip(Vec<usize>)`
- [ ] `Op::Squeeze` / `Op::Unsqueeze(usize)`
- [ ] `Op::Pad(T, Vec<(usize, usize)>)`

These don't need JIT codegen initially — the scheduler can realize them via the
existing `Shape` logic. But having them in the graph enables egglog to optimize
across shape+compute boundaries (e.g., reshape-of-reshape elimination).

---

## Phase 4 — Add Reduce Ops to the Graph

- [ ] `Op::Sum(Vec<usize>, bool)` — dimensions + keepdims
- [ ] `Op::Prod(Vec<usize>, bool)`
- [ ] `Op::Max(Vec<usize>, bool)`
- [ ] `Op::Min(Vec<usize>, bool)`
- [ ] Scheduling: reductions become their own `ScheduleItem::Reduce` (not fused
      with elementwise kernels, at least initially).

---

## Phase 5 — Add Matmul and Conv to the Graph

- [ ] `Op::Matmul`
- [ ] `Op::Conv { strides, padding }`
- [ ] These get their own `ScheduleItem` variants and codegen paths.
- [ ] Migrate examples (`matmul.rs`, `conv.rs`, `kernels.rs`) to use `Tensor`.

---

## Phase 6 — Multi-dtype Support

- [ ] Extend `DType` with `F64`, `I32`, `I64`, `U8`, etc.
- [ ] Extend `Buffer` accordingly.
- [ ] Make `Tensor` carry a `DType` (not generic over `T` — use enum dispatch
      like tinygrad/candle, not monomorphization like `EagerTensor<T>`).
- [ ] Thread dtype through JIT codegen (Cranelift supports i32/i64/f64 natively).

---

## Phase 7 — Feature-Gate Eager

- [ ] Gate `mod eager` behind `#[cfg(feature = "eager")]`.
- [ ] `EagerTensor` becomes opt-in for users who need stride tricks, in-place
      mutation patterns, or non-JIT workflows.
- [ ] Default `cargo add venum` gives only the lazy `Tensor`.

---

## Order of Operations

```
Phase 0 (graph allocation) — independent, do first for correctness/perf
Phase 1 (rename) — small diff, sets the tone
Phase 2 (decouple realize) — enables Phase 7
Phase 3 (shape ops) — biggest user-facing win
Phase 4 (reduce ops) — needed for real workloads
Phase 5 (matmul/conv) — needed for real workloads
Phase 6 (multi-dtype) — nice to have
Phase 7 (feature-gate) — cleanup, do last
```
