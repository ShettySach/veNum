# Engineering Audit

This document captures high-confidence software engineering findings across three categories:

- Unidiomatic Rust
- Inefficient or unoptimized code
- Dead/stale code

The findings are based on repo inspection and `cargo clippy` output.

## Unidiomatic Rust

- `src/core/lazy/schedule/topo.rs:283`
  - Nested `if matches!` checks on the same enum value.
  - Why: this is harder to read and maintain than a single `match`.
  - Suggestion: replace nested checks with one `match expr_node.op { ... }`.

- `src/core/lazy/schedule/topo.rs:121`
  - Multi-branch op classification via chained `if/else if`.
  - Why: a `match` over `Op` variants is more idiomatic and makes exhaustiveness clearer.
  - Suggestion: use a `match` for operation-class dispatch where practical.

- `src/core/naive/ops/reduce_ops.rs:65`
- `src/core/naive/ops/reduce_ops.rs:69`
- `src/core/naive/ops/reduce_ops.rs:83`
- `src/core/naive/ops/reduce_ops.rs:87`
  - Uses `partial_cmp(...).unwrap()`.
  - Why: can panic for non-total comparisons (for example NaN in float paths).
  - Suggestion: handle `None` explicitly or use total ordering for float-specialized paths.

- `src/core/shared/dtype.rs:43`
- `src/core/shared/dtype.rs:131`
  - Public `as_*` accessors panic on type mismatch.
  - Why: panicking API is brittle for library consumers.
  - Suggestion: add `try_as_*` returning `Result`/`Option`, keep panic variants internal if needed.

- `src/core/shared/dtype.rs:122`
  - `Buffer` exposes `len()` but not `is_empty()`.
  - Why: commonly expected API pair in Rust collections.
  - Suggestion: add `pub fn is_empty(&self) -> bool`.

- `src/core/solid/program/mod.rs:4`
  - Clippy `module_inception` (`program::program`).
  - Why: redundant naming and less clear module structure.
  - Suggestion: rename inner file/module (for example `compiled_program.rs`).

## Inefficient or Unoptimized Code

- `src/core/lazy/lru_cache.rs:31`
- `src/core/lazy/lru_cache.rs:61`
  - Pattern: `contains_key` followed by `get`/`get_mut`.
  - Why: causes double hash lookups in cache hot paths.
  - Suggestion: switch to single-lookup style (`if let Some(...) = map.get_mut(...)`).

- `src/core/lazy/plan/buffer_pool.rs:29`
- `src/core/lazy/tensor/realize.rs:145`
- `src/core/lazy/tensor/realize.rs:177`
  - `BufferPool` has acquire-only behavior; no release path is used.
  - Why: no effective reuse means extra complexity without allocation win.
  - Suggestion: implement `release` and return temporary buffers when lifetimes end, or remove pool until complete.

- `src/core/lazy/schedule/fused_kernel.rs:126`
- `src/core/lazy/schedule/fused_kernel.rs:146`
  - Uses repeated `Vec::contains` in recursive input collection.
  - Why: can become quadratic with larger fused expressions.
  - Suggestion: maintain a side `HashSet<NodeId>` for membership and keep `Vec` only for stable order.

- `src/core/solid/tensor/ops_reduce.rs:56`
  - `compute_reduced_shape` repeatedly checks membership with `contains` over a vector.
  - Why: avoidable repeated linear scans.
  - Suggestion: precompute axis membership structure (`HashSet`/bitset) once.

- `src/core/shared/codegen/cranelift_setup.rs:10`
  - Shared setup exists but currently duplicated setup logic remains in other paths.
  - Why: duplicated code paths increase maintenance and drift risk.
  - Suggestion: consolidate ISA/module setup through shared helper(s).

## Dead or Stale Code

- `src/core/shared/codegen/cranelift_setup.rs:10`
  - `#[allow(dead_code)]` on `create_native_isa`, currently unused.
  - Suggestion: wire into call sites or remove.

- `src/core/shared/codegen/generator.rs:16`
- `src/core/shared/codegen/generator.rs:37`
  - `KernelMetadata` and `GeneratedKernel.metadata` are marked dead and not consumed.
  - Suggestion: either use metadata in runtime/diagnostics or remove until needed.

- `src/core/shared/codegen/mod.rs:18`
- `src/core/shared/codegen/mod.rs:20`
- `src/core/lazy/mod.rs:18`
- `src/core/lazy/schedule/mod.rs:5`
- `src/core/lazy/schedule/mod.rs:9`
- `src/core/lazy/plan/mod.rs:8`
  - Multiple `#[allow(unused_imports)]` re-exports.
  - Why: often indicates stale public surface or partial migrations.
  - Suggestion: delete unused re-exports or make downstream usage explicit.

- `src/core/solid/program/program.rs:104`
- `src/core/solid/program/program.rs:113`
  - Dead stub APIs (`save/load`) guarded with `#[allow(dead_code)]`.
  - Suggestion: gate unfinished APIs behind feature flags or keep non-public until implemented.

- `src/core/display.rs:56`
  - Leftover TODO from prior implementation.
  - Suggestion: resolve or convert into tracked issue with owner and scope.

## Suggested Priority

1. **P0 (Correctness/Safety):** remove panic-prone compare unwraps in reduce ops.
2. **P1 (Performance):** fix LRU double lookups; make buffer pool reuse real.
3. **P2 (Maintainability):** remove stale re-exports/allow attributes and dead code.
4. **P3 (Style/API consistency):** convert repeated `if matches!` trees to `match`, add `is_empty`, reduce module inception.
