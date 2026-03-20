# veNum Execution Plan

This document is the project roadmap with emphasis on repeated execution
workloads (training loops + GPT-2 style inference).

The guiding idea is:
- Keep an inner kernel JIT (Cranelift for CPU; GPU compiler later).
- Add an outer *graph JIT* that captures a whole execution plan once and
  replays it many times.

---

## Current State (as of today)

- `Tensor::realize()` rebuilds work every call:
  - (optional) egglog optimize the root subgraph
  - build a schedule
  - execute the schedule
- Kernel-level cache exists:
  - Cranelift-compiled elementwise kernels are cached in `Context` keyed by
    `KernelSignature`.
- Shape ops and reduce ops are executed by interpreted CPU helpers when they
  are not absorbed into a fused elementwise kernel.

This is correct, but training/inference will pay repeated overhead per step.

---

## Phase 1 - Graph JIT (Execution Plan Cache) ✅

Goal: cache the whole program for a root node so repeated `realize()` calls turn
into: bind inputs, run `ExecItem`s, return output.

Deliverables

- [x] Introduce `ExecItem`:
  - `Kernel { compiled, inputs, output, numel, dtype }`
  - `Shape { op, input, output, input_shape, output_shape }`
  - `Reduce { op, input, output, input_shape, output_shape }`
  - `ConstFill { value, output, numel }`

- [x] Introduce `ExecutionPlan`:
  - `exec_graph: Graph` (self-contained execution graph)
  - `items: Vec<ExecItem>`
  - `inputs: Vec<PlanInput>` (what buffers must be supplied at runtime)
  - `output: NodeId`
  - `output_shape: Vec<usize>`

- [x] Introduce `GraphSignature`:
  - Hash structural identity of the reachable program for a given root.
  - Includes: op topology, dtypes, concrete shapes, op payloads, DAG structure.

- [x] Add `plan_cache` to `Context`:
  - `HashMap<GraphSignature, Arc<ExecutionPlan>>`

- [x] Refactor `Tensor::realize()`:
  - Plan cache lookup via `GraphSignature`
  - `build_plan()` on cache miss (builds exec graph, schedule, compiles kernels)
  - `run_plan()` replays cached plan against stored exec graph

Correctness

- [x] Validate runtime inputs against plan expectations (dtype + shape + device
  once devices exist). If mismatch, build a new plan (new `GraphSignature`).
- [ ] Detect alias hazards (input buffer also written later). Insert copies or
  require distinct buffers.

Performance

- [x] Run egglog once per plan build, never per iteration.
- [x] Ensure plan replay does not allocate per step except for outputs.

---

## Phase 2 - Buffer Pool + Memory Planning ✅

Goal: reduce allocation churn during repeated plan replay.

- [x] Add `BufferPool`:
  - Reuse allocations keyed by `(DType, numel)`.
  - Shared via `Arc<Mutex<BufferPool>>` on `Context`.
  - Kernel output byte buffers acquired from pool instead of fresh allocation.

- [x] Add simple lifetime-based reuse inside a plan:
  - `compute_last_use()` tracks last step each intermediate is read.
  - Dead intermediates removed from `realized` map after their last use.
  - Pool `release()` available for future byte-level buffer recycling.

---

## Phase 3 - Fusion to Remove Shape/Reduce Barriers ✅

Goal: compile `Load -> shape ops -> elementwise -> reduce` into a single kernel
where possible (no temp buffers, single pass), matching the spirit of tinygrad.

- [x] Extend scheduler/kernel representation to include:
  - `ReduceKind` enum (`Sum`, `Prod`, `Max`, `Min`)
  - `ReduceSpec` struct (`op`, `dims`, `keepdims`)
  - `FusedKernel` extended with `expr_root`, `iter_shape`, `reduce: Option<ReduceSpec>`
  - per-input `ShapeTracker` index math for reduce-fused kernels

- [x] Update Cranelift codegen to emit loop nests:
  - Outer loop over output elements, inner loop over flattened reduce space
  - `compose_iter_index` maps output + reduce indices to full iteration index
  - `flatten_multi_index` converts multi-dim index to flat offset
  - Proper identity values for sum/prod/max/min per dtype
  - `emit_reduce_combine` for accumulator updates

- [x] Scheduler fuses upstream into reduce kernels:
  - Direct Load/Const inputs
  - Shape op chains absorbed via `try_build_tracker`
  - Single-consumer elementwise subtrees via `collect_kernel_inputs`
  - Falls back to interpreted `ScheduleItem::Reduce` when fusion fails

- [ ] Retire interpreted `ScheduleItem::Shape` and `ScheduleItem::Reduce`
  (kept as fallback paths for now)

---

## Phase 4 - GPT-2 Inference Bucketing

We accept padding/bucketing.

- [ ] Prefill:
  - Bucket `seq_len` (powers of two or a small fixed set).
  - Pad + mask so shapes become stable and `plan_cache` hits are high.

- [ ] Decode:
  - Prefer a dedicated "one token" step plan.
  - Avoid recompiling per token; treat cache position as data/offset math.

---

## Phase 5 - Backend Abstraction (CPU now, GPU later)

Goal: make `ExecutionPlan` replay targetable to different backends.

- [ ] Introduce a backend trait:
  - compile lowered kernels
  - execute kernels

- [ ] CPU backend: keep Cranelift.
- [ ] GPU backend later:
  - compile kernels to device code
  - optional graph batching (command buffers / CUDA graphs equivalent)

---

## Work Order

1) Phase 1 (Graph JIT plan cache)
2) Phase 2 (Buffer pool + memory planning)
3) Phase 3 (Fuse shape+reduce into kernels)
4) Phase 4 (Inference bucketing)
5) Phase 5 (GPU backend scaffolding)

## Possible Refactors

- &Context in Tensor. Tensor has the lifetime of the context right.
