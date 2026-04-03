# Venum IR & Scheduling — Implementation Plan

## Current State

Venum today is a **concrete-shape, single-backend (CPU/Cranelift), hardcoded-fusion** tensor compiler:
- **Graph IR**: `Op` enum with concrete `Vec<usize>` shapes, `Sub`/`Div` as first-class ops, separate reduce variants (`Sum`, `Prod`, `Max`, `Min`)
- **Fusion**: Greedy policy-based inlining (`DefaultFusionPolicy`) — no search
- **Codegen**: Cranelift JIT emitting flat elementwise loops (1 output element per iteration), no loop nests / tiling / vectorization
- **Optimization**: egglog e-graph pass (algebraic simplification) — already aligned with the spec
- **No LLIR**: No loop nest representation, no affine expressions, no schedule transforms

The spec requires a **two-IR architecture** (HLIR → ScheduleDecision → LLIR → codegen) with symbolic shapes, search-based scheduling, polyhedral dependence analysis, and multi-backend codegen.

---

## Phase 1 — HLIR Type Foundation

**Goal**: Replace the current `Graph`/`Node`/`Op` with the spec's HLIR types. Everything downstream breaks; that's expected.

**Tasks**:
1. **`hlir/dim.rs`** — `Dim` enum (`Const`, `Sym`, `Add`, `Mul`, `Div`, `Mod`), `Symbol` type (interned string or u32 index)
2. **`hlir/types.rs`** — `DType` (expand to F16, BF16, I8, I16, U8, U16, U32, U64, Bool), `Layout` enum (`Contiguous`, `Strided(Vec<Dim>)`, `View`), `TensorType`, `Scalar` (expand variants), `BufferId`/`NodeId` as newtypes
3. **`hlir/op.rs`** — New `Op` enum per spec §4.2 (drop `Sub`/`Div`/`Transpose`/`Squeeze`/`Unsqueeze`/`Pad`/`Flip`; add `Recip`, `Sin`, `Cos`, `Cast`, `Max`/`Min` binary, `Cmp`, `Where`, `Concat`; unify reductions into `Reduce { input, axes, op, keepdim }`)
4. **`hlir/graph.rs`** — `HLIRGraph` (arena of `HLIRNode`s), topological iteration, builder API
5. **`hlir/decompose.rs`** — Decomposition helpers: `sub(a,b) → add(a, neg(b))`, `div(a,b) → mul(a, recip(b))`, `matmul → reduce+mul+expand`
6. **Update `tensor/` API** — The user-facing `Tensor` builds an `HLIRGraph` instead of the old `Graph`. Derived ops (`sub`, `div`, `matmul`, `softmax`, `layernorm`) call decomposition helpers
7. **Update egglog pass** — Adapt to new `Op` variants (mostly rename mapping)
8. **Delete** old `graph/`, `shape_tracker.rs` (ShapeTracker is subsumed by `Layout`)

**Tests**: HLIR graph construction, decomposition correctness, round-trip through e-graph optimization.

---

## Phase 2 — Schedule Types & Trivial Search

**Goal**: Define `ScheduleDecision`, `Opt`, `FusionGroup`, and a trivial "no-op" searcher that puts each HLIR node in its own `FusionGroup` with zero opts.

**Tasks**:
1. **`schedule/opt.rs`** — `Opt`, `OptOp` enums per spec §5.1
2. **`schedule/decision.rs`** — `ScheduleDecision`, `FusionGroup`, `FusionGroupId`, `FusionTopology`
3. **`schedule/search.rs`** — `ScheduleSearcher` struct, `HardwareModel` trait (§10), `KernelContext`, `CostEstimate`, `BackendClass`. Trivial impl: one node per group, empty opt list, cost = 0
4. **`schedule/fusion.rs`** — Fusion legality checks per spec §5.3
5. **Delete** old `schedule/`, `fusion_policy.rs`, `pass/fusion.rs`

**Tests**: Trivial search on simple graphs produces valid `ScheduleDecision`. Fusion legality rejects cycles.

---

## Phase 3 — LLIR & Basic Lowering

**Goal**: Build the LLIR representation and a lowerer that converts HLIR + trivial ScheduleDecision → sequential LLIR loop nests (no opts applied yet).

**Tasks**:
1. **`llir/affine.rs`** — `AffineExpr`, `Var`, arithmetic helpers
2. **`llir/loop_nest.rs`** — `Loop`, `LoopNest`, `LoopKind` (Sequential only initially), `LoopAnnotations`, `ReductionAccumulator`
3. **`llir/memory.rs`** — `MemoryAccess`, `BufferAlloc`, `MemorySpace`, `AccessKind`
4. **`llir/stmt.rs`** — `Stmt`, `Expr`, `AbstractVectorOp` enums
5. **`llir/program.rs`** — `LLIRProgram`, `Kernel`, `KernelId`
6. **`lower/mod.rs`** — `lower(hlir, decision, dep) → Result<LLIRProgram>`:
   - `shape_to_domain`: tensor shape → iteration domain
   - `strides_to_access`: strides → `AccessMap`
   - Base loop nest emission (one `Sequential` loop per output dimension + reduction axes)
   - Single-node fusion group lowering (no merging yet)
7. **`llir/dependence.rs`** — `Dependence`, `DependenceRelation`, `AffineConstraint`, `DepKind` types
8. **`traits.rs`** — `DependenceAnalyzer` trait (§9.1), `CodeGenerator` trait (§9.3), `ScheduleTransform` enum (§9.2)

**Tests**: Lower `Add(Load, Load)` → 1-loop sequential nest. Lower `Reduce(Sum, ...)` → 2-loop nest with inner Reduce kind. Verify affine expression arithmetic.

---

## Phase 4 — Opt Application & Loop Transforms

**Goal**: The lowerer applies `Opt` sequences from `ScheduleDecision`, producing transformed loop nests.

**Tasks**:
1. **`lower/apply_opt.rs`** — Per-opt lowering:
   - `Tile`: split loop, insert inner, emit epilogue
   - `Vectorize`: mark loop kind, lift scalar ops to `AbstractVectorOp`
   - `Unroll`: mark loop kind
   - `Parallelize`: split + mark outer as Parallel
   - `GroupReduce`: shared buffer alloc, parallel/serial split, barrier insertion
   - `PadTo`: extend bound, wrap in `Stmt::If`
2. **`lower/axis.rs`** — Axis coordinate tracking (§5.1.1): maintain index shifts after each opt
3. **`lower/legality.rs`** — Incremental legality checking per §5.1.2 (stub `DependenceAnalyzer` that always returns legal for now)
4. **`lower/fusion_merge.rs`** — Fused nest merging per §8 step 4: `Chain` (inline producer), `FanIn`, `FanOut`, `DAG`

**Gap**:
Fusion is represented more richly than it is realized. The spec spends real effort on Chain, FanIn, FanOut, and DAG lowering in specs/04/SPEC.md, and the types exist in src/core/schedule/decision.rs, but the lowerer only accepts Chain. So the shape of the design is there, but the implemented fusion model is still “single node or simple linear chain.”

**Tests**: Matmul worked example from §11.1 (tile M, N, K → parallelize → vectorize). Verify axis indices shift correctly. GroupReduce produces barrier.

---

## Phase 5 — Polyhedral Dependence Analysis

**Goal**: Real dependence analysis backing legality checks.

**Tasks**:
1. **`poly/domain.rs`** — `Domain`, `Aff`, `PolyVar`, `Constraint` types per §7.1
2. **`poly/access_map.rs`** — `AccessMap`, construction from `MemoryAccess`
3. **`poly/isl.rs`** — ISL FFI bindings (§7.3): `isl_ctx_alloc/free`, `isl_union_map_*`. `IslDependenceAnalyzer` implementing `DependenceAnalyzer`
4. **`poly/native.rs`** — Pure-Rust fallback for simple cases (1D/2D affine without ISL)
5. **Wire into lowerer** — Replace stub analyzer with real implementation

**Tests**: Dependence computation on tiled matmul. Legality rejection for illegal interchange.

**Note**: ISL is optional. Start with the native fallback; add ISL behind a feature flag.

---

## Phase 6 — Schedule Search (Beam Search)

**Goal**: Replace trivial searcher with beam search over fusion groups × opt sequences.

**Tasks**:
1. **`schedule/candidates.rs`** — `opt_candidates` implementation: enumerate valid tile sizes, vectorize widths, parallelize amounts for each axis based on `KernelContext`
2. **`schedule/beam.rs`** — Beam search: expand candidates, score with `HardwareModel::estimate_cost`, prune to beam width
3. **`cost/roofline.rs`** — Roofline cost model: compute cycles, memory cycles, arithmetic intensity, bottleneck classification
4. **`cost/hardware.rs`** — `CpuHardwareModel` with cache sizes, SIMD widths, core counts

**Tests**: Search on matmul produces tiled schedule. Search on elementwise chain discovers fusion. Cost ordering: tiled < untiled.

---

## Phase 7 — CPU Codegen from LLIR

**Goal**: Replace the old Cranelift flat-loop codegen with one that lowers LLIR loop nests.

**Tasks**:
1. **`codegen/cpu/lower_llir.rs`** — Walk `LLIRProgram`, emit Cranelift IR for each `Kernel`:
   - Sequential loops → Cranelift basic block loops
   - Vectorized loops → platform SIMD intrinsics (AVX2/NEON via `CpuCodeGen::lower_vector_op`)
   - Unrolled loops → replicated bodies
   - Parallel loops → thread launch (rayon or std::thread)
   - Reduce loops → accumulator init/finalize
   - Barrier → no-op on CPU (single address space)
2. **`codegen/cpu/vector_ops.rs`** — `AbstractVectorOp` → x86 intrinsic string mapping
3. **Delete** old `codegen/emit.rs`, `codegen/expr.rs`, `codegen/reduce.rs`
4. **Update `compile.rs`** — Wire the new pipeline: `HLIRGraph → optimize_hlir → search → lower → optimize_llir → codegen`

**Tests**: End-to-end: `compile` a matmul, execute, verify numerical output. Elementwise fusion. Reduce correctness.

---

## Phase 8 — LLIR Optimization & Schedule Transforms

**Goal**: Post-lowering LLIR optimization (interchange, cache read/write).

**Tasks**:
1. **`llir/optimize.rs`** — `optimize_llir(program, dep) → program`: apply `ScheduleTransform`s (interchange for locality, cache read/write insertion), check legality via `DependenceAnalyzer`
2. **Cross-kernel dependences** — `analyze_kernel` on pairs sharing buffers

**Tests**: Interchange on matmul improves cache access pattern (verify via cost model). Cache read on tiled matmul inserts buffer alloc.

---

## Phase 9 — GPU Backend (stretch)

**Goal**: WGSL or CUDA codegen from the same LLIR.

**Tasks**:
1. **`codegen/gpu/wgsl.rs`** — `GpuCodeGen` impl for WGSL: Parallel loops → workgroup dispatch, Vectorized → vec4, Barrier → `workgroupBarrier()`, shared memory → `var<workgroup>`
2. **`cost/gpu_hardware.rs`** — GPU hardware model
3. **`GpuBackend`** — Kernel launch, buffer management

---

## Dependency Graph

```
Phase 1 (HLIR types)
  └→ Phase 2 (Schedule types)
       └→ Phase 3 (LLIR + basic lowering)
            ├→ Phase 4 (Opt application)
            │    └→ Phase 5 (Polyhedral analysis)
            │         └→ Phase 6 (Beam search)
            └→ Phase 7 (CPU codegen from LLIR)
                 └→ Phase 8 (LLIR optimization)
                      └→ Phase 9 (GPU backend)
```

Phases 4-6 and Phase 7 can be developed in parallel once Phase 3 is done. The critical path is **1 → 2 → 3 → 7** to get end-to-end execution working again, then **4 → 5 → 6** to add the search-based scheduling that is the spec's core contribution.
