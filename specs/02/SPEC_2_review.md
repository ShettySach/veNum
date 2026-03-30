Solid v2. Here's a thorough analysis — strengths first, then the real concerns, roughly ordered by severity.

---

## What's Strong

**Three-tier separation is the right call.** The key insight — that Plan IR is where search happens, not HLIR or LLIR — is architecturally sound. HLIR stays semantically pure, LLIR stays committed and analyzable, and Plan IR is the combinatorial search space. This mirrors what Halide does with its schedule/algorithm split but more explicitly.

**Primitive set is well-chosen.** ~15 ops covering the tinygrad-style compute basis. The derived ops table (Matmul, Softmax, etc. as subgraphs) is exactly right for the bitter lesson philosophy — you don't lose the ability to recognize them, you just push recognition to the e-graph layer.

**Symbol normalization (§4.3) is an underrated detail.** The `Reshape([N, M] → [N*M])` → introduce `P_NM` trick correctly handles the HLIR-to-LLIR affinity boundary. Non-affine `Dim::Mul(Sym, Sym)` at HLIR gets flattened into a new affine parameter. Good.

**Provenance threading through LLIR** (`Kernel::provenance: SemanticTag`) is important for cost modeling and backend hints. Keeping it all the way to codegen is the right call.

---

## Issues

### 1. E-graph at Plan IR is underspecified — and may be the wrong tool

The rules given are pseudocode:
```
fuse(region_a, region_b) ↔ region_ab
    if producer_consumer(region_a, region_b) ∧ fusable(region_a, region_b)
```

The problem: e-graphs work on *expressions in a term algebra*. A `PlannedRegion` is a struct with a schedule, not an expression tree. To use an e-graph here, you'd need to either (a) encode schedules as expressions (like Tensat does with relay/operator graphs), or (b) use something more like equality saturation over a DAG of scheduling decisions, which is closer to what Halide's autoscheduler or TVM's Ansor does with sketch-based search. The spec conflates "e-graph for algebraic rewrites" (totally valid at HLIR) with "e-graph for schedule space exploration" (a different and harder problem). Worth deciding if Plan IR search is actually e-graph equality saturation or beam/evolutionary search over a parameterized schedule space.

### 2. `Specialization::Search` at Plan IR punts on algorithmic rewrites

The Softmax lowering example defers to `Specialization::Search` with the comment *"Let LLIR search find online algorithm."* But online softmax (the Flash Attention single-pass trick) isn't a loop transformation — it's an *algorithmic restructuring* that changes what state you maintain across iterations. No sequence of `ScheduleTransform` variants (tile, interchange, vectorize, etc.) can derive it from the naive multi-pass HLIR decomposition. If you want to discover online softmax from primitives, the rewrite has to happen at HLIR or Plan IR level, not LLIR. Either add the rewrite rule explicitly (`Softmax` region → online single-pass kernel) or acknowledge this as a known gap.

### 3. `fused_with: Option<RegionId>` can't represent multi-way fusion

```rust
pub struct PlannedRegion {
    pub fused_with: Option<RegionId>,
    ...
}
```

A → B → C fused is a 3-way fusion group. This field only lets you fuse with one other region, and there's no grouping mechanism. Either use a `fusion_group: Option<FusionGroupId>` with a separate `fusion_groups: HashMap<FusionGroupId, Vec<RegionId>>`, or a union-find. For softmax (5 nodes), elementwise chains, or transformer attention blocks, you regularly need ≥3-way fusion.

### 4. `Loop::step: i64` vs. `TileSize::SearchParam(String)`

Plan IR tile sizes can be `SearchParam("tile_m")` — unresolved. But LLIR has `step: i64` which is concrete. The resolution step (search → concrete tile sizes → applied to LLIR) isn't described anywhere. What produces the concrete LLIR from a Plan IR with unresolved search params? This is a pipeline gap that needs an explicit pass: *parameter binding* or *search result application*.

### 5. `Intrinsic` in LLIR `Expr` breaks backend-agnosticism

```rust
Intrinsic { name: String, args: Vec<Expr> },
```

An `Intrinsic` with a name string is by definition backend-specific (e.g., `_mm256_fmadd_ps`). Putting it in LLIR breaks the stated goal of LLIR being "backend-agnostic but hardware-mappable." Better to model abstract SIMD operations (FMA, horizontal add, etc.) as LLIR-level concepts that backends lower to concrete intrinsics. Or push intrinsics entirely into codegen, with LLIR only having annotated vectorized loops.

### 6. `Reduction` loop kind has `init: Scalar` — no mixed reductions

```rust
Reduction { op: ReduceOp, init: Scalar }
```

A loop can only be one kind of reduction. For fused kernels that maintain multiple accumulators (e.g., running max and running sum simultaneously in the online softmax derivation, or argmax needing both value and index), this is insufficient. You'd need either multiple parallel reduction annotations or a different accumulator representation.

### 7. `Dim::Div` and `Dim::Mod` have unclear lowering semantics

You allow `Div` and `Mod` over symbolic dims at HLIR. `Div(Sym, Sym)` is non-affine. The symbol normalization section only covers the Reshape case. `Slice` with dynamic ranges, strided views, and tiled access patterns all potentially produce these. Need a complete account of which `Dim` expressions are normalizable to affine params vs. which block lowering.

---

## Minor

- `SemanticTag::Attention { seq_axis }` is listed but Flash Attention tiling requires a tiled matmul + online softmax joint rewrite — it's not clear how this tag guides lowering beyond hinting at the pattern.
- `BufferAlloc::shape: Vec<AffineExpr>` but shape was `Vec<Dim>` in HLIR — the conversion is implicit; should be explicit in the lowering pass description.
- `Dependence::from/to` is `(KernelId, StmtId)` but `StmtId` isn't defined anywhere in the spec.

---

## Summary

The HLIR and LLIR layers are well-specified. The weak points are concentrated at Plan IR: the e-graph search mechanism is conceptually fuzzy, multi-way fusion is unrepresentable, and unresolved search parameters have no explicit resolution pass. The online RL component seems to have been dropped — worth deciding if it's deferred or removed. The `Specialization::Search` escape hatch is doing too much work and needs to be replaced with explicit algorithmic rewrite rules for at least the known patterns (online softmax, blocked matmul).
