# Venum Implementation Plan — SPEC\_3 Migration

## Architecture Decisions

### AOT with Symbolic Shapes

Venum is an **ahead-of-time** compiler. `compile()` produces a `CompiledProgram` that is executed repeatedly with different inputs. Today, shapes are concrete `usize` — any shape change requires recompilation.

SPEC\_3 changes this: shapes like `batch_size`, `seq_len`, `N`, `M`, `K` become `RuntimeParameter` symbols. They flow through the compiler as `Dim::Sym("seq_len")` at HLIR, remain symbolic through Plan IR and LLIR, and appear as `Var::Param("seq_len")` in `AffineExpr` loop bounds. The compiled kernel receives them as scalar arguments at dispatch time. **No recompilation on shape change**, provided the chosen tile sizes remain valid (epilogues handle remainders).

Tile sizes are fixed at **tune-time** (after schedule search, before codegen). Shared memory sizes must also be statically known. Only loop iteration counts and buffer sizes carry runtime symbols.

### Global Schedule Search

The schedule search is **global over the whole program**, not per-kernel. The `ScheduleSpace` enumerates all valid combinations of:

- Which regions to fuse into which `FusionGroup`s
- What tile sizes to use for each dimension
- What parallelism strategy for each group

Beam search explores this joint space, producing ranked candidate `PlanGraph`s (each a complete program schedule). The `HardwareModel` cost estimator scores each candidate. This is global search in the style of Luminal/TVM AutoScheduler — the search sees the full dataflow graph and makes globally-informed fusion/tiling decisions, not per-op local choices.

The search is **not** e-graph saturation. E-graphs are used only at HLIR for algebraic rewriting and pattern recognition. Plan IR search is combinatorial beam search over a schedule space.

---

## Guiding Principles

1. **Never break existing tests.** Every phase ends with all 39 tests passing. New IR layers are built alongside the old pipeline, swapped in at the end.
2. **No forward stubs.** Types are defined together with everything they reference. No placeholder `AffineExpr` in Phase 0 that doesn't exist until Phase 2.
3. **Dual-path transition.** Old `Graph → FusedKernel → Cranelift` stays alive until the new path is proven equivalent via differential testing. Then deleted.

---

## Phase 1 — IR Data Structures

**Goal:** Define all three IR tiers (HLIR, Plan IR, LLIR) as pure data structures in a new `src/core/ir/` module tree. No passes, no lowering, no changes to existing code. Just types and unit tests.

Everything goes in at once so there are no forward-reference stubs.

### 1.1 Foundation: `Dim`, `Symbol`, `AffineExpr`

`src/core/ir/dim.rs` — Symbolic dimension expressions (HLIR-level):

```rust
pub type Symbol = Arc<str>;   // Interned for cheap equality + hashing

pub enum Dim {
    Const(i64),
    Sym(Symbol),
    Add(Box<Dim>, Box<Dim>),
    Mul(Box<Dim>, Box<Dim>),
    Div(Box<Dim>, Box<Dim>),
    Mod(Box<Dim>, Box<Dim>),
}
```

Methods: `is_const()`, `as_const()`, `is_affine()` (classification from §3.3), `Display`.

`src/core/ir/affine.rs` — Affine expressions (LLIR-level):

```rust
pub struct AffineExpr {
    pub constant: i64,
    pub terms: Vec<(i64, Var)>,
}

pub enum Var {
    Loop(LoopVar),
    Param(Symbol),
}
```

Methods: `constant()`, `var()`, `add()`, `scale()`, `substitute()`, `evaluate()`.

`src/core/ir/symbol.rs` — Symbol resolution and binding:

```rust
pub enum SymbolResolution {
    CompileTime(i64),
    TuneTime(i64),
    RuntimeParameter,
    RuntimeDerived { expr: AffineExpr },
}

pub struct SymbolBindingTable {
    pub resolutions: HashMap<Symbol, SymbolResolution>,
    pub tune_time_bindings: HashMap<String, i64>,
}
```

`src/core/ir/normalize.rs` — `Dim` → `AffineExpr` normalization (§3.3):

```rust
pub fn normalize_dim(dim: &Dim, bindings: &mut SymbolBindingTable) -> AffineExpr;
```

Tests: all cases from §3.3 validity classification.

### 1.2 Shared Types

`src/core/ir/types.rs`:

```rust
pub struct TensorType {
    pub shape: Vec<Dim>,
    pub dtype: DType,
    pub layout: Layout,
}

pub enum Layout {
    Contiguous,
    Strided(Vec<Dim>),
    View { base: TensorId, offset: Dim, strides: Vec<Dim> },
}

pub enum ReduceOp { Sum, Prod, Max, Min }
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }
pub enum UnaryOp { Neg, Recip, Exp, Log, Sqrt, Sin, Cos }
pub enum BinaryOp { Add, Mul, Max, Min }

// ID newtypes
pub struct NodeId(pub usize);
pub struct TensorId(pub usize);
pub struct BufferId(pub usize);
pub struct RegionId(pub usize);
pub struct FusionGroupId(pub usize);
pub struct KernelId(pub usize);
```

Note: `ir::NodeId` is separate from the existing `graph::NodeId`. They coexist until Phase 5.

### 1.3 HLIR

`src/core/ir/hlir.rs`:

```rust
pub enum Op {
    Const { value: Scalar, shape: Vec<Dim>, dtype: DType },
    Load { buffer: BufferId },
    Store { buffer: BufferId, value: NodeId },
    Neg(NodeId), Recip(NodeId), Exp(NodeId), Log(NodeId), Sqrt(NodeId),
    Sin(NodeId), Cos(NodeId),
    Cast { input: NodeId, to: DType },
    Add(NodeId, NodeId), Mul(NodeId, NodeId),
    Max(NodeId, NodeId), Min(NodeId, NodeId),
    Cmp { op: CmpOp, lhs: NodeId, rhs: NodeId },
    Where { cond: NodeId, then_val: NodeId, else_val: NodeId },
    Reduce { input: NodeId, axes: Vec<usize>, op: ReduceOp, keepdim: bool },
    Reshape { input: NodeId, shape: Vec<Dim> },
    Permute { input: NodeId, axes: Vec<usize> },
    Slice { input: NodeId, ranges: Vec<Range> },
    Expand { input: NodeId, shape: Vec<Dim> },
    Concat { inputs: Vec<NodeId>, axis: usize },
}

pub struct HLIRGraph {
    pub nodes: Vec<HLIRNode>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub buffers: Vec<BufferDecl>,
    pub symbols: HashSet<Symbol>,
}

pub struct HLIRNode {
    pub id: NodeId,
    pub op: Op,
    pub ty: TensorType,
    pub region: Option<RegionId>,
}

pub struct SemanticRegion {
    pub id: RegionId,
    pub nodes: Vec<NodeId>,
    pub tag: SemanticTag,
    pub metadata: RegionMetadata,
}

pub enum SemanticTag {
    Elementwise,
    Contraction { m: Dim, n: Dim, k: Dim },
    Reduction { axes: Vec<usize> },
    Softmax { axis: usize },
    Normalization { axes: Vec<usize> },
    Attention { seq_axis: usize, head_dim: Dim },
    Generic,
}
```

No `Sub`/`Div` — decomposed at construction: `Sub(a,b)` → `Add(a, Neg(b))`, `Div(a,b)` → `Mul(a, Recip(b))`.

`AnnotatedHLIR` — output of HLIR optimization + pattern recognition, input to schedule search:

```rust
pub struct AnnotatedHLIR {
    pub graph: HLIRGraph,
    pub regions: Vec<SemanticRegion>,
}
```

### 1.4 Plan IR

`src/core/ir/plan.rs`:

```rust
pub struct PlanGraph {
    pub regions: Vec<PlannedRegion>,
    pub fusion_groups: Vec<FusionGroup>,
    pub execution_order: Vec<FusionGroupId>,
    pub materializations: HashMap<NodeId, MaterializationChoice>,
    pub symbol_bindings: SymbolBindingTable,
}

pub struct PlannedRegion {
    pub id: RegionId,
    pub semantic: SemanticTag,
    pub nodes: Vec<NodeId>,
    pub schedule: RegionSchedule,
    pub fusion_group: Option<FusionGroupId>,
    pub algorithmic_rewrite: Option<AlgorithmicRewrite>,
}

pub struct RegionSchedule {
    pub tiling: Option<TilingSpec>,
    pub parallelism: ParallelismSpec,
    pub memory_placement: MemoryPlacement,
    pub specialization: Specialization,
}

pub struct TilingSpec {
    pub tile_sizes: Vec<TileSize>,
    pub tile_order: Vec<usize>,
}

pub enum TileSize { Const(i64), SearchParam(String) }

pub enum ParallelismSpec {
    Sequential,
    Parallel { axis: usize, num_threads: ParamOrConst },
    GPU { grid: Vec<Dim>, block: Vec<Dim> },
}

pub struct FusionGroup {
    pub id: FusionGroupId,
    pub regions: Vec<RegionId>,
    pub topology: FusionTopology,
}

pub enum FusionTopology {
    Chain,
    FanIn { consumer: RegionId, producers: Vec<RegionId> },
    FanOut { producer: RegionId, consumers: Vec<RegionId> },
    DAG { edges: Vec<(RegionId, RegionId)> },
}

pub enum MaterializationChoice {
    Fused,
    Materialized { buffer: BufferId },
    Alias { base: BufferId, offset: Dim },
}

pub enum AlgorithmicRewrite {
    OnlineSoftmax { axis: usize },
    BlockedMatmul { block_m: i64, block_n: i64, block_k: i64 },
    BlockedAttention { block_q: i64, block_k: i64 },
    WelfordNormalization,
}

pub enum RewriteSpec {
    OnlineSoftmax { axis: usize, tile_size: i64 },
    BlockedMatmul { block_m: i64, block_n: i64, block_k: i64 },
    BlockedAttention { block_q: i64, block_k: i64 },
    WelfordNormalization { axes: Vec<usize> },
}

pub struct TunedPlan {
    pub plan: PlanGraph,
    pub tile_bindings: HashMap<String, i64>,
    pub divisibility: HashMap<(String, Symbol), Divisibility>,
}

pub enum Divisibility { AlwaysDivisible, RequiresEpilogue }
```

### 1.5 LLIR

`src/core/ir/llir.rs`:

```rust
pub struct LLIRProgram {
    pub kernels: Vec<Kernel>,
    pub buffers: Vec<BufferAlloc>,
    pub dependencies: Vec<Dependence>,
}

pub struct Kernel {
    pub id: KernelId,
    pub name: String,
    pub params: Vec<KernelParam>,
    pub body: LoopNest,
    pub reads: Vec<MemoryAccess>,
    pub writes: Vec<MemoryAccess>,
    pub provenance: SemanticTag,
    pub rewrite: Option<RewriteSpec>,
}

pub struct LoopNest { pub loops: Vec<Loop>, pub body: Vec<Stmt> }

pub struct Loop {
    pub var: LoopVar,
    pub lower: AffineExpr,
    pub upper: AffineExpr,
    pub step: i64,
    pub kind: LoopKind,
    pub annotations: LoopAnnotations,
}

pub enum LoopKind {
    Sequential,
    Parallel,
    Vectorized { width: usize },
    Unrolled { factor: usize },
    GridDim { axis: usize },
    BlockDim { axis: usize },
    Reduction { accumulators: Vec<ReductionAccumulator> },
}

pub struct ReductionAccumulator {
    pub var: String,
    pub op: ReduceOp,
    pub init: Scalar,
    pub dtype: DType,
}

pub enum Stmt {
    Assign { dst: MemoryAccess, src: Expr },
    Accumulate { dst: MemoryAccess, op: ReduceOp, src: Expr },
    If { cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt> },
    Loop(Loop, Vec<Stmt>),
    Barrier { scope: BarrierScope },
    Epilogue { main_loop_var: LoopVar, remainder_body: Vec<Stmt> },
    Nop,
}

pub enum Expr {
    Literal(Scalar),
    Load(MemoryAccess),
    Unary { op: UnaryOp, arg: Box<Expr> },
    Binary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Ternary { cond: Box<Expr>, then_val: Box<Expr>, else_val: Box<Expr> },
    Cast { arg: Box<Expr>, to: DType },
    AbstractVector(AbstractVectorOp),
}

pub enum AbstractVectorOp {
    Fma { acc: Box<Expr>, lhs: Box<Expr>, rhs: Box<Expr>, width: usize },
    HorizontalReduce { op: ReduceOp, arg: Box<Expr>, width: usize },
    Broadcast { scalar: Box<Expr>, width: usize },
    Gather { base: BufferId, indices: Box<Expr>, width: usize },
    Scatter { base: BufferId, indices: Box<Expr>, value: Box<Expr>, width: usize },
    VecBinary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr>, width: usize },
    VecCast { arg: Box<Expr>, from: DType, to: DType, width: usize },
}

pub struct MemoryAccess {
    pub buffer: BufferId,
    pub indices: Vec<AffineExpr>,
    pub access_kind: AccessKind,
}

pub enum AccessKind { Read, Write, ReadWrite }

pub struct BufferAlloc {
    pub id: BufferId,
    pub shape: Vec<AffineExpr>,
    pub dtype: DType,
    pub memory_space: MemorySpace,
}

pub enum MemorySpace { Global, Shared, Local, Constant }
pub enum BarrierScope { Workgroup, Subgroup, Device }

// Schedule transforms (applied post-lowering, legality-checked)
pub enum ScheduleTransform {
    Tile { loop_var: LoopVar, factor: i64 },
    Interchange { outer: LoopVar, inner: LoopVar },
    Fuse { loop_a: LoopVar, loop_b: LoopVar },
    Parallelize { loop_var: LoopVar, kind: ParallelKind },
    Unroll { loop_var: LoopVar, factor: usize },
    Vectorize { loop_var: LoopVar, width: usize },
    ComputeAt { producer: KernelId, consumer: KernelId, loop_var: LoopVar },
    CacheRead { buffer: BufferId, at_loop: LoopVar, memory: MemorySpace },
    CacheWrite { buffer: BufferId, at_loop: LoopVar, memory: MemorySpace },
}

pub enum ParallelKind { Thread, SIMD, GPU { dim: usize } }

// Dependence analysis types
pub struct StmtId { pub kernel: KernelId, pub index: usize }

pub struct Dependence {
    pub from: StmtId,
    pub to: StmtId,
    pub kind: DepKind,
    pub distance: Option<Vec<i64>>,
    pub relation: DependenceRelation,
}

pub enum DepKind { RAW, WAR, WAW }
```

### 1.6 LLIR Pretty-Printer

`fn display_llir(program: &LLIRProgram) -> String` — pseudo-code output for debugging loop structures. Essential for every subsequent phase.

### Deliverable

- New `src/core/ir/` module tree compiles.
- Unit tests on `Dim` arithmetic/classification, `AffineExpr` arithmetic, `normalize_dim`.
- No existing code touched. `cargo test` passes.

---

## Phase 2 — Old Graph → HLIR Bridge + E-Graph Enhancement

**Goal:** Connect the existing user-facing `Graph` to the new HLIR, and extend e-graph to work on it.

### 2.1 Graph → HLIR Converter

`fn lower_graph_to_hlir(graph: &Graph) -> HLIRGraph`:

- Maps `usize` shapes → `Dim::Const`.
- Decomposes `graph::Op::Sub` → `hlir::Op::Add(a, hlir::Op::Neg(b))` (inserts new node).
- Decomposes `graph::Op::Div` → `hlir::Op::Mul(a, hlir::Op::Recip(b))`.
- Maps `graph::Op::Ln` → `hlir::Op::Log`.
- Wraps reduce ops: `Op::Sum(dims, kd)` → `hlir::Op::Reduce { op: Sum, axes: dims, keepdim: kd }`.

This is the entry point for the new pipeline. The old pipeline's `graph::Graph` is still the user-facing API; this converter is internal.

### 2.2 Migrate Egglog to HLIR

Rewrite `optimize/egglog_program.rs` to accept `HLIRGraph`:

- Handle `Recip` (new), `Log` (renamed from `Ln`).
- New rewrite rules: `Recip(Recip(x)) → x`, `Mul(x, Recip(x)) → Const(1)`.
- Unmodeled ops (Slice, Pad, Reduce, Cmp, Where) remain opaque leaves.
- Same extraction + rebuild roundtrip.

Regression test: run old egglog bridge and new egglog bridge on same input graphs, confirm identical simplification results.

### 2.3 Pattern Recognition Rules

Add egglog rules that **annotate** (not rewrite) recognized patterns:

- `Reduce(Sum, Mul(Expand(_), Expand(_)))` → `Contraction` tag.
- `Exp(Add(x, Neg(Reduce(Max, x))))` chain → `Softmax` tag.
- Mean + Variance over same input → `Normalization` tag.

Output: `Vec<SemanticRegion>` extracted after saturation, attached to `HLIRNode::region` fields.

### 2.4 EGraphConfig

Replace hardcoded `10` iterations with configurable `EGraphConfig { max_nodes, max_iterations, timeout, extraction }` (§9.2). Default: 100K nodes, 30 iterations, 10s timeout.

### Deliverable

- `Graph → HLIRGraph → egglog optimize → HLIRGraph` roundtrip produces same results as old pipeline.
- Pattern recognition tested: hand-built matmul decomposition gets `Contraction` tag, softmax decomposition gets `Softmax` tag.
- Old pipeline still default.

---

## Phase 3 — HLIR → Plan IR → LLIR Pipeline (Greedy Baseline)

**Goal:** First working end-to-end new pipeline: HLIR → Plan IR → LLIR. Uses greedy scheduling (no beam search), producing an `LLIRProgram` with explicit loop nests.

### 3.1 Region Extraction

`fn extract_regions(hlir: &HLIRGraph) -> Vec<SemanticRegion>`:

- Uses semantic tags from Phase 2 e-graph output.
- Untagged nodes default to `Elementwise` or `Generic`.
- Each region is a maximal connected subgraph of same-tag nodes.

### 3.2 Greedy Fusion + Default Schedule

`fn greedy_schedule(regions: &[SemanticRegion], hlir: &HLIRGraph) -> PlanGraph`:

- Fuse producer-consumer pairs where legality allows (no cycle, no split reduction).
- `FusionTopology::Chain` for linear sequences.
- Default `RegionSchedule`: no tiling, `Sequential` (or `Parallel` for outermost non-reduce), `MemoryPlacement::Default`, `Specialization::LoopNest`.
- No `SearchParam`s — all tile sizes are `Const` or absent.
- `bind_parameters` is a trivial pass-through (nothing to resolve).

### 3.3 Plan IR → LLIR Lowering

`fn lower_to_llir(tuned: &TunedPlan) -> Result<LLIRProgram>`:

For each `FusionGroup` in `execution_order`:
1. Determine iteration domain from region shapes.
2. Generate outer loops (non-reduce dims) + inner reduction loops.
3. If `TilingSpec` present, split loops into outer tile / inner point loops.
4. If `Divisibility::RequiresEpilogue`, emit `Stmt::Epilogue`.
5. Convert `Dim` shapes → `AffineExpr` via `normalize_dim`.
6. Elementwise fused nodes → `Stmt::Assign { dst, src: nested Expr tree }`.
7. Reductions → `Stmt::Accumulate` with `ReductionAccumulator`.
8. Generate `BufferAlloc` for intermediates; validate shared memory has no runtime params.

### 3.4 Verify via Pretty-Print

Use the LLIR pretty-printer from Phase 1 to verify loop structures:
- `exp(x)` → single sequential loop, one load, one exp, one store.
- `a + b` → single loop, two loads, one add, one store.
- `reduce_sum(x, axis=1)` → outer loop over non-reduced dims, inner reduction loop with `Sum` accumulator.

### Deliverable

- `HLIRGraph → PlanGraph → TunedPlan → LLIRProgram` tested end-to-end.
- LLIR pretty-printed output matches expected loop structures.
- Still no execution — old pipeline handles that.

---

## Phase 4 — LLIR → Cranelift Backend + Cutover

**Goal:** Lower LLIR to executable Cranelift IR, prove correctness via differential testing, then delete the old pipeline.

### 4.1 CodegenBackend Trait

```rust
pub trait CodegenBackend: Send + Sync {
    fn lower_kernel(&self, kernel: &llir::Kernel) -> Result<Arc<dyn ExecutableKernel>>;
}
```

### 4.2 LLIR → Cranelift Lowering

New `src/core/codegen/cpu/llir_lower.rs`:

- Walk `LoopNest`: each `Loop` → Cranelift loop header/body/exit pattern (reuse the block structure from current `emit.rs`).
- `LoopKind::Sequential` → counted loop with `icmp + brif`.
- `LoopKind::Reduction` → loop with phi-node accumulator(s) (one block param per accumulator).
- `LoopKind::Parallel` → sequential for now (thread parallelism is Phase 6).
- `Stmt::Assign` → evaluate `src` Expr, store to `dst` address.
- `Stmt::Accumulate` → load acc, evaluate src, combine, store back.
- `Stmt::Epilogue` → conditional tail loop for remainder.
- `Expr::Load(MemoryAccess)` → compute affine address, Cranelift `load`.
- `Expr::Unary/Binary` → Cranelift `fadd/fmul/fneg/...` (reuse patterns from current `expr.rs`).
- `Expr::AbstractVector` → scalar fallback initially.
- `AffineExpr` → Cranelift `iconst + imul + iadd` chain for address computation.
- `RuntimeParameter` symbols → kernel function arguments (Cranelift `FuncParam`).

### 4.3 Dual-Path Integration

Modify `compile.rs` to run **both** pipelines and compare:

```
old: Graph → FusedKernel → old Cranelift → execute → results_old
new: Graph → HLIR → PlanIR → LLIR → new Cranelift → execute → results_new
assert_eq!(results_old, results_new)
```

Run this for all 39 existing tests.

### 4.4 Cutover: Delete Old Pipeline

Once differential testing passes for all tests:

- Delete `codegen/emit.rs`, `codegen/expr.rs`, `codegen/tracker.rs`, `codegen/reduce.rs`.
- Delete `schedule/fused_kernel.rs`, `schedule/topo.rs`.
- Delete `fusion_policy.rs`.
- `compile.rs` uses new pipeline only.
- Keep `exec/` interpreted paths as reference implementation and shape-op fallback.

### 4.5 Final Module Layout

```
src/core/
  ir/
    dim.rs, affine.rs, symbol.rs, normalize.rs  # Foundation
    types.rs                                     # Shared enums, ID newtypes
    hlir.rs                                      # HLIR graph, ops, semantic regions
    plan.rs                                      # Plan IR, fusion groups, schedules
    llir.rs                                      # LLIR program, kernels, loops, stmts
    display.rs                                   # Pretty-printer
  optimize/               # egglog (now on HLIR)
  lower/
    graph_to_hlir.rs      # Old Graph → HLIR bridge
    hlir_to_plan.rs       # Region extraction + greedy fusion
    plan_to_llir.rs       # Loop nest generation
  codegen/
    backend.rs            # CodegenBackend trait
    cpu/
      llir_lower.rs       # LLIR → Cranelift
      math.rs             # libm function refs
      ...
  pass/                   # Compiler pass manager
  program/                # CompiledProgram, execution
  dtype.rs
  tensor/                 # User-facing API (unchanged)
```

### Deliverable

- All 39 tests pass on new-only pipeline.
- Old codegen/schedule deleted.
- Codebase is strictly HLIR → Plan IR → LLIR → backend.

---

## Phase 5 — Global Schedule Search + Cost Model

**Goal:** Replace greedy fusion with beam search over the whole-program schedule space, guided by a hardware cost model.

### 5.1 HardwareModel Trait

```rust
pub trait HardwareModel: Send + Sync {
    fn cache_sizes(&self) -> &[usize];        // [L1, L2, L3] in bytes
    fn vector_width(&self, dtype: DType) -> usize;
    fn compute_throughput(&self, op: &str, dtype: DType) -> f64;  // ops/cycle
    fn memory_bandwidth(&self, level: usize) -> f64;              // bytes/cycle
}
```

Implement `CpuModel` with defaults (L1=32KB, L2=256KB, vector\_width=8 for F32).

### 5.2 Cost Estimator

`fn estimate_cost(plan: &PlanGraph, hw: &dyn HardwareModel) -> CostEstimate`:

- Count FLOPs per region (from semantic tag + shape).
- Estimate memory traffic: materialized buffers = read+write; fused = 0.
- Estimate cache pressure: tile working set vs. cache sizes.
- Roofline-style combine: `cost = max(compute_bound, memory_bound)`.

### 5.3 Schedule Space Enumeration

`fn build_schedule_space(hlir: &AnnotatedHLIR, hw: &dyn HardwareModel) -> ScheduleSpace`:

- Enumerate valid `FusionCandidate`s (check legality: no cycle, no split reduction, working set fits).
- Enumerate `tile_param_ranges` from hardware model: powers of 2 up to cache-fitting sizes.
- Enumerate `parallelism_options`.

### 5.4 Beam Search

```rust
pub struct ScheduleSearcher<H: HardwareModel> {
    pub hardware: H,
    pub config: SearchConfig,
}

impl<H: HardwareModel> ScheduleSearcher<H> {
    pub fn search(
        &self,
        annotated_hlir: &AnnotatedHLIR,
        space: &ScheduleSpace,
    ) -> Vec<(PlanGraph, CostEstimate)>;
}
```

`SearchConfig`: beam\_width (default 8), max\_iterations (50), timeout (30s), initial\_bindings.

The search is **global**: each beam candidate is a complete `PlanGraph` (all fusion groups + tile sizes + parallelism for the whole program). Neighbors are generated by mutating one scheduling decision at a time (fuse/unfuse a pair, change a tile size, switch parallelism).

### 5.5 `bind_parameters` (Real)

After search picks the best `PlanGraph`, resolve all `TileSize::SearchParam` to concrete `TileSize::Const`. Compute `Divisibility` for each (tile, loop\_bound) pair.

### Deliverable

- Beam search tested on matmul graph: produces tiled schedule.
- Cost model ranks tiled > untiled.
- Greedy baseline is still available as `beam_width=1` fallback.

---

## Phase 6 — Algorithmic Rewrites + Parallelism

**Goal:** Implement algorithmic rewrites and thread parallelism.

### 6.1 AlgorithmicRewrite Application

`fn apply_algorithmic_rewrites(plan: &mut PlanGraph)`:

For each `PlannedRegion` with an eligible `SemanticTag`:
- `Softmax` → try `OnlineSoftmax`: single-pass joint `(running_max, running_sum)` reduction. Produces `RewriteSpec::OnlineSoftmax`.
- `Contraction` → try `BlockedMatmul`: register-blocked tiled loop nest. Produces `RewriteSpec::BlockedMatmul`.
- `Normalization` → try `WelfordNormalization`: one-pass online mean/variance.

Each rewrite has `check_preconditions` (verify tag, verify fused, verify shapes) and `apply` (modify region + produce `RewriteSpec`). The LLIR lowerer in Phase 4 already handles `RewriteSpec` — it uses the spec to emit the specialized loop structure (e.g., `Vec<ReductionAccumulator>` for online softmax's joint accumulators).

Cost model decides whether to apply: estimate cost with and without rewrite, pick cheaper.

### 6.2 Thread Parallelism

- `LoopKind::Parallel` → emit Cranelift code that splits the iteration space across threads.
- Start with simple static partitioning (each thread gets `N/num_threads` iterations).
- Rayon or `std::thread::scope` for CPU backend.

### Deliverable

- Online softmax: correct results, LLIR shows single-pass joint accumulator.
- Blocked matmul: correct tiled loop nest.
- Parallel outer loop: speedup on multi-core for large elementwise ops.

---

## Phase 7 — Additional Backends (Future)

Not part of the initial migration. Listed for roadmap.

- **GPU (CUDA/Metal)**: `CodegenBackend` lowering `GridDim`/`BlockDim` loops to PTX or MSL. `MemorySpace::Shared` → `__shared__`.
- **WebGPU (WGSL)**: `CodegenBackend` emitting WGSL compute shaders.
- **SIMD**: Lower `AbstractVectorOp` to platform intrinsics (AVX2, NEON, SVE) instead of scalar fallback.
- **DType expansion**: Add `F16`, `BF16`, `Bool` to `DType`/`Scalar`/`Buffer` when a backend needs them.

---

## Dependency Graph

```
Phase 1 (All IR Data Structures)
    │
    ├── Phase 2 (Graph→HLIR Bridge + E-Graph)
    │       │
    │       └── Phase 3 (HLIR→Plan→LLIR Pipeline)
    │               │
    │               └── Phase 4 (LLIR→Cranelift + Cutover)
    │                       │
    │                       ├── Phase 5 (Global Search + Cost Model)
    │                       │
    │                       └── Phase 6 (Algorithmic Rewrites + Parallelism)
    │
    └── Phase 7 (Future Backends)
```

**Critical path**: 1 → 2 → 3 → 4 (gets us to a working three-tier compiler).
**Optimization path**: 5, 6 (search + rewrites, where SPEC\_3 philosophy pays off).

---

## Test Strategy

| Phase | Test Kind | What |
|-------|-----------|------|
| 1 | Unit | `Dim` classification, `AffineExpr` arithmetic, `normalize_dim`, type construction |
| 2 | Regression | New egglog bridge matches old results; pattern recognition tags correct |
| 3 | Integration | HLIR→Plan→LLIR produces correct loop structures (verified via pretty-printer) |
| 4 | **Differential** | Old vs new pipeline produce identical results for all 39 tests, then cutover |
| 5 | Benchmark | Beam search cost < greedy cost on matmul/softmax graphs |
| 6 | Correctness | Online softmax, blocked matmul produce correct numerical results |

---

## Open Questions (Resolved)

### Q: Where do `ScheduleTransform`s live?

The spec §7.2 defines LLIR-level transforms (Tile, Interchange, Fuse, Parallelize, Unroll, Vectorize, ComputeAt, CacheRead, CacheWrite) that are applied **after** Plan IR → LLIR lowering. They mutate a `Kernel`'s loop nest and require legality checking via `DependenceAnalyzer`.

**Resolution:** These are post-LLIR optimizations. They live in Phase 5/6 alongside the `DependenceAnalyzer` trait. They are not needed for the baseline pipeline (Phases 1–4), which emits correct-but-unoptimized loop nests. The greedy baseline just doesn't apply any transforms. Beam search (Phase 5) can propose transforms as part of its schedule candidates, and the LLIR optimizer applies them after lowering.

The types (`ScheduleTransform`, `ScheduleTransformer` trait) are defined in Phase 1 alongside the other LLIR types so they're available when needed.

### Q: What is `AnnotatedHLIR`?

The spec §9.1 uses `AnnotatedHLIR` as input to schedule search. It's the output of HLIR optimization + pattern recognition: an `HLIRGraph` with `SemanticRegion` annotations attached.

**Resolution:** `AnnotatedHLIR` is a simple struct:

```rust
pub struct AnnotatedHLIR {
    pub graph: HLIRGraph,
    pub regions: Vec<SemanticRegion>,
}
```

Defined in Phase 1 alongside HLIR types. Populated in Phase 2 by the e-graph pattern recognition pass.

### Q: What about the `Compiler` struct from §9.1?

The spec defines a `Compiler<H, D, C>` parameterized by `HardwareModel`, `DependenceAnalyzer`, and `CodeGenerator`. This is the full pipeline orchestrator.

**Resolution:** This replaces the current `compile()` function. Not needed until Phase 5 (when `HardwareModel` and search exist). During Phases 1–4, the pipeline is orchestrated by a simpler function chain. Phase 5 introduces the `Compiler` struct with its full generic parameterization.

### Q: How do `RuntimeParameter` symbols become kernel arguments?

At LLIR, `Var::Param("seq_len")` appears in `AffineExpr` loop bounds. The Cranelift backend needs to:

1. Add `seq_len` as a `KernelParam { kind: ParamKind::Dim }` to the kernel signature.
2. Emit it as an i64 function parameter in the Cranelift function.
3. When lowering an `AffineExpr`, replace `Var::Param("seq_len")` with the corresponding Cranelift `Value` from the function's parameter list.

At dispatch time, `CompiledProgram::execute()` collects the concrete shape values from the input buffers and passes them as scalar arguments to the kernel.

**Resolution:** Phase 4 (LLIR → Cranelift) handles this. The `Kernel::params` list already contains `ParamKind::Dim` entries. The Cranelift lowerer maps each to a function parameter.

### Q: What happens to `ShapeTracker`?

The existing `ShapeTracker` tracks strides/offsets for fused shape-op chains in the old pipeline. In the new pipeline, this role is absorbed by:

- `Layout::Strided(Vec<Dim>)` in `TensorType` at HLIR — symbolic strides.
- `MemoryAccess { indices: Vec<AffineExpr> }` at LLIR — explicit affine index expressions.

**Resolution:** `ShapeTracker` is deleted in Phase 4 (cutover). Its logic is replaced by the `Layout` type system and LLIR affine addressing.

### Q: Memory planning in the new pipeline?

The current `MemoryPlanningPass` does liveness analysis and greedy slot assignment on the flat `Graph`. In the new pipeline:

- `MaterializationChoice` in Plan IR decides what gets a buffer vs. fused.
- `BufferAlloc` in LLIR describes the actual allocations with `MemorySpace`.
- Liveness analysis moves to LLIR level, operating on `BufferAlloc`s and `Kernel` read/write sets.

**Resolution:** The existing `MemoryPlanningPass` is replaced by the Plan IR materialization decisions + an LLIR-level buffer allocation pass. Defined in Phase 3 (Plan IR → LLIR lowering).

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Symbolic shapes break concrete tests | All existing tests use `Dim::Const(n)` paths. Symbolic tested separately. |
| Egglog migration breaks optimization | Diff old and new egglog on same inputs before switching. |
| LLIR→Cranelift correctness bugs | Differential testing against old pipeline for every op. |
| Search scope creep | Greedy baseline (Phase 3) ships first; search (Phase 5) is additive. |
| Forward reference tangles | Phase 1 defines all types in one shot — no stubs. |
