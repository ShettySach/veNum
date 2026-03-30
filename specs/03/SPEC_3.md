# Venum IR & Scheduling Specification v3
## Backend-Agnostic, Search-First Design

---

## Changelog v2 → v3

| Issue                                                            | Resolution                                                                                      |
|-------                                                           |-----------                                                                                      |
| E-graph conflated across HLIR and Plan IR                        | E-graphs restricted to HLIR only; Plan IR uses explicit schedule search                         |
| `Specialization::Search` punted on algorithmic rewrites          | Replaced with `AlgorithmicRewrite` enum applied at Plan IR before lowering                      |
| `fused_with: Option<RegionId>` cannot represent multi-way fusion | Replaced with `FusionGroup` + union-find topology                                               |
| `TileSize::SearchParam` had no resolution pass                   | Added `TunedPlan` + `ParameterBinding` pass; LLIR receives only concrete tile sizes with guards |
| `Intrinsic { name: String }` broke backend-agnosticism           | Replaced with `AbstractVector` carrying typed SIMD operations                                   |
| `Reduction` loop kind had single accumulator                     | Replaced with `Vec<ReductionAccumulator>` to support joint reductions                           |
| `Dim::Div` / `Dim::Mod` had undefined lowering semantics         | Added normalization rules and validity classification                                           |
| `StmtId` undefined                                               | Defined                                                                                         |
| `BufferAlloc::shape` conversion from `Dim` implicit              | Made explicit in lowering description                                                           |
| Shared memory required runtime `Dim`                             | Constrained to `CompileTime`-resolved sizes                                                     |
| No symbol concretization model                                   | Added §4 `SymbolResolution` with full pipeline account                                          |

---

## 1. Philosophy

### 1.1 The Bitter Lesson Applied

The bitter lesson teaches that general methods leveraging computation scale better than domain-specific engineering. For a tensor compiler, this means:

1. **Search over hand-tuning** — The scheduler discovers good schedules via search, not hardcoded heuristics
2. **Primitives over special ops** — No `Attention`, `LayerNorm`, `Matmul` as IR nodes; they decompose to primitives
3. **Patterns are data, not code** — Fusion patterns are recognized via e-graph rules at HLIR, not hardcoded `if` branches
4. **Backend-agnostic representation** — Scheduling decisions are expressed in a neutral IR; backends interpret them

### 1.2 E-Graph Scope (Clarified)

E-graphs are used **only at HLIR** for two distinct purposes:

- **Algebraic rewriting** — Equivalence-preserving simplification over the expression DAG (identity elimination, reassociation, etc.)
- **Pattern recognition** — Annotating subgraphs with semantic tags (Contraction, Softmax, etc.)

Plan IR search is **not** e-graph equality saturation. Plan IR represents a space of scheduling decisions (fusion groups, tile sizes, parallelism strategies) over a DAG of semantic regions. This space is explored via **beam search** guided by the hardware cost model. The distinction matters: e-graph saturation works on term algebras where rewrites preserve a single semantic equivalence class; schedule search explores a combinatorial space where different schedules are not semantically equivalent but cost-equivalent candidates.

### 1.3 Non-Goals

- Hand-tuned kernel libraries as the primary path
- ISL or any single polyhedral library as a hard dependency
- Domain-specific ops that encode implementation choices
- E-graph equality saturation at Plan IR level

---

## 2. Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  HLIR (High-Level IR)                                       │
│  - Tensor operations on symbolic shapes                     │
│  - E-graph rewriting: algebraic simplification,             │
│    pattern recognition, region annotation                   │
└────────────────────────┬────────────────────────────────────┘
                         │ Region extraction
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Plan IR                                                    │
│  - Fusion groups with semantic tags                         │
│  - Algorithmic rewrites (online softmax, etc.)              │
│  - Beam search over candidate schedules                     │
│  - Parameter binding: SearchParams → concrete tile sizes    │
│  - Materialization vs. fused view decisions                 │
└────────────────────────┬────────────────────────────────────┘
                         │ Commit & lower
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LLIR (Low-Level IR)                                        │
│  - Explicit loop nests                                      │
│  - Concrete tile sizes, guards for non-divisible cases      │
│  - Abstract SIMD operations (backend lowers to intrinsics)  │
│  - Backend-agnostic but hardware-mappable                   │
└────────────────────────┬────────────────────────────────────┘
                         │ Codegen
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Target Code (CUDA, Metal, CPU SIMD, WebGPU WGSL)           │
└─────────────────────────────────────────────────────────────┘
```

**Why three tiers?**

- **HLIR** is for expressing computation without implementation bias
- **Plan IR** separates *what to fuse* from *how to lower* — search happens here
- **LLIR** is the committed schedule, ready for backend-specific codegen

---

## 3. Symbol Resolution Model

Symbols flow through the compiler at different granularities. This section is the definitive account of when each kind of symbol becomes concrete.

### 3.1 Resolution Phases

```rust
/// When does a symbol get its concrete value?
#[derive(Debug, Clone)]
pub enum SymbolResolution {
    /// Fixed before compilation begins.
    /// Examples: dtype, tensor rank.
    /// Must be concrete for loop nest structure.
    CompileTime(i64),

    /// Fixed after schedule search completes, before LLIR emission.
    /// Examples: tile_m, tile_n, tile_k (outputs of beam search).
    /// Must be concrete for LLIR loop steps and shared memory sizes.
    TuneTime(i64),

    /// Provided by the caller at kernel dispatch.
    /// Examples: batch_size, seq_len, N, M, K.
    /// Appear as `Var::Param` in AffineExpr; loops iterate symbolically.
    RuntimeParameter,

    /// Derived from RuntimeParameters via affine expression.
    /// Introduced by normalization of non-affine Dim expressions.
    /// Example: P_NM = N * M introduced by Reshape([N, M] -> [N*M]).
    RuntimeDerived { expr: AffineExpr },
}
```

### 3.2 Per-Tier Symbol Rules

| Tier | Allowed resolutions | Notes |
|------|-------------------|-------|
| HLIR | `RuntimeParameter`, `RuntimeDerived` | No concrete values; `Dim::Sym` everywhere |
| Plan IR (pre-binding) | All four | `SearchParam` tile sizes are `TuneTime`-pending |
| Plan IR (post-binding) | `CompileTime`, `TuneTime`, `RuntimeParameter`, `RuntimeDerived` | All `SearchParam` resolved |
| LLIR | `CompileTime`, `TuneTime`, `RuntimeParameter`, `RuntimeDerived` | No unresolved search params |
| Shared memory sizes | `CompileTime` or `TuneTime` only | CUDA requires statically-known `__shared__` size |

### 3.3 Non-Affine Dim Normalization

`Dim` at HLIR permits `Mul(Sym, Sym)`, `Div(Sym, Sym)`, `Mod(Sym, Sym)`. These are not affine and cannot appear in `AffineExpr` at LLIR. They must be normalized before lowering.

**Validity classification:**

```
Dim::Const(_)                 → affine (constant)
Dim::Sym(s)                   → affine (parameter)
Dim::Add(affine, affine)      → affine
Dim::Mul(Const, affine)       → affine (scalar multiplication)
Dim::Mul(Sym, Sym)            → NON-AFFINE → normalize
Dim::Div(Sym, Const)          → affine iff Const divides all values (floor div)
Dim::Div(Sym, Sym)            → NON-AFFINE → normalize
Dim::Mod(Sym, Const)          → affine-representable via floordiv identity
Dim::Mod(Sym, Sym)            → NON-AFFINE → normalize
```

**Normalization rules:**

```rust
/// Introduce a fresh RuntimeDerived parameter for a non-affine sub-expression.
/// Called during Plan IR → LLIR lowering.
pub fn normalize_dim(
    dim: &Dim,
    bindings: &mut SymbolBindingTable,
) -> AffineExpr {
    match dim {
        // Sym × Sym → fresh P_AB = A * B
        Dim::Mul(Sym(a), Sym(b)) => {
            let name = format!("P_{}_{}", a, b);
            bindings.introduce_derived(name.clone(), dim.clone());
            AffineExpr::var(Var::Param(name.into()))
        }
        // Sym / Sym → fresh Q_AB = A / B (asserted divisible or with guard)
        Dim::Div(Sym(a), Sym(b)) => {
            let name = format!("Q_{}_{}", a, b);
            bindings.introduce_derived(name.clone(), dim.clone());
            AffineExpr::var(Var::Param(name.into()))
        }
        // Sym % Const → representable: i % C = i - C * floor(i/C)
        Dim::Mod(e, Dim::Const(c)) => {
            let inner = normalize_dim(e, bindings);
            // Emit as: inner - c * floor(inner / c)
            // Represented as two affine params if inner is symbolic
            let floor_name = format!("floor_div_{}", c);
            bindings.introduce_derived(floor_name.clone(), dim.clone());
            inner.add(&AffineExpr::var(Var::Param(floor_name.into())).scale(-c))
        }
        // Sym % Sym → non-normalizable without runtime check; introduce fresh param
        Dim::Mod(Sym(a), Sym(b)) => {
            let name = format!("Mod_{}_{}", a, b);
            bindings.introduce_derived(name.clone(), dim.clone());
            AffineExpr::var(Var::Param(name.into()))
        }
        _ => /* affine cases: direct translation */ unreachable!("handled above"),
    }
}
```

`RuntimeDerived` parameters are computed in the kernel prologue from `RuntimeParameter` inputs before any loop begins.

---

## 4. HLIR Specification

### 4.1 Core Types

```rust
/// Symbolic dimension expression.
/// Non-affine forms (Mul(Sym,Sym), Div(Sym,Sym), Mod(Sym,Sym)) are valid at
/// HLIR but must be normalized before LLIR emission (see §3.3).
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Dim {
    Const(i64),
    Sym(Symbol),                    // Named parameter: "N", "seq_len"
    Add(Box<Dim>, Box<Dim>),
    Mul(Box<Dim>, Box<Dim>),        // Sym × Sym allowed at HLIR; normalized at lowering
    Div(Box<Dim>, Box<Dim>),        // Sym / Sym allowed at HLIR; normalized at lowering
    Mod(Box<Dim>, Box<Dim>),        // Sym % Sym allowed at HLIR; normalized at lowering
}

pub type Symbol = InternedString;   // Interned for cheap equality

/// Tensor type with shape and layout.
#[derive(Debug, Clone)]
pub struct TensorType {
    pub shape: Vec<Dim>,
    pub dtype: DType,
    pub layout: Layout,
}

#[derive(Debug, Clone)]
pub enum Layout {
    Contiguous,                           // Row-major, computed strides
    Strided(Vec<Dim>),                    // Explicit strides (may include 0 for broadcast)
    View { base: TensorId, offset: Dim, strides: Vec<Dim> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32, F16, BF16, F64,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
}
```

### 4.2 Primitive Operations (~15)

```rust
#[derive(Debug, Clone)]
pub enum Op {
    // ===== Constants & Memory =====
    Const { value: Scalar, shape: Vec<Dim>, dtype: DType },
    Load { buffer: BufferId },
    Store { buffer: BufferId, value: NodeId },

    // ===== Unary =====
    Neg(NodeId),
    Recip(NodeId),          // 1/x
    Exp(NodeId),
    Log(NodeId),
    Sqrt(NodeId),
    Sin(NodeId),
    Cos(NodeId),
    Cast { input: NodeId, to: DType },

    // ===== Binary =====
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Max(NodeId, NodeId),
    Min(NodeId, NodeId),
    Cmp { op: CmpOp, lhs: NodeId, rhs: NodeId },

    // ===== Ternary =====
    Where { cond: NodeId, then_val: NodeId, else_val: NodeId },

    // ===== Reductions =====
    Reduce { input: NodeId, axes: Vec<usize>, op: ReduceOp, keepdim: bool },

    // ===== Shape/View =====
    Reshape { input: NodeId, shape: Vec<Dim> },
    Permute { input: NodeId, axes: Vec<usize> },
    Slice { input: NodeId, ranges: Vec<Range> },
    Expand { input: NodeId, shape: Vec<Dim> },  // Broadcast
    Concat { inputs: Vec<NodeId>, axis: usize },
}

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum, Prod, Max, Min,
}

#[derive(Debug, Clone, Copy)]
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

#[derive(Debug, Clone)]
pub struct Range {
    pub start: Dim,
    pub end: Dim,
    pub step: Dim,
}
```

**Derived operations** (expressed as HLIR subgraphs, not primitive ops):

| Operation | Decomposition |
|-----------|---------------|
| `Sub(a, b)` | `Add(a, Neg(b))` |
| `Div(a, b)` | `Mul(a, Recip(b))` |
| `Matmul(A, B)` | `Reduce(Sum, Mul(Expand(A), Expand(B)), axis=-1)` |
| `Softmax(x)` | `Div(Exp(Sub(x, Reduce(Max, x))), Reduce(Sum, Exp(Sub(x, Reduce(Max, x)))))` |
| `LayerNorm(x)` | `Div(Sub(x, mean), Sqrt(Add(var, eps)))` |

### 4.3 HLIR Graph

```rust
pub struct HLIRGraph {
    pub nodes: Vec<HLIRNode>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub buffers: Vec<BufferDecl>,
    pub symbols: HashSet<Symbol>,   // All symbolic params used
}

pub struct HLIRNode {
    pub id: NodeId,
    pub op: Op,
    pub ty: TensorType,
    pub region: Option<RegionId>,   // Assigned after pattern recognition
}

pub struct BufferDecl {
    pub id: BufferId,
    pub ty: TensorType,
    pub kind: BufferKind,
}

pub enum BufferKind { Input, Output, Intermediate }
```

### 4.4 E-Graph Integration at HLIR

E-graphs operate **only at HLIR**. Two distinct uses:

**Algebraic rewrites** (equivalence-preserving, drive toward smaller e-classes):
- `Add(x, Const(0))` ↔ `x`
- `Mul(x, Const(1))` ↔ `x`
- `Mul(x, Const(0))` ↔ `Const(0)`
- `Neg(Neg(x))` ↔ `x`
- `Exp(Log(x))` ↔ `x` (domain-restricted)
- `Mul(Recip(x), y)` ↔ `Mul(y, Recip(x))` (commutativity for cost)

**Pattern recognition** (annotation, not rewrite — does not alter e-class membership):
- `Reduce(Sum, Mul(Expand(_), Expand(_)))` → tag as `Contraction`
- `Exp(Sub(x, Reduce(Max, x)))` followed by `Reduce(Sum, ...)` → tag as `Softmax`
- Mean + Variance over same input → tag as `Normalization`

Pattern recognition produces `SemanticRegion` annotations on the HLIR graph. These annotations drive Plan IR fusion group formation and algorithmic rewrite eligibility, but they do not change the graph structure.

**Region annotation output:**

```rust
pub struct SemanticRegion {
    pub id: RegionId,
    pub nodes: Vec<NodeId>,
    pub tag: SemanticTag,
    pub metadata: RegionMetadata,
}

#[derive(Debug, Clone)]
pub enum SemanticTag {
    Elementwise,
    Contraction { m: Dim, n: Dim, k: Dim },
    Reduction { axes: Vec<usize> },
    Softmax { axis: usize },
    Normalization { axes: Vec<usize> },
    /// Tagged but lowered as a fusion of Contraction + Softmax regions.
    /// Eligible for algorithmic rewrite to blocked attention at Plan IR.
    Attention { seq_axis: usize, head_dim: Dim },
    Generic,
}

pub struct RegionMetadata {
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub estimated_flops: Dim,       // Symbolic
    pub estimated_memory: Dim,      // Symbolic
}
```

---

## 5. Plan IR Specification

Plan IR captures scheduling decisions *before* lowering to explicit loops. It is the output of HLIR-level search and the input to LLIR lowering. Plan IR search is **beam search over a schedule space**, not e-graph saturation.

### 5.1 Fusion Groups

Multi-region fusion is represented via explicit fusion groups, not per-region pointers.

```rust
pub struct FusionGroup {
    pub id: FusionGroupId,
    /// Topologically ordered; must form a connected producer-consumer DAG.
    pub regions: Vec<RegionId>,
    pub topology: FusionTopology,
}

#[derive(Debug, Clone)]
pub enum FusionTopology {
    /// Linear chain: R0 → R1 → R2 → ...
    /// Common case: elementwise chains, softmax stages.
    Chain,
    /// One or more producers fan into a single consumer.
    /// Example: two elementwise ops feeding a reduction.
    FanIn { consumer: RegionId, producers: Vec<RegionId> },
    /// One producer fans out into multiple consumers (rare; implies duplication or barrier).
    FanOut { producer: RegionId, consumers: Vec<RegionId> },
    /// Arbitrary DAG for complex patterns (attention = matmul + softmax + matmul).
    /// Edges are (producer_region, consumer_region) pairs.
    DAG { edges: Vec<(RegionId, RegionId)> },
}

/// Check that all regions in a fusion group form a valid fusable subgraph.
/// Legality: no region in the group has an output consumed outside the group
/// except for the group's designated output regions.
pub fn check_fusion_legality(
    group: &FusionGroup,
    graph: &PlanGraph,
) -> Result<(), FusionError>;
```

### 5.2 Core Plan IR Types

```rust
pub struct PlanGraph {
    pub regions: Vec<PlannedRegion>,
    pub fusion_groups: Vec<FusionGroup>,
    pub execution_order: Vec<FusionGroupId>,    // Topological order over groups
    pub materializations: HashMap<NodeId, MaterializationChoice>,
    pub symbol_bindings: SymbolBindingTable,
}

pub struct PlannedRegion {
    pub id: RegionId,
    pub semantic: SemanticTag,
    pub nodes: Vec<NodeId>,
    pub schedule: RegionSchedule,
    /// Which fusion group this region belongs to, if any.
    pub fusion_group: Option<FusionGroupId>,
    /// Algorithmic rewrite applied to this region before lowering.
    pub algorithmic_rewrite: Option<AlgorithmicRewrite>,
}

#[derive(Debug, Clone)]
pub struct RegionSchedule {
    pub tiling: Option<TilingSpec>,
    pub parallelism: ParallelismSpec,
    pub memory_placement: MemoryPlacement,
    /// After parameter binding, this is always `Specialization::LoopNest`
    /// or `Specialization::LibraryCall`. `Search` is only valid pre-binding.
    pub specialization: Specialization,
}

#[derive(Debug, Clone)]
pub struct TilingSpec {
    pub tile_sizes: Vec<TileSize>,
    pub tile_order: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum TileSize {
    /// Fully concrete: either hardware-fixed or bound by parameter binding pass.
    Const(i64),
    /// Unresolved search parameter. Only valid in pre-binding PlanGraph.
    /// Must not appear in LLIR.
    SearchParam(String),
}

#[derive(Debug, Clone)]
pub enum ParallelismSpec {
    Sequential,
    Parallel { axis: usize, num_threads: ParamOrConst },
    GPU { grid: Vec<Dim>, block: Vec<Dim> },
}

#[derive(Debug, Clone)]
pub enum MemoryPlacement {
    Default,
    /// Size must resolve to CompileTime or TuneTime (not RuntimeParameter).
    /// Constraint: the backend codegen will assert this at LLIR emission.
    SharedMemory { size: Dim },
    Registers,
    ExplicitCache { level: usize },
}

#[derive(Debug, Clone)]
pub enum Specialization {
    LoopNest,
    LibraryCall { name: String },
    CustomKernel { template: String },
}
```

### 5.3 Algorithmic Rewrites

Algorithmic rewrites are transformations that change the *algorithm*, not just the schedule. They cannot be derived from loop transformations alone and must be applied explicitly at Plan IR before lowering. Each rewrite targets a recognized `SemanticTag`.

```rust
/// An algorithmic rewrite transforms a PlannedRegion into an equivalent
/// but algorithmically different implementation before LLIR lowering.
/// Unlike schedule transforms, these change what state is maintained across
/// iterations, not merely the order of existing operations.
#[derive(Debug, Clone)]
pub enum AlgorithmicRewrite {
    /// Two-pass softmax → single-pass online softmax (Flash attention style).
    /// Applicable to regions tagged Softmax { axis }.
    /// Precondition: the region is fused (not materialized between stages).
    /// Effect: introduces a joint (running_max, running_sum, accumulator)
    ///   reduction loop replacing the separate max-pass and sum-pass loops.
    OnlineSoftmax {
        axis: usize,
    },

    /// Standard matmul → register-blocked tiled matmul with explicit
    /// register accumulation. Required for tensor core and SIMD utilization.
    /// Applicable to regions tagged Contraction { m, n, k }.
    BlockedMatmul {
        block_m: i64,
        block_n: i64,
        block_k: i64,
    },

    /// Attention = QK^T + softmax + AV → fused blocked attention.
    /// Applicable only to Attention-tagged regions (or fusion groups
    /// containing Contraction + Softmax + Contraction in that topology).
    /// Subsumes OnlineSoftmax and BlockedMatmul.
    BlockedAttention {
        block_q: i64,
        block_k: i64,
    },

    /// Mean + variance in two passes → Welford one-pass online algorithm.
    /// Applicable to Normalization-tagged regions.
    WelfordNormalization,
}

impl AlgorithmicRewrite {
    /// Verify this rewrite is applicable to the given region.
    pub fn check_preconditions(
        &self,
        region: &PlannedRegion,
        group: Option<&FusionGroup>,
    ) -> Result<(), RewriteError>;

    /// Apply the rewrite, producing a modified region with updated schedule
    /// and a RewriteSpec that the LLIR lowerer uses to emit the correct loops.
    pub fn apply(
        &self,
        region: &PlannedRegion,
    ) -> Result<(PlannedRegion, RewriteSpec), RewriteError>;
}

/// Instructions passed to the LLIR lowerer describing how to emit
/// algorithmically-rewritten loop nests.
#[derive(Debug, Clone)]
pub enum RewriteSpec {
    OnlineSoftmax { axis: usize, tile_size: i64 },
    BlockedMatmul { block_m: i64, block_n: i64, block_k: i64 },
    BlockedAttention { block_q: i64, block_k: i64 },
    WelfordNormalization { axes: Vec<usize> },
}
```

### 5.4 Materialization Decisions

```rust
#[derive(Debug, Clone)]
pub enum MaterializationChoice {
    /// Compute inline, no intermediate buffer.
    Fused,
    /// Allocate buffer, store result.
    Materialized { buffer: BufferId },
    /// Reuse existing buffer (aliasing).
    Alias { base: BufferId, offset: Dim },
}
```

### 5.5 Symbol Binding Table

```rust
pub struct SymbolBindingTable {
    /// Resolution status for every symbol in the program.
    pub resolutions: HashMap<Symbol, SymbolResolution>,
    /// SearchParam name → concrete TuneTime value.
    /// Populated by the parameter binding pass.
    pub tune_time_bindings: HashMap<String, i64>,
}

pub struct SymbolBinding {
    pub symbol: Symbol,
    pub definition: SymbolDef,
}

pub enum SymbolDef {
    Parameter,                                  // RuntimeParameter
    Derived { expr: Dim, from: Vec<Symbol> },   // RuntimeDerived: N*M → P_NM
}
```

### 5.6 Schedule Search

Plan IR search is beam search over the schedule space. It is not equality saturation.

```rust
/// The schedule space for a PlanGraph: the set of all valid combinations
/// of fusion decisions, tile sizes, and parallelism strategies.
pub struct ScheduleSpace {
    pub fusion_candidates: Vec<FusionCandidate>,
    pub tile_param_ranges: HashMap<String, Vec<i64>>,   // "tile_m" → [16, 32, 64, 128]
    pub parallelism_options: Vec<ParallelismSpec>,
}

pub struct FusionCandidate {
    pub group: FusionGroup,
    pub legality: FusionLegality,
}

pub enum FusionLegality {
    Legal,
    IllegalDueToReduction,      // Would split a reduction incorrectly
    IllegalDueToCycle,          // Would introduce a dependency cycle
    IllegalDueToMemory,         // Working set would exceed target memory level
}

/// Beam search over ScheduleSpace guided by HardwareModel cost.
pub struct ScheduleSearcher<H: HardwareModel> {
    pub hardware: H,
    pub beam_width: usize,
    pub max_iterations: usize,
}

impl<H: HardwareModel> ScheduleSearcher<H> {
    /// Returns a set of candidate PlanGraphs ranked by estimated cost.
    /// The caller selects the best (or runs multiple for profiling).
    pub fn search(
        &self,
        annotated_hlir: &AnnotatedHLIR,
        space: &ScheduleSpace,
    ) -> Vec<(PlanGraph, CostEstimate)>;
}
```

### 5.7 Parameter Binding Pass

After search selects a schedule, all `SearchParam` tile sizes must be resolved to concrete values. This pass also generates divisibility guards that will appear in the LLIR.

```rust
/// A PlanGraph with all SearchParams resolved.
/// Invariant: no TileSize::SearchParam exists anywhere in this graph.
pub struct TunedPlan {
    pub plan: PlanGraph,
    /// The concrete tile sizes chosen by search.
    pub tile_bindings: HashMap<String, i64>,
    /// For each (tile_param, loop_bound_symbol), whether the tile evenly
    /// divides all known values of that symbol.
    /// If false, LLIR lowering must emit a remainder epilogue.
    pub divisibility: HashMap<(String, Symbol), Divisibility>,
}

#[derive(Debug, Clone, Copy)]
pub enum Divisibility {
    /// tile_size divides loop_bound for all valid inputs.
    /// No epilogue needed.
    AlwaysDivisible,
    /// May not divide; LLIR must emit main loop + remainder epilogue.
    RequiresEpilogue,
}

pub fn bind_parameters(
    plan: PlanGraph,
    bindings: &HashMap<String, i64>,
) -> Result<TunedPlan>;
```

---

## 6. LLIR Specification

LLIR is the committed, lowered representation. It expresses explicit loop nests and memory accesses. All `SearchParam` values are resolved; remainder epilogues are explicit; symbolic params are `RuntimeParameter` only.

### 6.1 Core Types

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
    pub provenance: SemanticTag,        // Preserved from Plan IR
    /// If this kernel was produced by an algorithmic rewrite, records which one.
    pub rewrite: Option<RewriteSpec>,
}

pub struct KernelParam {
    pub name: String,
    pub kind: ParamKind,
}

pub enum ParamKind {
    Buffer { ty: TensorType },
    Scalar { ty: DType },
    /// A RuntimeParameter or RuntimeDerived symbol passed as a kernel argument.
    Dim { symbol: Symbol },
}
```

### 6.2 Statement Identifiers

```rust
/// Uniquely identifies a statement within a kernel for dependence tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StmtId {
    pub kernel: KernelId,
    /// Index in a depth-first pre-order traversal of the LoopNest's statements.
    pub index: usize,
}
```

### 6.3 Loop Representation

```rust
#[derive(Debug, Clone)]
pub struct LoopNest {
    pub loops: Vec<Loop>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub struct Loop {
    pub var: LoopVar,
    pub lower: AffineExpr,
    pub upper: AffineExpr,
    /// Always a concrete i64 after parameter binding.
    pub step: i64,
    pub kind: LoopKind,
    pub annotations: LoopAnnotations,
}

pub type LoopVar = String;

#[derive(Debug, Clone)]
pub enum LoopKind {
    Sequential,
    Parallel,
    /// Abstract vectorization. Width is a TuneTime constant.
    /// Backend lowers to concrete SIMD instructions.
    Vectorized { width: usize },
    Unrolled { factor: usize },
    // GPU-specific
    GridDim { axis: usize },        // blockIdx.x/y/z
    BlockDim { axis: usize },       // threadIdx.x/y/z
    /// Reduction loop with one or more simultaneous accumulators.
    /// Multiple accumulators support joint reductions (e.g., online softmax
    /// maintaining running_max and running_sum simultaneously).
    Reduction { accumulators: Vec<ReductionAccumulator> },
}

/// A single accumulator variable maintained across a reduction loop.
#[derive(Debug, Clone)]
pub struct ReductionAccumulator {
    /// Name of the accumulator variable in the loop body.
    pub var: String,
    pub op: ReduceOp,
    pub init: Scalar,
    pub dtype: DType,
}

#[derive(Debug, Clone, Default)]
pub struct LoopAnnotations {
    pub tile_origin: Option<LoopVar>,
    pub cache_at: Option<usize>,
    pub unroll_hint: Option<usize>,
}
```

### 6.4 Affine Expressions

```rust
/// Affine expression: c0 + c1*v1 + c2*v2 + ...
/// Strictly linear in loop variables and RuntimeParameter/RuntimeDerived symbols.
/// No TuneTime or CompileTime symbols appear here; those were substituted
/// during Plan IR → LLIR lowering.
#[derive(Debug, Clone)]
pub struct AffineExpr {
    pub constant: i64,
    pub terms: Vec<(i64, Var)>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Var {
    Loop(LoopVar),
    /// A RuntimeParameter or RuntimeDerived symbol.
    Param(Symbol),
}

impl AffineExpr {
    pub fn constant(c: i64) -> Self { Self { constant: c, terms: vec![] } }
    pub fn var(v: Var) -> Self { Self { constant: 0, terms: vec![(1, v)] } }
    pub fn add(&self, other: &Self) -> Self { /* ... */ todo!() }
    pub fn scale(&self, c: i64) -> Self { /* ... */ todo!() }
    pub fn is_affine(&self) -> bool { true }  // True by construction
}
```

### 6.5 Memory Access

```rust
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub buffer: BufferId,
    pub indices: Vec<AffineExpr>,
    pub access_kind: AccessKind,
}

#[derive(Debug, Clone, Copy)]
pub enum AccessKind {
    Read,
    Write,
    ReadWrite,      // Reductions
}

/// Buffer allocation at LLIR.
/// shape is Vec<AffineExpr> (converted from Vec<Dim> during lowering).
/// For SharedMemory, all AffineExprs must evaluate to CompileTime or TuneTime
/// constants — no RuntimeParameter terms allowed.
#[derive(Debug, Clone)]
pub struct BufferAlloc {
    pub id: BufferId,
    /// Converted from HLIR Vec<Dim> by normalize_dim (§3.3).
    /// RuntimeParameter symbols appear as Var::Param terms.
    pub shape: Vec<AffineExpr>,
    pub dtype: DType,
    pub memory_space: MemorySpace,
}

#[derive(Debug, Clone, Copy)]
pub enum MemorySpace {
    Global,
    /// Requires shape to be statically evaluable (CompileTime/TuneTime only).
    Shared,
    Local,      // GPU registers / CPU stack
    Constant,
}
```

### 6.6 Statements

```rust
#[derive(Debug, Clone)]
pub enum Stmt {
    Assign {
        dst: MemoryAccess,
        src: Expr,
    },
    /// For reductions: dst[indices] op= src
    Accumulate {
        dst: MemoryAccess,
        op: ReduceOp,
        src: Expr,
    },
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    Loop(Loop, Vec<Stmt>),
    Barrier { scope: BarrierScope },
    /// Remainder epilogue marker: contains the tail loop for non-divisible
    /// tile sizes. Generated by the parameter binding pass when
    /// Divisibility::RequiresEpilogue.
    Epilogue { main_loop_var: LoopVar, remainder_body: Vec<Stmt> },
    Nop,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Scalar),
    Load(MemoryAccess),
    Unary { op: UnaryOp, arg: Box<Expr> },
    Binary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Ternary { cond: Box<Expr>, then_val: Box<Expr>, else_val: Box<Expr> },
    Cast { arg: Box<Expr>, to: DType },
    /// Abstract SIMD operation. Does NOT name a backend intrinsic.
    /// The backend codegen trait lowers this to e.g. _mm256_fmadd_ps, vfmadd.
    AbstractVector(AbstractVectorOp),
}

/// Backend-agnostic SIMD operations.
/// Width is always a TuneTime constant matching the target's vector register width.
#[derive(Debug, Clone)]
pub enum AbstractVectorOp {
    /// Fused multiply-add: acc + (lhs * rhs)
    Fma { acc: Box<Expr>, lhs: Box<Expr>, rhs: Box<Expr>, width: usize },
    /// Horizontal reduction across a vector register
    HorizontalReduce { op: ReduceOp, arg: Box<Expr>, width: usize },
    /// Broadcast scalar to all lanes
    Broadcast { scalar: Box<Expr>, width: usize },
    /// Gather: load from non-contiguous indices
    Gather { base: BufferId, indices: Box<Expr>, width: usize },
    /// Scatter: store to non-contiguous indices
    Scatter { base: BufferId, indices: Box<Expr>, value: Box<Expr>, width: usize },
    /// Elementwise op over a vector register (lowered to e.g. _mm256_add_ps)
    VecBinary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr>, width: usize },
    /// Type-conversion across a vector (e.g. F16→F32 widening)
    VecCast { arg: Box<Expr>, from: DType, to: DType, width: usize },
}

#[derive(Debug, Clone, Copy)]
pub enum BarrierScope {
    Workgroup,      // __syncthreads()
    Subgroup,       // Warp-level
    Device,
}
```

### 6.7 Dependence Representation

```rust
pub struct Dependence {
    pub from: StmtId,
    pub to: StmtId,
    pub kind: DepKind,
    pub distance: Option<Vec<i64>>,
    pub relation: DependenceRelation,
}

#[derive(Debug, Clone, Copy)]
pub enum DepKind {
    RAW,    // Read after write (true dependence)
    WAR,    // Write after read (anti-dependence)
    WAW,    // Write after write (output dependence)
}

/// Backend-agnostic dependence relation.
/// { [source_iters] -> [sink_iters] : constraints }
#[derive(Debug, Clone)]
pub struct DependenceRelation {
    pub source_vars: Vec<LoopVar>,
    pub sink_vars: Vec<LoopVar>,
    pub constraints: Vec<AffineConstraint>,
}

#[derive(Debug, Clone)]
pub struct AffineConstraint {
    pub expr: AffineExpr,
    pub kind: ConstraintKind,
}

#[derive(Debug, Clone, Copy)]
pub enum ConstraintKind {
    Eq,     // expr = 0
    Ge,     // expr >= 0
}
```

---

## 7. Backend Traits

### 7.1 Dependence Analysis

```rust
pub trait DependenceAnalyzer {
    fn analyze_kernel(&self, kernel: &Kernel) -> Result<Vec<Dependence>>;

    fn check_legality(
        &self,
        deps: &[Dependence],
        transform: &ScheduleTransform,
    ) -> Result<bool>;

    fn access_dependence(
        &self,
        write: &MemoryAccess,
        read: &MemoryAccess,
        loops: &[Loop],
    ) -> Result<Option<DependenceRelation>>;
}
```

### 7.2 Schedule Transformations

```rust
#[derive(Debug, Clone)]
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

#[derive(Debug, Clone, Copy)]
pub enum ParallelKind {
    Thread,
    SIMD,
    GPU { dim: usize },
}

pub trait ScheduleTransformer {
    fn apply(&self, kernel: &Kernel, transform: &ScheduleTransform) -> Result<Kernel>;

    fn apply_sequence(&self, kernel: &Kernel, transforms: &[ScheduleTransform]) -> Result<Kernel> {
        let mut k = kernel.clone();
        for t in transforms {
            k = self.apply(&k, t)?;
        }
        Ok(k)
    }
}
```

### 7.3 Code Generation

```rust
pub trait CodeGenerator {
    type Output;
    fn generate(&self, program: &LLIRProgram) -> Result<Self::Output>;
    fn generate_kernel(&self, kernel: &Kernel) -> Result<String>;
}

pub trait CpuCodeGen: CodeGenerator<Output = CpuModule> {
    fn intrinsics(&self) -> &CpuIntrinsics;
    /// Lower AbstractVectorOp to platform intrinsic string.
    fn lower_vector_op(&self, op: &AbstractVectorOp, dtype: DType) -> String;
}

pub trait GpuCodeGen: CodeGenerator<Output = GpuModule> {
    fn max_threads_per_block(&self) -> usize;
    fn max_shared_memory(&self) -> usize;
    fn warp_size(&self) -> usize;
    /// Assert all SharedMemory buffers have statically-evaluable sizes.
    fn validate_shared_memory(&self, program: &LLIRProgram) -> Result<()>;
}

pub struct CpuModule {
    pub source: String,
    pub symbols: Vec<String>,
}

pub struct GpuModule {
    pub source: String,
    pub entry_points: Vec<GpuEntryPoint>,
}

pub struct GpuEntryPoint {
    pub name: String,
    pub grid_dims: usize,
    pub block_dims: usize,
    pub shared_memory: usize,   // Concrete bytes; validated by GpuCodeGen
}
```

---

## 8. Cost Model & Hardware Abstraction

### 8.1 Hardware Model Trait

```rust
pub trait HardwareModel {
    fn memory_levels(&self) -> &[MemoryLevel];
    fn compute(&self) -> &ComputeCapabilities;
    fn estimate_cost(&self, kernel: &Kernel) -> CostEstimate;
    fn memory_cost(&self, access: &MemoryAccess, loops: &[Loop]) -> f64;
}

#[derive(Debug, Clone)]
pub struct MemoryLevel {
    pub name: String,
    pub size_bytes: usize,
    pub bandwidth_gbps: f64,
    pub latency_cycles: usize,
}

#[derive(Debug, Clone)]
pub struct ComputeCapabilities {
    pub vector_width: HashMap<DType, usize>,
    pub peak_flops: HashMap<DType, f64>,
    pub num_cores: usize,
    pub num_threads_per_core: usize,
    pub num_sms: Option<usize>,
    pub warp_size: Option<usize>,
    pub tensor_cores: Option<TensorCoreSpec>,
}

#[derive(Debug, Clone)]
pub struct TensorCoreSpec {
    pub supported_shapes: Vec<(usize, usize, usize)>,
    pub supported_dtypes: Vec<(DType, DType)>,
    pub throughput: f64,
}
```

### 8.2 Cost Estimation

```rust
#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub compute_cycles: f64,
    pub memory_cycles: f64,
    pub total_cycles: f64,          // max(compute, memory) — roofline model
    pub working_set_size: usize,
    pub arithmetic_intensity: f64,
    pub bottleneck: Bottleneck,
}

#[derive(Debug, Clone, Copy)]
pub enum Bottleneck {
    Compute,
    MemoryL1, MemoryL2, MemoryL3, MemoryDRAM,
    MemoryShared, MemoryGlobal,
    Latency,
}

impl HardwareModel for GenericCpuModel {
    fn estimate_cost(&self, kernel: &Kernel) -> CostEstimate {
        let flops = count_flops(&kernel.body);
        let (reads, writes) = count_memory_ops(&kernel.reads, &kernel.writes);
        let working_set = estimate_working_set(kernel);

        let mem_level = self.memory_levels()
            .iter()
            .find(|l| working_set <= l.size_bytes)
            .unwrap_or(self.memory_levels().last().unwrap());

        let compute_time = flops / self.compute().peak_flops[&DType::F32];
        let memory_time = (reads + writes) as f64 / (mem_level.bandwidth_gbps * 1e9);

        CostEstimate {
            compute_cycles: compute_time * self.clock_ghz() * 1e9,
            memory_cycles: memory_time * self.clock_ghz() * 1e9,
            total_cycles: compute_time.max(memory_time) * self.clock_ghz() * 1e9,
            working_set_size: working_set,
            arithmetic_intensity: flops as f64 / (reads + writes) as f64,
            bottleneck: if compute_time > memory_time {
                Bottleneck::Compute
            } else {
                match mem_level.name.as_str() {
                    "L1" => Bottleneck::MemoryL1,
                    "L2" => Bottleneck::MemoryL2,
                    "L3" => Bottleneck::MemoryL3,
                    _ => Bottleneck::MemoryDRAM,
                }
            },
        }
    }
}
```

### 8.3 Predefined Hardware Models

```rust
pub fn cpu_x86_64_generic() -> impl HardwareModel {
    GenericCpuModel {
        memory_levels: vec![
            MemoryLevel { name: "L1".into(), size_bytes: 32 * 1024, bandwidth_gbps: 1000.0, latency_cycles: 4 },
            MemoryLevel { name: "L2".into(), size_bytes: 256 * 1024, bandwidth_gbps: 500.0, latency_cycles: 12 },
            MemoryLevel { name: "L3".into(), size_bytes: 8 * 1024 * 1024, bandwidth_gbps: 200.0, latency_cycles: 40 },
            MemoryLevel { name: "DRAM".into(), size_bytes: usize::MAX, bandwidth_gbps: 50.0, latency_cycles: 200 },
        ],
        compute: ComputeCapabilities {
            vector_width: [(DType::F32, 8), (DType::F64, 4)].into(),  // AVX-256
            peak_flops: [(DType::F32, 500e9)].into(),
            num_cores: 8,
            num_threads_per_core: 2,
            ..Default::default()
        },
    }
}

pub fn gpu_cuda_generic() -> impl HardwareModel {
    GenericGpuModel {
        memory_levels: vec![
            MemoryLevel { name: "Registers".into(), size_bytes: 256 * 1024, bandwidth_gbps: 10000.0, latency_cycles: 0 },
            MemoryLevel { name: "Shared".into(), size_bytes: 48 * 1024, bandwidth_gbps: 5000.0, latency_cycles: 20 },
            MemoryLevel { name: "L2".into(), size_bytes: 6 * 1024 * 1024, bandwidth_gbps: 2000.0, latency_cycles: 200 },
            MemoryLevel { name: "Global".into(), size_bytes: usize::MAX, bandwidth_gbps: 900.0, latency_cycles: 400 },
        ],
        compute: ComputeCapabilities {
            num_sms: Some(80),
            warp_size: Some(32),
            peak_flops: [(DType::F32, 20e12)].into(),
            tensor_cores: Some(TensorCoreSpec {
                supported_shapes: vec![(16, 16, 16), (8, 32, 16)],
                supported_dtypes: vec![(DType::F16, DType::F32), (DType::BF16, DType::F32)],
                throughput: 300e12,
            }),
            ..Default::default()
        },
    }
}
```

---

## 9. Pipeline Integration

### 9.1 Full Compilation Pipeline

```rust
pub struct Compiler<H: HardwareModel, D: DependenceAnalyzer, C: CodeGenerator> {
    hardware: H,
    dep_analyzer: D,
    codegen: C,
    egraph_config: EGraphConfig,
    search_config: SearchConfig,
}

impl<H, D, C> Compiler<H, D, C>
where
    H: HardwareModel,
    D: DependenceAnalyzer,
    C: CodeGenerator,
{
    pub fn compile(&self, hlir: HLIRGraph) -> Result<C::Output> {
        // Phase 1: HLIR e-graph optimization (algebraic rewrites only)
        let optimized_hlir = self.optimize_hlir(hlir)?;

        // Phase 2: Pattern recognition via e-graph matching → SemanticRegions
        let annotated = self.annotate_regions(optimized_hlir)?;

        // Phase 3: Build schedule space (fusion candidates, tile param ranges)
        let space = self.build_schedule_space(&annotated)?;

        // Phase 4: Beam search over schedule space → ranked PlanGraphs
        let searcher = ScheduleSearcher {
            hardware: &self.hardware,
            beam_width: self.search_config.beam_width,
            max_iterations: self.search_config.max_iterations,
        };
        let candidates = searcher.search(&annotated, &space);
        let best_plan = candidates.into_iter().next()
            .ok_or(CompileError::NoViableSchedule)?.0;

        // Phase 5: Apply algorithmic rewrites to eligible regions
        let rewritten_plan = self.apply_algorithmic_rewrites(best_plan)?;

        // Phase 6: Parameter binding — resolve all SearchParams to concrete i64
        let tuned = bind_parameters(rewritten_plan, &self.search_config.initial_bindings)?;

        // Phase 7: Lower TunedPlan → LLIR (with epilogues where required)
        let llir = self.lower_to_llir(tuned)?;

        // Phase 8: LLIR-level schedule transforms (legality-checked)
        let optimized_llir = self.optimize_llir(llir)?;

        // Phase 9: Codegen
        self.codegen.generate(&optimized_llir)
    }

    fn optimize_hlir(&self, hlir: HLIRGraph) -> Result<HLIRGraph> {
        // E-graph equality saturation with algebraic rewrite rules.
        // Extraction: MinCost based on operation count.
        todo!()
    }

    fn annotate_regions(&self, hlir: HLIRGraph) -> Result<AnnotatedHLIR> {
        // Pattern matching via e-graph: produces SemanticRegion annotations.
        // Does not modify graph structure.
        todo!()
    }

    fn build_schedule_space(&self, annotated: &AnnotatedHLIR) -> Result<ScheduleSpace> {
        // Enumerate valid fusion candidates (check legality).
        // Enumerate tile param ranges from hardware model.
        todo!()
    }

    fn apply_algorithmic_rewrites(&self, plan: PlanGraph) -> Result<PlanGraph> {
        // For each PlannedRegion with an eligible SemanticTag,
        // check if an AlgorithmicRewrite improves estimated cost.
        // Apply rewrites in topological order.
        todo!()
    }

    fn lower_to_llir(&self, tuned: TunedPlan) -> Result<LLIRProgram> {
        // For each FusionGroup in execution_order:
        //   - Normalize Dim → AffineExpr (see §3.3)
        //   - Emit loop nest from RegionSchedule
        //   - If rewrite: use RewriteSpec to emit specialized loop structure
        //   - If Divisibility::RequiresEpilogue: emit Stmt::Epilogue
        //   - Emit BufferAlloc with converted shape: Vec<Dim> → Vec<AffineExpr>
        //   - Validate SharedMemory buffers have no RuntimeParameter shape terms
        todo!()
    }

    fn optimize_llir(&self, llir: LLIRProgram) -> Result<LLIRProgram> {
        // Apply ScheduleTransforms; check legality via dep_analyzer.
        todo!()
    }
}
```

### 9.2 E-Graph Configuration

```rust
pub struct EGraphConfig {
    pub max_nodes: usize,
    pub max_iterations: usize,
    pub timeout: Duration,
    pub extraction: ExtractionStrategy,
}

#[derive(Debug, Clone)]
pub enum ExtractionStrategy {
    MinCost,
    MinSize,
    Beam { width: usize },
}

impl Default for EGraphConfig {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_iterations: 30,
            timeout: Duration::from_secs(10),
            extraction: ExtractionStrategy::MinCost,
        }
    }
}
```

### 9.3 Search Configuration

```rust
pub struct SearchConfig {
    pub beam_width: usize,
    pub max_iterations: usize,
    pub timeout: Duration,
    /// Pre-seeded tile bindings (e.g., from a prior profile run).
    /// If a SearchParam has no entry here, search explores the full range.
    pub initial_bindings: HashMap<String, i64>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 8,
            max_iterations: 50,
            timeout: Duration::from_secs(30),
            initial_bindings: HashMap::new(),
        }
    }
}
```

---

## 10. Appendix: Lowering Examples

### 10.1 Matmul Lowering

HLIR (after decomposition):
```
Reduce(Sum, Mul(Expand(A, [M, 1, K]), Expand(B, [1, N, K])), axis=2)
```

Annotated:
```rust
SemanticRegion {
    tag: Contraction { m: Sym("M"), n: Sym("N"), k: Sym("K") },
    nodes: [expand_a, expand_b, mul, reduce],
}
```

Algorithmic rewrite applied:
```rust
AlgorithmicRewrite::BlockedMatmul { block_m: 64, block_n: 64, block_k: 8 }
```

Plan IR (post-binding — all tile sizes concrete):
```rust
PlannedRegion {
    schedule: RegionSchedule {
        tiling: Some(TilingSpec {
            tile_sizes: [TileSize::Const(64), TileSize::Const(64), TileSize::Const(8)],
            tile_order: [0, 1, 2],
        }),
        parallelism: ParallelismSpec::Parallel { axis: 0, num_threads: ParamOrConst::Const(8) },
        specialization: Specialization::LoopNest,
    },
    algorithmic_rewrite: Some(AlgorithmicRewrite::BlockedMatmul {
        block_m: 64, block_n: 64, block_k: 8
    }),
    ..
}
```

LLIR (M, N, K remain as RuntimeParameters; tile sizes are concrete):
```rust
Kernel {
    body: LoopNest {
        loops: [
            Loop { var: "m_outer", lower: 0, upper: AffineExpr { terms: [(1, Param("M")), ...] / 64 }, kind: Parallel, step: 1 },
            Loop { var: "n_outer", lower: 0, upper: N/64, kind: Sequential, step: 1 },
            Loop { var: "k_outer", lower: 0, upper: K/8, kind: Sequential, step: 1 },
            Loop { var: "m_inner", lower: 0, upper: 64, kind: Sequential, step: 1 },
            Loop { var: "n_inner", lower: 0, upper: 64, kind: Vectorized { width: 8 }, step: 1 },
            Loop { var: "k_inner", lower: 0, upper: 8, kind: Reduction {
                accumulators: [ReductionAccumulator { var: "acc", op: Sum, init: 0.0, dtype: F32 }]
            }, step: 1 },
        ],
        body: [
            Accumulate {
                dst: C[m_outer*64 + m_inner, n_outer*64 + n_inner],
                op: Sum,
                src: Expr::AbstractVector(AbstractVectorOp::Fma {
                    acc: Load(acc),
                    lhs: Load(A[m_outer*64 + m_inner, k_outer*8 + k_inner]),
                    rhs: Load(B[k_outer*8 + k_inner, n_outer*64 + n_inner]),
                    width: 8,
                }),
            },
            // Epilogue for M % 64 != 0 (if Divisibility::RequiresEpilogue):
            Stmt::Epilogue {
                main_loop_var: "m_outer",
                remainder_body: [ /* scalar tail */ ],
            },
        ],
    },
    provenance: Contraction { m: Sym("M"), n: Sym("N"), k: Sym("K") },
    rewrite: Some(RewriteSpec::BlockedMatmul { block_m: 64, block_n: 64, block_k: 8 }),
}
```

### 10.2 Softmax Lowering (Fused, Online Algorithm)

HLIR:
```
x_max  = Reduce(Max, x, axis=-1)
x_sub  = Sub(x, Expand(x_max))
x_exp  = Exp(x_sub)
x_sum  = Reduce(Sum, x_exp, axis=-1)
y      = Div(x_exp, Expand(x_sum))
```

Annotated (single region):
```rust
SemanticRegion {
    tag: Softmax { axis: -1 },
    nodes: [reduce_max, sub, exp, reduce_sum, div],
}
```

Algorithmic rewrite applied:
```rust
AlgorithmicRewrite::OnlineSoftmax { axis: -1 }
```

This rewrite is non-trivial: it replaces the two-pass structure (max pass, then sum pass) with a single pass that maintains `(running_max, running_sum)` jointly. This cannot be derived from any `ScheduleTransform`; it requires changing what state exists across iterations.

Plan IR (post-binding):
```rust
PlannedRegion {
    schedule: RegionSchedule {
        tiling: Some(TilingSpec {
            tile_sizes: [TileSize::Const(128)],
            tile_order: [0],
        }),
        specialization: Specialization::LoopNest,
    },
    algorithmic_rewrite: Some(AlgorithmicRewrite::OnlineSoftmax { axis: -1 }),
    ..
}
```

LLIR (seq_len remains as RuntimeParameter):
```rust
Kernel {
    body: LoopNest {
        loops: [
            // Row loop — parallel over batch dimension
            Loop { var: "row", lower: 0, upper: AffineExpr::param("batch"), kind: Parallel, step: 1 },
            // Single-pass reduction: joint (running_max, running_sum) accumulators
            Loop { var: "col", lower: 0, upper: AffineExpr::param("seq_len"), kind: Reduction {
                accumulators: [
                    ReductionAccumulator { var: "running_max", op: Max, init: f32::NEG_INFINITY, dtype: F32 },
                    ReductionAccumulator { var: "running_sum", op: Sum, init: 0.0, dtype: F32 },
                ]
            }, step: 1 },
        ],
        body: [
            // Online update: new_max = max(running_max, x[row, col])
            // running_sum = running_sum * exp(running_max - new_max) + exp(x - new_max)
            // (exact update sequence specified by RewriteSpec::OnlineSoftmax)
        ],
    },
    provenance: Softmax { axis: -1 },
    rewrite: Some(RewriteSpec::OnlineSoftmax { axis: -1, tile_size: 128 }),
}
```

---

## 11. Summary

| Layer | Purpose | Search mechanism | Representation |
|-------|---------|-----------------|----------------|
| **HLIR** | Express computation | E-graph: algebraic rewrites + pattern recognition | Tensor ops, symbolic shapes |
| **Plan IR** | Schedule decisions | Beam search: fusion, tiling, parallelism | Fusion groups with schedules; SearchParams pre-binding |
| **LLIR** | Executable form | Local transforms (legality-checked) | Explicit loops, abstract SIMD, concrete tile sizes |

**Key principles:**
1. E-graphs are used only at HLIR — for algebraic optimization and pattern recognition, not schedule search.
2. Schedule search at Plan IR is beam search over a combinatorial space, guided by the hardware cost model.
3. Algorithmic rewrites (online softmax, blocked matmul, Welford) are applied explicitly at Plan IR; they cannot be derived from loop transformations.
4. Fusion is multi-way via `FusionGroup`; `fused_with: Option<RegionId>` is insufficient.
5. All `SearchParam` tile sizes are resolved by the parameter binding pass before LLIR emission; non-divisible cases produce `Epilogue` statements.
6. LLIR SIMD is expressed via `AbstractVectorOp`, not named intrinsics; backends lower to platform specifics.
7. Reduction loops carry `Vec<ReductionAccumulator>` to support joint reductions.
8. Symbols have explicit resolution phases (`CompileTime`, `TuneTime`, `RuntimeParameter`, `RuntimeDerived`); shared memory is constrained to the first two.
9. `seq_len` and similar inference-time shapes remain `RuntimeParameter` throughout  —no recompilation on shape change, provided the tiling remains valid (epilogues handle remainders).
