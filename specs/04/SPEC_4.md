# Venum IR & Scheduling Specification v7
## Backend-Agnostic, Search-First, Polyhedral-Grounded

---

## 1. Philosophy

1. **Search over hand-tuning** — The scheduler discovers good schedules via
   search, not hardcoded heuristics
2. **Primitives over special ops** — No `Attention`, `LayerNorm`, `Matmul` as IR
   nodes; they decompose to primitives
3. **Patterns are data, not code** — Fusion is discovered by search, not
   recognized via hardcoded pattern tags
4. **Backend-agnostic representation** — Scheduling decisions are expressed in a
   neutral IR; backends interpret them

### Non-Goals

- Hand-tuned kernel libraries as the primary path
- ISL as a hard dependency
- Domain-specific ops that encode implementation choices
- Semantic tags of any kind — the HLIR carries no annotations beyond node types

---

## 2. Architecture

```
┌──────────────────────────────────────────────┐
│  HLIR                                        │
│  Tensor ops on symbolic shapes.              │
│  E-graph: algebraic simplification only.     │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │  Schedule Search │
        │                  │
        │  Beam search over│
        │  fusion groups × │
        │  Opt sequences.  │
        └────────┬─────────┘
                  │ ScheduleDecision
                  ▼
┌──────────────────────────────────────────────┐
│  LLIR                                        │
│  Explicit loop nests, affine access maps,    │
│  abstract SIMD, concrete values throughout.  │
│  Polyhedral dependence analysis lives here.  │
└─────────────────┬────────────────────────────┘
                  │ codegen
                  ▼
      CUDA / Metal / CPU SIMD / WGSL
```

Two IRs. The middle layer is not an IR — it is a `ScheduleDecision`: a record of
which nodes are fused together and which loop transformations are applied to each
resulting kernel. Search produces it; the lowerer consumes it directly.

---

## 3. Symbol Resolution

```rust
pub enum SymbolResolution {
    CompileTime(i64),  // fixed before compilation: rank, dtype
    RuntimeParam,      // provided at dispatch: N, M, K, seq_len
    RuntimeDerived,    // computed from RuntimeParams: P_NM = N*M for Reshape
}
```

Tile sizes and all loop transformation parameters are concrete `i64` values
inside `ScheduleDecision`. They are never symbols. This eliminates any binding or
resolution pass between search and lowering.

| Tier                | Allowed                                           |
|---------------------|---------------------------------------------------|
| HLIR                | `RuntimeParam`, `RuntimeDerived`                  |
| LLIR                | All three; loop params folded to integer constants |
| Shared memory sizes | `CompileTime` only, or concrete values from opts  |

Non-affine `Dim` forms (`Mul(Sym,Sym)`, `Div(Sym,Sym)`, `Mod(Sym,Sym)`) are valid
at HLIR but normalized to `RuntimeDerived` parameters before lowering. Derived
parameters are computed in the kernel prologue at dispatch time.

---

## 4. HLIR

### 4.1 Types

```rust
pub enum Dim {
    Const(i64),
    Sym(Symbol),
    Add(Box<Dim>, Box<Dim>),
    Mul(Box<Dim>, Box<Dim>),
    Div(Box<Dim>, Box<Dim>),
    Mod(Box<Dim>, Box<Dim>),
}

pub struct TensorType {
    pub shape:  Vec<Dim>,
    pub dtype:  DType,
    pub layout: Layout,   // Contiguous | Strided(Vec<Dim>) | View { base, offset, strides }
}

pub enum DType { F32, F16, BF16, F64, I8, I16, I32, I64, U8, U16, U32, U64, Bool }
```

### 4.2 Primitive Operations

```rust
pub enum Op {
    // Memory
    Const { value: Scalar, shape: Vec<Dim>, dtype: DType },
    Load  { buffer: BufferId },
    Store { buffer: BufferId, value: NodeId },

    // Unary
    Neg(NodeId), Recip(NodeId), Exp(NodeId), Log(NodeId),
    Sqrt(NodeId), Sin(NodeId), Cos(NodeId),
    Cast { input: NodeId, to: DType },

    // Binary
    Add(NodeId, NodeId), Mul(NodeId, NodeId),
    Max(NodeId, NodeId), Min(NodeId, NodeId),
    Cmp { op: CmpOp, lhs: NodeId, rhs: NodeId },

    // Ternary
    Where { cond: NodeId, then_val: NodeId, else_val: NodeId },

    // Reduction
    Reduce { input: NodeId, axes: Vec<usize>, op: ReduceOp, keepdim: bool },

    // Shape
    Reshape { input: NodeId, shape: Vec<Dim> },
    Permute { input: NodeId, axes: Vec<usize> },
    Slice   { input: NodeId, ranges: Vec<Range> },
    Expand  { input: NodeId, shape: Vec<Dim> },
    Concat  { inputs: Vec<NodeId>, axis: usize },
}
```

Derived operations — never primitive ops:

| Operation   | Decomposition |
|-------------|---------------|
| `Sub(a,b)`  | `Add(a, Neg(b))` |
| `Div(a,b)`  | `Mul(a, Recip(b))` |
| `Matmul`    | `Reduce(Sum, Mul(Expand(A), Expand(B)), axis=-1)` |
| `Softmax`   | `Div(Exp(Sub(x, Reduce(Max,x))), Reduce(Sum, Exp(Sub(x, Reduce(Max,x)))))` |
| `LayerNorm` | `Div(Sub(x, mean), Sqrt(Add(var, eps)))` |

### 4.3 E-Graph Integration

E-graphs operate **only at HLIR** for algebraic simplification. Typical rules:
`Add(x, Const(0)) ↔ x`, `Neg(Neg(x)) ↔ x`, `Exp(Log(x)) ↔ x`
(domain-restricted), commutativity for cost normalization. Saturation runs to a
node or iteration budget; extraction minimizes operation count.

There are no semantic annotations. The e-graph produces a simplified `HLIRGraph`
— nothing more. Fusion, tiling, and all scheduling decisions are made downstream
by search, operating on node structure alone.

---

## 5. Schedule Search

### 5.1 The Opt

A single loop transformation applied to a kernel. Directly analogous to tinygrad's
`Opt`. All parameters are concrete integers — no symbolic search parameters.

```rust
pub struct Opt {
    pub op:   OptOp,
    pub axis: usize,  // which loop dimension to target
    pub amt:  i64,    // transformation parameter
}

pub enum OptOp {
    /// Split loop at axis into (outer: N/amt, inner: amt).
    Tile,
    /// Each thread processes amt contiguous elements along axis.
    /// Maps to Vectorized loop in LLIR; emits AbstractVectorOp in body.
    Vectorize,
    /// Inline the loop body amt times; eliminates the loop variable.
    Unroll,
    /// Move axis to thread level (threadIdx / lid).
    Parallelize,
    /// Shared memory reduction: split reduce axis into parallel threads,
    /// stage through threadgroup memory, barrier, then reduce.
    GroupReduce,
    /// Pad loop bound to next multiple of amt; guard body with conditional.
    PadTo,
}
```

Opts are applied in sequence. The order matters: `Tile` then `Vectorize` on the
inner loop is the standard vectorized-tiled pattern.

### 5.2 Schedule Decision

The output of search. Contains everything the lowerer needs.

```rust
pub struct ScheduleDecision {
    /// Ordered list of kernels to emit. Each group is one kernel.
    pub fusion_groups: Vec<FusionGroup>,
    /// Opt sequence per kernel.
    pub opts: HashMap<FusionGroupId, Vec<Opt>>,
}

pub struct FusionGroup {
    pub id:       FusionGroupId,
    pub nodes:    Vec<NodeId>,      // topologically ordered HLIR nodes
    pub topology: FusionTopology,
}

pub enum FusionTopology {
    Chain,
    FanIn  { consumer: NodeId, producers: Vec<NodeId> },
    FanOut { producer: NodeId, consumers: Vec<NodeId> },
    DAG    { edges: Vec<(NodeId, NodeId)> },
}
```

### 5.3 Fusion Legality

A fusion group is legal iff:
- No node within the group has an output consumed outside the group except through
  the group's designated outputs.
- Reduction boundaries are respected: a `Reduce` node can only be fused with its
  producer if the full reduction domain is preserved in the fused loop nest.

Legality is checked before search commits to a fusion candidate.

### 5.4 Search Algorithm

Beam search over the space of valid `ScheduleDecision` values. The search space
is the product of fusion candidates × `Vec<Opt>` sequences per group. Opt
candidates are generated from the `HardwareModel` — axis ranges, valid amounts,
and legality constraints (e.g., `Vectorize` is invalid on a reduce axis).

```rust
pub struct ScheduleSearcher<H: HardwareModel> {
    pub hardware:       H,
    pub beam_width:     usize,
    pub max_iterations: usize,
}

impl<H: HardwareModel> ScheduleSearcher<H> {
    pub fn search(
        &self,
        hlir: &AnnotatedHLIR,
    ) -> Vec<(ScheduleDecision, CostEstimate)>;
}
```

The cost oracle is `HardwareModel::estimate_cost`. Returns candidates ranked by
estimated cost. For a tuned deployment, actual hardware measurement can retrain
the cost model (XGBoost surrogate, Ansor-style), improving future searches on the
same device.

---

## 6. LLIR

LLIR is the committed schedule. All values are concrete. Runtime symbols appear
only as `Var::Param` in affine expressions, passed as kernel arguments at
dispatch.

### 6.1 Loop Representation

```rust
pub struct LoopNest {
    pub loops: Vec<Loop>,
    pub body:  Vec<Stmt>,
}

pub struct Loop {
    pub var:         String,
    pub lower:       AffineExpr,
    pub upper:       AffineExpr,
    pub step:        i64,
    pub kind:        LoopKind,
    pub annotations: LoopAnnotations,
}

pub enum LoopKind {
    /// Sequential execution; single-threaded loop.
    Sequential,
    /// Backend chooses concurrency substrate (threads, work-items, GPU lanes).
    /// The backend maps this to GPU launch dimensions or CPU parallel forges.
    Parallel,
    /// Fixed-width SIMD/lane parallelism. Emits AbstractVectorOp in body.
    /// The width must be realizable on the target (e.g., 8, 16, 32 for SIMD).
    Vectorized { width: usize },
    /// Fully unrolled at compile time; loop eliminated, body replicated.
    Unrolled   { factor: usize },
    /// Reduction loop: partial results accumulated in accumulators.
    /// Scalar loop over reduction dimension after parallel load + barrier.
    Reduction  { accumulators: Vec<ReductionAccumulator> },
}

pub struct ReductionAccumulator {
    pub var:   String,
    pub op:    ReduceOp,
    pub init:  Scalar,
    pub dtype: DType,
}
```

`GroupReduce` opt produces a `Parallel` loop with a `Shared` buffer staging area
and a `Stmt::Barrier`, followed by a scalar `Reduction` loop in thread 0.

### 6.2 Affine Expressions

```rust
/// c0 + c1·v1 + c2·v2 + ...
/// Loop transformation parameters fold to integer constants; no symbolic tile sizes.
pub struct AffineExpr {
    pub constant: i64,
    pub terms:    Vec<(i64, Var)>,
}

pub enum Var {
    Loop(String),   // loop iteration variable
    Param(Symbol),  // RuntimeParam or RuntimeDerived
}
```

### 6.3 Memory

```rust
pub struct MemoryAccess {
    pub buffer:      BufferId,
    pub indices:     Vec<AffineExpr>,
    pub access_kind: AccessKind,      // Read | Write | ReadWrite
}

pub struct BufferAlloc {
    pub id:           BufferId,
    pub shape:        Vec<AffineExpr>,
    pub dtype:        DType,
    pub memory_space: MemorySpace,    // Global | Shared | Local | Constant
}
```

Shared buffers must have shapes that evaluate to concrete integers at lowering
time. `RuntimeParam` terms in a `Shared` buffer shape are a lowering error.

### 6.4 Statements and Expressions

```rust
pub enum Stmt {
    Assign     { dst: MemoryAccess, src: Expr },
    Accumulate { dst: MemoryAccess, op: ReduceOp, src: Expr },
    If         { cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt> },
    Loop       (Loop, Vec<Stmt>),
    Barrier    { scope: BarrierScope },           // __syncthreads / threadgroup_barrier
    Epilogue   { main_loop_var: String, remainder_body: Vec<Stmt> },
}

pub enum Expr {
    Literal(Scalar),
    Load(MemoryAccess),
    Unary   { op: UnaryOp,  arg: Box<Expr> },
    Binary  { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Ternary { cond: Box<Expr>, then_val: Box<Expr>, else_val: Box<Expr> },
    Cast    { arg: Box<Expr>, to: DType },
    /// Backend-agnostic SIMD. Does not name a platform intrinsic.
    /// Backends lower to _mm256_fmadd_ps, vfmadd, wmma, etc.
    AbstractVector(AbstractVectorOp),
}

pub enum AbstractVectorOp {
    Fma              { acc: Box<Expr>, lhs: Box<Expr>, rhs: Box<Expr>, width: usize },
    HorizontalReduce { op: ReduceOp, arg: Box<Expr>, width: usize },
    Broadcast        { scalar: Box<Expr>, width: usize },
    Gather           { base: BufferId, indices: Box<Expr>, width: usize },
    Scatter          { base: BufferId, indices: Box<Expr>, value: Box<Expr>, width: usize },
    VecBinary        { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr>, width: usize },
    VecCast          { arg: Box<Expr>, from: DType, to: DType, width: usize },
}
```

`AbstractVectorOp::Fma` is emitted by the lowerer when it encounters a
`Vectorized` loop over a `Mul`-then-`Reduce(Sum)` pattern in the HLIR body —
no semantic tag required.

### 6.5 Dependences

```rust
pub struct Dependence {
    pub from:     StmtId,
    pub to:       StmtId,
    pub kind:     DepKind,           // RAW | WAR | WAW
    pub distance: Option<Vec<i64>>,
    pub relation: DependenceRelation,
}

/// { [source_iters] → [sink_iters] : affine constraints }
pub struct DependenceRelation {
    pub source_vars: Vec<String>,
    pub sink_vars:   Vec<String>,
    pub constraints: Vec<AffineConstraint>,
}

pub struct AffineConstraint {
    pub expr: AffineExpr,
    pub kind: ConstraintKind,        // Eq (expr=0) | Ge (expr≥0)
}
```

---

## 7. Polyhedral Model

The polyhedral model grounds dependence analysis and schedule legality at LLIR.
Encapsulated behind `DependenceAnalyzer` (§9.1); ISL is the reference
implementation, not a hard dependency.

### 7.1 Iteration Domains and Access Maps

```rust
pub struct Aff {
    pub constant: i64,
    pub terms:    Vec<(i64, PolyVar)>,
}

pub enum PolyVar { Iter(String), Param(Symbol) }

pub enum Constraint { Eq(Aff), Ineq(Aff) }

/// { [i0, ..., in] : constraints }
pub struct Domain {
    pub iters:       Vec<String>,
    pub params:      Vec<Symbol>,
    pub constraints: Vec<Constraint>,
}

/// { [i0, ..., in] → [addr] : addr = s0·i0 + s1·i1 + ... }
pub struct AccessMap {
    pub domain_iters: Vec<String>,
    pub mapping:      Vec<Aff>,
}
```

### 7.2 Constructing from Tensor Metadata

**`shape_to_domain`** converts a tensor shape to an iteration domain. Each
dimension at index `i` introduces iterator `i{i}` and two constraints:
`i{i} ≥ 0` and `dim - 1 - i{i} ≥ 0`. Called after non-affine `Dim`
normalization.

```
shape [N, M] → iters: ["i0","i1"], params: ["N","M"]
               constraints: [i0≥0, N-1-i0≥0, i1≥0, M-1-i1≥0]
```

**`strides_to_access`** converts a stride vector to an `AccessMap` over a domain.
Strides must be `Dim::Const` — symbolic strides produce non-affine `Param×Iter`
products and are a normalization error.

```
strides [128, 1], iters ["i0","i1"]
  → mapping: Aff { constant: 0, terms: [(128, Iter("i0")), (1, Iter("i1"))] }
```

### 7.3 ISL Integration

ISL is used for three purposes only:

| Purpose | ISL operation |
|---------|---------------|
| Dependence computation | `dep = R ∘ W⁻¹` |
| Legality checking | Verify a transform violates no dependence |
| AST generation | Emit a loop nest from a polyhedral schedule (optional) |

Minimal FFI surface:

```rust
extern "C" {
    fn isl_ctx_alloc() -> *mut IslCtx;
    fn isl_ctx_free(ctx: *mut IslCtx);
    fn isl_union_map_read_from_str(ctx: *mut IslCtx, s: *const i8) -> *mut IslUnionMap;
    fn isl_union_map_apply_range(m1: *mut IslUnionMap, m2: *mut IslUnionMap) -> *mut IslUnionMap;
    fn isl_union_map_reverse(m: *mut IslUnionMap) -> *mut IslUnionMap;
    fn isl_union_map_is_empty(m: *mut IslUnionMap) -> i32;
    fn isl_union_map_free(m: *mut IslUnionMap);
    fn isl_ast_build_alloc(ctx: *mut IslCtx) -> *mut IslAstBuild;
    fn isl_ast_build_node_from_schedule(b: *mut IslAstBuild, s: *mut IslSchedule) -> *mut IslAstNode;
}
```

`IslDependenceAnalyzer` holds an `isl_ffi::Ctx`, serializes `AccessMap`s to ISL
strings (e.g. `"{ S[i0,i1] -> A[128*i0 + i1] }"`), and implements the three
operations above. `isl_ctx` is not `Send`; confine to the lowering thread.

---

## 8. Lowering Pipeline

```rust
pub fn compile<H, D, C>(
    hlir:   HLIRGraph,
    hw:     &H,
    dep:    &D,
    cg:     &C,
    config: &SearchConfig,
) -> Result<C::Output>
where
    H: HardwareModel,
    D: DependenceAnalyzer,
    C: CodeGenerator,
{
    let hlir     = optimize_hlir(hlir)?;           // e-graph saturation + extraction
    let decision  = search(&hlir, hw, config)?;
    let llir      = lower(&hlir, &decision, dep)?;
    let llir      = optimize_llir(llir, dep)?;
    cg.generate(&llir)
}
```

### HLIR Optimization

E-graph saturation with algebraic rewrite rules. Extraction minimizes operation
count. No fusion or scheduling decisions are made here.

### Schedule Search

Enumerate valid fusion candidates and `Opt` sequences. Run beam search with
`HardwareModel::estimate_cost` as oracle. Return ranked `(ScheduleDecision,
CostEstimate)` list.

### Lowering

For each `FusionGroup` in topological order:

1. **Build iteration domain and access maps** — call `shape_to_domain` on the
   output tensor shape; `strides_to_access` for each input and output. Normalize
   any non-affine `Dim` forms first.

2. **Emit base loop nest** — derive sequential loops from domain bounds, one per
   iterator. This is the unoptimized baseline.

3. **Apply Opts in sequence** — for each `Opt` in `decision.opts[group_id]`:
   - `Tile { axis, amt }` — split loop at `axis` into outer `(N/amt)` and inner
     `(amt)`. Check legality via `dep_analyzer`. Emit `Stmt::Epilogue` if `N % amt
     != 0`.
   - `Vectorize { axis, amt }` — mark innermost loop `LoopKind::Vectorized {
     width: amt }`. Detect `Mul`-then-`Reduce(Sum)` in body and emit
     `AbstractVectorOp::Fma`; otherwise emit `VecBinary`.
   - `Unroll { axis, amt }` — mark loop `LoopKind::Unrolled { factor: amt }`.
   - `Parallelize { axis, amt }` — split loop; mark outer loop as `Parallel`.
      The backend maps this to `GridDim` or `BlockDim` based on GPU launch config.
   - `GroupReduce { axis, amt }` — split reduce axis; emit `BufferAlloc { memory:
     Shared }` staging area, `Parallel` loop for parallel load, `Stmt::Barrier`,
     then scalar `Reduction` loop in thread 0.
   - `PadTo { axis, amt }` — extend loop bound to `ceil(N/amt)*amt`; wrap body
     in `Stmt::If` guarding against out-of-bounds.

4. **Merge fused nests** — `Chain` groups share outer loops up to the depth where
   iteration domains align; producer body inlined into consumer nest. `FanIn`
   groups emit all producers before the consumer loop. `DAG` groups follow
   topological order with materialization at edges crossing loop boundaries.

### LLIR Optimization

Apply additional `ScheduleTransform`s (interchange, cache read/write) to the
committed loop nests. Each is checked via `dep_analyzer` before application.
Cross-kernel dependences are computed here by calling `analyze_kernel` on kernel
pairs sharing a buffer.

---

## 9. Backend Traits

### 9.1 Dependence Analysis

```rust
pub trait DependenceAnalyzer {
    fn analyze_kernel(&self, kernel: &Kernel) -> Result<Vec<Dependence>>;

    fn check_legality(
        &self,
        deps:      &[Dependence],
        transform: &ScheduleTransform,
    ) -> Result<bool>;

    fn access_dependence(
        &self,
        write: &MemoryAccess,
        read:  &MemoryAccess,
        loops: &[Loop],
    ) -> Result<Option<DependenceRelation>>;
}
```

### 9.2 Schedule Transforms

```rust
pub enum ScheduleTransform {
    Tile        { loop_var: String, factor: i64 },
    Interchange { outer: String, inner: String },
    Parallelize { loop_var: String, kind: ParallelKind },
    Unroll      { loop_var: String, factor: usize },
    Vectorize   { loop_var: String, width: usize },
    ComputeAt   { producer: KernelId, consumer: KernelId, loop_var: String },
    CacheRead   { buffer: BufferId, at_loop: String, memory: MemorySpace },
    CacheWrite  { buffer: BufferId, at_loop: String, memory: MemorySpace },
}
```

### 9.3 Code Generation

```rust
pub trait CodeGenerator {
    type Output;
    fn generate(&self, program: &LLIRProgram) -> Result<Self::Output>;
}

pub trait CpuCodeGen: CodeGenerator<Output = CpuModule> {
    /// Lower AbstractVectorOp to a platform intrinsic string.
    fn lower_vector_op(&self, op: &AbstractVectorOp, dtype: DType) -> String;
}

pub trait GpuCodeGen: CodeGenerator<Output = GpuModule> {
    fn warp_size(&self) -> usize;
    fn max_shared_memory(&self) -> usize;
    fn validate_shared_memory(&self, program: &LLIRProgram) -> Result<()>;
}
```

---

## 10. Cost Model

```rust
pub trait HardwareModel {
    fn memory_levels(&self)  -> &[MemoryLevel];
    fn compute(&self)        -> &ComputeCapabilities;
    fn estimate_cost(&self, kernel: &Kernel) -> CostEstimate;
    /// Valid Opt instances for a given kernel shape and axis.
    /// Constrains the search space to hardware-meaningful choices.
    fn opt_candidates(&self, shape: &[i64], dtype: DType) -> Vec<Opt>;
}

pub struct CostEstimate {
    pub compute_cycles:       f64,
    pub memory_cycles:        f64,
    pub total_cycles:         f64,   // max(compute, memory) — roofline
    pub working_set_bytes:    usize,
    pub arithmetic_intensity: f64,
    pub bottleneck:           Bottleneck,
}
```

`opt_candidates` replaces `tile_size_candidates` from earlier drafts. The
hardware model produces valid `Opt` instances directly, including axis ranges and
valid amounts, so the searcher never proposes illegal or pointless transforms.

---

## 11. Appendix: Lowering Examples

### 11.1 Matmul

**HLIR:** `Reduce(Sum, Mul(Expand(A,[M,1,K]), Expand(B,[1,N,K])), axis=2)`

**ScheduleDecision opts:**
```
Tile      { axis: 0, amt: 64 }   // tile M
Tile      { axis: 1, amt: 64 }   // tile N
Tile      { axis: 2, amt: 8  }   // tile K
Parallelize { axis: 0, amt: 64 } // m_outer → Parallel (GPU: GridDim)
Vectorize { axis: 4, amt: 64 }   // n_inner → Vectorized
```

**LLIR** (lowerer detects Mul+Reduce(Sum) under Vectorized loop and emits Fma):
```
for m_outer in [0, M/64)    [Parallel]
  for n_outer in [0, N/64)  [Sequential]
    for k_outer in [0, K/8) [Sequential]
      for m_inner in [0, 64) [Sequential]
        for n_inner in [0, 64) [Vectorized(64)]
          for k_inner in [0, 8) [Reduction{acc: Sum, init: 0}]
            C[m_outer*64+m_inner, n_outer*64+n_inner] +=
              Fma(A[m_outer*64+m_inner, k_outer*8+k_inner],
                  B[k_outer*8+k_inner, n_outer*64+n_inner], acc)
```

Epilogues emitted for M%64, N%64, K%8 where required.

### 11.2 Softmax (search-discovered fusion)

**HLIR:**
```
x_max = Reduce(Max, x, axis=-1)
x_sub = Sub(x, Expand(x_max))
x_exp = Exp(x_sub)
x_sum = Reduce(Sum, x_exp, axis=-1)
y     = Div(x_exp, Expand(x_sum))
```

There is no `Softmax` tag. Search discovers that fusing all five nodes into one
kernel is cheaper than materializing intermediate buffers — the cost model sees
reduced memory traffic. The fused kernel's loop structure is determined entirely
by the `Opt` sequence search selects; no hardcoded two-pass or online algorithm
is prescribed.

If search finds a `Tile + GroupReduce` opt sequence on the reduce axis, the
emitted LLIR will use shared memory staging. If it finds a simple `Sequential`
reduction, it will emit a scalar loop. Both are valid outcomes. The compiler
does not know what softmax is.
