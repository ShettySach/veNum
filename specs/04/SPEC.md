# Venum IR & Scheduling Specification
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
    pub layout: Layout,   // Contiguous | Strided(Vec<Dim>) | View { offset, strides }
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
    pub axis: usize,  // index into the current loop nest at the time this opt is applied
    pub amt:  i64,    // transformation parameter; meaning is op-specific (see below)
}

pub enum OptOp {
    /// Split loop at axis into (outer: ceil(N/amt), inner: amt).
    /// amt = inner loop extent (tile size).
    /// Inserts inner loop at axis+1; all loops at axis+1 and beyond shift up by one.
    Tile,
    /// Mark loop at axis as Vectorized { width: amt }.
    /// amt = SIMD vector width in elements.
    /// Loop body must be over a Spatial axis; illegal on Reduce axes.
    /// No new loops inserted; no index shift.
    Vectorize,
    /// Mark loop at axis as Unrolled { factor: amt }.
    /// amt = unroll factor; loop variable is eliminated.
    /// No new loops inserted; no index shift.
    Unroll,
    /// Split loop at axis into (outer: ceil(N/amt), inner: amt),
    /// then mark the outer loop as Parallel.
    /// amt = per-thread chunk size (inner loop extent).
    /// Inserts inner loop at axis+1; all loops at axis+1 and beyond shift up by one.
    /// The outer loop at axis becomes the parallel dimension.
    Parallelize,
    /// Split the Reduce loop at axis into (parallel_outer: amt threads, serial_inner: ceil(N/amt)),
    /// stage partial results through a Shared buffer, emit Barrier, then reduce partials.
    /// amt = number of parallel threads in the reduction.
    /// Inserts inner loop at axis+1; shifts subsequent indices.
    /// Only legal on Reduce-kinded loops.
    GroupReduce,
    /// Extend loop at axis to next multiple of amt; wrap body in a conditional guard.
    /// amt = alignment factor.
    /// No new loops inserted; no index shift.
    PadTo,
}
```

### 5.1.1 Axis Coordinate Rule

`axis` always refers to the loop nest **as it exists after all preceding opts
in the sequence have been applied**. The nest starts in canonical order: one
loop per output dimension, outermost first, followed by reduction axes.

Opts that insert a new loop (`Tile`, `Parallelize`, `GroupReduce`) place the
inner loop immediately after the targeted loop. Every loop at a higher index
shifts up by one. Opts that do not insert loops (`Vectorize`, `Unroll`,
`PadTo`) modify the targeted loop in-place; no index shift occurs.

**Worked example — matmul, starting from `[M(0), N(1), K(2)]`:**

```
Tile { axis: 0, amt: 64 }
  → [m_outer(0), m_inner(1), N(2), K(3)]

Tile { axis: 2, amt: 64 }       // axis 2 is now N
  → [m_outer(0), m_inner(1), n_outer(2), n_inner(3), K(4)]

Tile { axis: 4, amt: 8 }        // axis 4 is now K
  → [m_outer(0), m_inner(1), n_outer(2), n_inner(3), k_outer(4), k_inner(5)]

Parallelize { axis: 0, amt: 64 } // split m_outer; inner at axis 1, rest shift
  → [m_par_outer(0), m_par_inner(1), m_inner(2), n_outer(3), n_inner(4), k_outer(5), k_inner(6)]
  // m_par_outer is Parallel

Vectorize { axis: 4, amt: 64 }  // axis 4 is n_inner
  → n_inner becomes Vectorized { width: 64 }
```

Search generates `axis` values that account for prior shifts. The lowerer
applies opts strictly in sequence, updating its internal loop index after each
insertion.

### 5.1.2 Opt Sequence Legality

Individual opts are filtered by `HardwareModel::opt_candidates` based on
`KernelContext` (axis role, available shared memory, backend class). However,
individually valid opts can form a collectively illegal sequence — for example,
tiling then interchanging the resulting loops may violate a dependence that
tiling alone respects.

The lowerer therefore checks legality **after each opt application**, not only
after fusion:

```rust
fn apply_opts(
    nest:     LoopNest,
    opts:     &[Opt],
    dep:      &impl DependenceAnalyzer,
) -> Result<LoopNest, LoweringError> {
    let mut nest = nest;
    let mut deps = dep.analyze_kernel(&nest.as_kernel())?;
    for opt in opts {
        nest = apply_opt(nest, opt)?;
        // Re-check all dependences after each transformation.
        // Fail fast: return LegalityViolation on the first illegal opt.
        if !dep.check_legality(&deps, &opt.as_transform())? {
            return Err(LoweringError::LegalityViolation(opt.clone()));
        }
        // Recompute deps over the updated nest for the next iteration.
        deps = dep.analyze_kernel(&nest.as_kernel())?;
    }
    Ok(nest)
}
```

`compile` catches `LegalityViolation` and falls back to the next-ranked
candidate (see §8). Search should treat a legality failure as a signal to
prune that region of the search space.

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
    /// Returns all candidates ranked by estimated cost, best first.
    /// The caller is responsible for selecting among them.
    /// Typical usage: take the first candidate for compilation, fall back
    /// to subsequent candidates if hardware validation fails.
    pub fn search(
        &self,
        hlir: &HLIRGraph,
    ) -> Vec<(ScheduleDecision, CostEstimate)>;

    /// Convenience wrapper: runs search and returns the best candidate.
    /// Equivalent to search(...).into_iter().next().ok_or(SearchError::Empty).
    pub fn search_best(
        &self,
        hlir: &HLIRGraph,
    ) -> Result<(ScheduleDecision, CostEstimate)>;
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
    /// Single-threaded sequential execution.
    Sequential,
    /// Iterations are independent; the backend chooses the concurrency substrate.
    /// CPU backend: parallel-for (e.g. rayon). GPU backend: thread launch.
    /// WGSL backend: workgroup dispatch. The LLIR makes no further distinction —
    /// mapping to blockIdx / threadIdx / local_id is a backend concern.
    Parallel,
    /// Fixed-width SIMD: each logical iteration processes `width` elements.
    /// The loop body must contain only AbstractVectorOp expressions —
    /// the lowerer lifts scalar body ops into the appropriate variant
    /// (Fma, VecBinary, etc.) based on the HLIR body structure.
    Vectorized { width: usize },
    /// Loop fully unrolled at compile time; loop variable eliminated.
    Unrolled { factor: usize },
    /// Reduction loop: carries explicit accumulator state across iterations.
    /// Distinct from Op::Reduce (HLIR tensor op) and Stmt::Accumulate
    /// (per-iteration update). This variant describes the loop's *shape* —
    /// it initialises each accumulator before the loop and finalises after.
    Reduce { accumulators: Vec<ReductionAccumulator> },
}

pub struct ReductionAccumulator {
    pub var:   String,
    pub op:    ReduceOp,
    pub init:  Scalar,
    pub dtype: DType,
}
```

`GroupReduce` opt produces a `Parallel` loop with a `Shared` buffer staging area
and a `Stmt::Barrier`, followed by a scalar `Reduce` loop in the designated executor.

**Reduction across the three tiers:** `Op::Reduce` in HLIR is the tensor-level
operation (what to reduce and along which axes). `LoopKind::Reduce` in LLIR
describes the loop's shape — it carries accumulator initialisation and
finalisation. `Stmt::Accumulate` is the per-iteration update inside that loop
(`dst op= src`). They are three distinct concepts at three distinct levels; the
naming reflects their roles rather than collapsing them.

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
Strides must be `Dim::Const` at the point this function is called — symbolic
strides (`Dim::Sym`, `Dim::Mul(Sym, Sym)`, etc.) produce non-affine
`Param×Iter` products which are not representable in `AffineExpr`.

This is a **lowering restriction**, not a general property of `TensorType`.
`TensorType.layout` may carry `Strided(Vec<Dim>)` with symbolic strides at
HLIR — this is valid. The normalization step in §8 lowering (step 1: "normalize
any non-affine `Dim` forms") must concretize or `RuntimeDerived`-parametrize
all stride expressions before `strides_to_access` is called. Symbolic strides
that cannot be reduced to constants or affine Param expressions at that point
are a lowering error: the tensor requires a `Contiguous` materialization barrier
before entering the polyhedral model.

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
    let hlir      = optimize_hlir(hlir)?;           // e-graph saturation + extraction
    let candidates = search(&hlir, hw, config)?;    // ranked Vec<(ScheduleDecision, CostEstimate)>
    // Walk candidates in cost order. Lower each; skip on legality error.
    // Returns the first that passes lowering and hardware validation.
    // Returns Err if no candidate survives.
    for (decision, _cost) in candidates {
        match lower(&hlir, &decision, dep) {
            Ok(llir) => {
                let llir = optimize_llir(llir, dep)?;
                return cg.generate(&llir);
            }
            Err(LoweringError::LegalityViolation(_)) => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(CompileError::NoValidSchedule)
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
   any non-affine `Dim` forms first (see §7.2 for the normalization requirement
   on symbolic strides).

2. **Emit base loop nest** — derive sequential loops from domain bounds, one per
   iterator. This is the unoptimized baseline.

3. **Apply Opts in sequence** — see §5.1.1 for the axis coordinate rule and
   §5.1.2 for incremental legality checking. Per-opt lowering:
   - `Tile { axis, amt }` — split loop at `axis` into outer `(ceil(N/amt))` and
     inner `(amt)`. Inner loop inserted at `axis+1`. Emit `Stmt::Epilogue` if
     `N % amt != 0`.
   - `Vectorize { axis, amt }` — mark loop at `axis` as `LoopKind::Vectorized {
     width: amt }`. Detect `Mul`-then-`Reduce(Sum)` in body and emit
     `AbstractVectorOp::Fma`; otherwise emit `VecBinary`.
   - `Unroll { axis, amt }` — mark loop `LoopKind::Unrolled { factor: amt }`.
   - `Parallelize { axis, amt }` — split loop at `axis` into outer
     `(ceil(N/amt))` and inner `(amt)`; mark outer as `Parallel`. Inner loop
     inserted at `axis+1`.
   - `GroupReduce { axis, amt }` — see §5.4 below for the exact LLIR sequence
     and barrier placement guarantee.
   - `PadTo { axis, amt }` — extend loop bound to `ceil(N/amt)*amt`; wrap body
     in `Stmt::If` guarding against out-of-bounds.

4. **Merge fused nests** — lowering varies by `FusionTopology`:

   **`Chain`**: producer and consumer share the same iteration domain or a
   subset thereof. Merge by aligning outer loops up to the depth where domains
   agree. Producer body is inlined into the consumer nest at that depth;
   the producer's output buffer is scoped to the shared loop level and may
   be scalarized if its lifetime does not escape the fused nest.

   **`FanIn { consumer, producers }`**: each producer has a domain that is
   compatible with the consumer's domain (same shape or broadcastable). Lower
   each producer as a separate loop nest in topological order, materializing
   their outputs into temporary buffers. Then lower the consumer loop nest,
   reading from those temporaries. Temporary buffers live in `Local` or `Shared`
   memory depending on size relative to the shared memory budget; if both are
   exceeded, fall back to `Global`. Producers with identical domains may be
   merged into a single loop nest before the consumer (i.e., treated as a Chain
   among themselves) if their dependences permit.

   **`FanOut { producer, consumers }`**: the producer is lowered once. Each
   consumer reads from the single materialized producer output. Consumers are
   lowered in topological order as separate loop nests. No inlining — the
   producer result is always materialized because multiple consumers require it.

   **`DAG { edges }`**: generalization of the above. Process nodes in
   topological order. For each edge `(producer, consumer)`:
   - If the edge is **internal** (both producer and consumer are in the same
     fused group and the producer result is not consumed outside the group):
     treat as Chain — inline producer at the appropriate loop depth in the
     consumer nest, scalarize the intermediate buffer if possible.
   - If the edge **crosses a loop boundary** (the producer's iteration domain
     does not align with the consumer's at any loop level, or the producer is
     consumed by more than one node): materialize the producer's output into a
     temporary buffer. The consumer reads from that buffer. A loop boundary
     crossing is defined as: the producer's output shape differs from the
     consumer's input shape at one or more dimensions after accounting for
     broadcast, or the producer has already been inlined into a different
     consumer at a loop level incompatible with this consumer's loop structure.
   Temporary buffers introduced by DAG materialization follow the same memory
   space selection rule as FanIn.

### LLIR Optimization

Apply additional `ScheduleTransform`s (interchange, cache read/write) to the
committed loop nests. Each is checked via `dep_analyzer` before application.
Cross-kernel dependences are computed here by calling `analyze_kernel` on kernel
pairs sharing a buffer.

### 8.1 GroupReduce LLIR Expansion

`GroupReduce { axis, amt }` on a Reduce loop at `axis` with bound `N` produces
the following LLIR sequence. The barrier placement is a **guarantee** of this
opt — the lowerer always emits exactly this structure:

```
// 1. Allocate shared staging buffer: shape [amt], dtype = accumulator dtype.
BufferAlloc { id: staging, shape: [amt], memory: Shared }

// 2. Parallel load: each of the `amt` threads computes a partial reduction
//    over its assigned slice of the reduction axis (ceil(N/amt) elements).
Loop { var: "t", lower: 0, upper: amt, kind: Parallel }
  Loop { var: "k", lower: t*(N/amt), upper: min((t+1)*(N/amt), N), kind: Sequential }
    Accumulate { dst: staging[t], op: reduce_op, src: body_expr }

// 3. Barrier: all threads must have written to staging before any thread reads.
//    This barrier is ALWAYS emitted here, between the parallel write and the
//    serial read. It is never elided, even if amt == 1.
Barrier { scope: Shared }

// 4. Serial reduction: a single designated thread reduces the staging buffer.
//    Emitted as a Sequential loop; the backend maps to thread 0 or a warp reduce.
Loop { var: "r", lower: 0, upper: amt, kind: Sequential,
       kind: Reduce { accumulators: [{ var: "acc", op: reduce_op, init: identity }] } }
  Accumulate { dst: final_output, op: reduce_op, src: staging[r] }
```

The `Barrier` node at step 3 separates the parallel write phase from the serial
read phase. The LLIR representation guarantees this ordering structurally — it
is not a hint or annotation. Codegen emits `__syncthreads()` (CUDA),
`threadgroup_barrier(mem_flags::mem_threadgroup)` (Metal), or
`workgroupBarrier()` (WGSL) at exactly this position.

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
    /// Mark loop_var as Parallel. The backend decides the concurrency substrate.
    Parallelize { loop_var: String },
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
    /// Valid Opt instances for a kernel in a given context.
    /// KernelContext carries all information needed to constrain the search space:
    /// loop bounds, which axes are reduction axes, available shared memory,
    /// and backend class. Shape and dtype alone are insufficient.
    fn opt_candidates(&self, ctx: &KernelContext) -> Vec<Opt>;
}

/// Full context for Opt candidate generation.
pub struct KernelContext {
    /// Bound of each loop axis, in loop order.
    pub loop_bounds:   Vec<i64>,
    /// Which axes are reduction axes (map to LoopKind::Reduce).
    pub reduce_axes:   Vec<usize>,
    pub dtype:         DType,
    /// Shared memory budget in bytes (backend-provided).
    pub shared_budget: usize,
    pub backend:       BackendClass,
}

pub enum BackendClass { Cpu, Gpu, Wgsl }

pub struct CostEstimate {
    pub compute_cycles:       f64,
    pub memory_cycles:        f64,
    pub total_cycles:         f64,   // max(compute, memory) — roofline
    pub working_set_bytes:    usize,
    pub arithmetic_intensity: f64,
    pub bottleneck:           Bottleneck,
}
```

`opt_candidates` uses `KernelContext` rather than shape and dtype alone because
valid opts depend on axis role (`Vectorize` is illegal on a reduce axis),
available shared memory (`GroupReduce` requires a staging buffer that fits the
budget), and backend class (e.g. `GroupReduce` is meaningless on CPU without a
GPU thread model). The searcher never proposes opts the hardware cannot realise.

---

## 11. Appendix: Lowering Examples

### 11.1 Matmul

**HLIR:** `Reduce(Sum, Mul(Expand(A,[M,1,K]), Expand(B,[1,N,K])), axis=2)`

Starting loop nest after domain construction: `[M(0), N(1), K(2)]`

**ScheduleDecision opts** (axis values reflect the nest after prior opts):
```
Tile      { axis: 0, amt: 64 }   // tile M       → [m_outer(0), m_inner(1), N(2), K(3)]
Tile      { axis: 2, amt: 64 }   // tile N        → [m_outer(0), m_inner(1), n_outer(2), n_inner(3), K(4)]
Tile      { axis: 4, amt: 8  }   // tile K        → [m_outer(0), m_inner(1), n_outer(2), n_inner(3), k_outer(4), k_inner(5)]
Parallelize { axis: 0, amt: 64 } // m_outer → Parallel+split → [m_par_outer(0), m_par_inner(1), m_inner(2), n_outer(3), n_inner(4), k_outer(5), k_inner(6)]
Vectorize { axis: 4, amt: 64 }   // n_inner (axis 4) → Vectorized
```

**LLIR** (lowerer detects Mul+Reduce(Sum) under Vectorized loop and emits Fma):
```
for m_outer in [0, M/64)    [Parallel]
  for n_outer in [0, N/64)  [Sequential]
    for k_outer in [0, K/8) [Sequential]
      for m_inner in [0, 64) [Sequential]
        for n_inner in [0, 64) [Vectorized(64)]
          for k_inner in [0, 8) [Reduce{acc: Sum, init: 0}]
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
