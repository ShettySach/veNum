# Venum IR & Scheduling Specification v2
## Backend-Agnostic, Search-First Design

---

## 1. Philosophy

### 1.1 The Bitter Lesson Applied

The bitter lesson teaches that general methods leveraging computation scale better than domain-specific engineering. For a tensor compiler, this means:

1. **Search over hand-tuning** — The scheduler discovers good schedules via search, not hardcoded heuristics
2. **Primitives over special ops** — No `Attention`, `LayerNorm`, `Matmul` as IR nodes; they decompose to primitives
3. **Patterns are data, not code** — Fusion patterns are recognized via e-graph rules, not hardcoded `if` branches
4. **Backend-agnostic representation** — Scheduling decisions are expressed in a neutral IR; backends interpret them

### 1.2 Non-Goals

- Hand-tuned kernel libraries as the primary path
- ISL or any single polyhedral library as a hard dependency
- Domain-specific ops that encode implementation choices

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
│  - Fusion regions with semantic tags                        │
│  - Scheduling decisions (what to fuse, tile sizes, etc.)    │
│  - E-graph search over candidate plans                      │
│  - Materialization vs. symbolic view decisions              │
└────────────────────────┬────────────────────────────────────┘
                         │ Commit & lower
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LLIR (Low-Level IR)                                        │
│  - Explicit loop nests                                      │
│  - Concrete access patterns                                 │
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

## 3. HLIR Specification

### 3.1 Core Types

```rust
/// Symbolic dimension expression
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Dim {
    Const(i64),
    Sym(Symbol),                    // Named parameter: "N", "seq_len"
    Add(Box<Dim>, Box<Dim>),
    Mul(Box<Dim>, Box<Dim>),        // Sym × Sym allowed at HLIR
    Div(Box<Dim>, Box<Dim>),
    Mod(Box<Dim>, Box<Dim>),
}

pub type Symbol = InternedString;   // Interned for cheap equality

/// Tensor type with shape and layout
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

### 3.2 Primitive Operations (~15)

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
    Sum, Prod, Max, Min,    // Affine-compatible
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

### 3.3 HLIR Graph

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

### 3.4 E-Graph Integration at HLIR

HLIR is the natural level for algebraic rewrites and pattern recognition. The e-graph operates on the HLIR graph:

**Algebraic rewrites** (equivalence-preserving):
- `Add(x, Const(0))` ↔ `x`
- `Mul(x, Const(1))` ↔ `x`
- `Mul(x, Const(0))` ↔ `Const(0)`
- `Neg(Neg(x))` ↔ `x`
- `Exp(Log(x))` ↔ `x` (domain-restricted)

**Pattern recognition** (annotation, not rewrite):
- `Reduce(Sum, Mul(Expand(_), Expand(_)))` → tag as `Contraction`
- `Exp(Sub(x, Reduce(Max, x)))` followed by `Reduce(Sum, ...)` → tag as `Softmax`
- Mean + Variance over same input → tag as `Normalization`

**Region annotation** output:

```rust
pub struct SemanticRegion {
    pub id: RegionId,
    pub nodes: Vec<NodeId>,         // Nodes belonging to this region
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
    Attention { seq_axis: usize },
    Generic,                        // Unrecognized pattern
}

pub struct RegionMetadata {
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub estimated_flops: Dim,       // Symbolic
    pub estimated_memory: Dim,      // Symbolic
}
```

---

## 4. Plan IR Specification

Plan IR captures scheduling decisions *before* lowering to explicit loops. It is the output of HLIR-level search and the input to LLIR lowering.

### 4.1 Core Types

```rust
pub struct PlanGraph {
    pub regions: Vec<PlannedRegion>,
    pub execution_order: Vec<RegionId>,
    pub materializations: HashMap<NodeId, MaterializationChoice>,
    pub symbol_bindings: HashMap<Symbol, SymbolBinding>,
}

pub struct PlannedRegion {
    pub id: RegionId,
    pub semantic: SemanticTag,
    pub nodes: Vec<NodeId>,         // HLIR nodes in this region
    pub schedule: RegionSchedule,
    pub fused_with: Option<RegionId>,
}

#[derive(Debug, Clone)]
pub struct RegionSchedule {
    pub tiling: Option<TilingSpec>,
    pub parallelism: ParallelismSpec,
    pub memory_placement: MemoryPlacement,
    pub specialization: Specialization,
}

#[derive(Debug, Clone)]
pub struct TilingSpec {
    pub tile_sizes: Vec<TileSize>,
    pub tile_order: Vec<usize>,     // Which dims to tile, in order
}

#[derive(Debug, Clone)]
pub enum TileSize {
    Const(i64),
    SearchParam(String),            // Determined by search: "tile_m", "tile_n"
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
    SharedMemory { size: Dim },
    Registers,
    ExplicitCache { level: usize },
}

#[derive(Debug, Clone)]
pub enum Specialization {
    LoopNest,                       // Generate explicit loops
    LibraryCall { name: String },   // Dispatch to cuBLAS, Accelerate, etc.
    CustomKernel { template: String },
    Search,                         // Defer to LLIR-level search
}
```

### 4.2 Materialization Decisions

```rust
#[derive(Debug, Clone)]
pub enum MaterializationChoice {
    /// Compute inline, no intermediate buffer
    Fused,
    /// Allocate buffer, store result
    Materialized { buffer: BufferId },
    /// Reuse existing buffer (aliasing)
    Alias { base: BufferId, offset: Dim },
}
```

### 4.3 Symbol Normalization

Before lowering, non-affine symbolic expressions must be normalized:

```rust
pub struct SymbolBinding {
    pub symbol: Symbol,
    pub definition: SymbolDef,
}

pub enum SymbolDef {
    Parameter,                              // User-provided runtime value
    Derived { expr: Dim, from: Vec<Symbol> }, // N*M → P_NM
}
```

Example: `Reshape([N, M] → [N*M])` introduces `P_NM = N * M`. The loop bound `0 <= i < P_NM` is affine.

### 4.4 E-Graph Search at Plan IR

Plan IR is where fusion and scheduling decisions are explored:

**Search dimensions:**
- Fusion boundaries (which regions merge)
- Tiling factors (powers of 2, hardware-specific)
- Parallelization strategy (which loops parallelize)
- Materialization vs. fusion

**Cost function:** Estimated runtime from `HardwareModel` (see §7)

**E-graph rules at Plan IR level:**

```
# Fusion rules
fuse(region_a, region_b) ↔ region_ab
    if producer_consumer(region_a, region_b) ∧ fusable(region_a, region_b)

# Tiling rules  
tile(region, axis, size) ↔ tiled_region
    if legal_tile(region, axis)

# Interchange rules
interchange(region, axis_i, axis_j) ↔ reordered_region
    if legal_interchange(region, axis_i, axis_j)
```

---

## 5. LLIR Specification

LLIR is the committed, lowered representation. It expresses explicit loop nests and memory accesses without being tied to any particular backend (ISL, Halide, handwritten).

### 5.1 Core Types

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
}

pub struct KernelParam {
    pub name: String,
    pub kind: ParamKind,
}

pub enum ParamKind {
    Buffer { ty: TensorType },
    Scalar { ty: DType },
    Dim { symbol: Symbol },
}
```

### 5.2 Loop Representation (Backend-Agnostic)

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
    pub step: i64,
    pub kind: LoopKind,
    pub annotations: LoopAnnotations,
}

pub type LoopVar = String;

#[derive(Debug, Clone, Copy)]
pub enum LoopKind {
    Sequential,
    Parallel,
    Vectorized { width: usize },
    Unrolled { factor: usize },
    // GPU-specific
    GridDim { axis: usize },        // blockIdx.x/y/z
    BlockDim { axis: usize },       // threadIdx.x/y/z
    // Reduction
    Reduction { op: ReduceOp, init: Scalar },
}

#[derive(Debug, Clone, Default)]
pub struct LoopAnnotations {
    pub tile_origin: Option<LoopVar>,   // For tiled loops: which outer loop
    pub cache_at: Option<usize>,        // Suggested cache level
    pub unroll_hint: Option<usize>,
}
```

### 5.3 Affine Expressions

```rust
/// Affine expression: c0 + c1*v1 + c2*v2 + ...
/// Strictly linear in loop variables and parameters
#[derive(Debug, Clone)]
pub struct AffineExpr {
    pub constant: i64,
    pub terms: Vec<(i64, Var)>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Var {
    Loop(LoopVar),
    Param(Symbol),
}

impl AffineExpr {
    pub fn constant(c: i64) -> Self {
        Self { constant: c, terms: vec![] }
    }
    
    pub fn var(v: Var) -> Self {
        Self { constant: 0, terms: vec![(1, v)] }
    }
    
    pub fn add(&self, other: &Self) -> Self { /* ... */ }
    pub fn scale(&self, c: i64) -> Self { /* ... */ }
    
    /// Check if this expression is affine (always true by construction)
    pub fn is_affine(&self) -> bool { true }
}
```

### 5.4 Memory Access

```rust
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub buffer: BufferId,
    pub indices: Vec<AffineExpr>,   // One per buffer dimension
    pub access_kind: AccessKind,
}

#[derive(Debug, Clone, Copy)]
pub enum AccessKind {
    Read,
    Write,
    ReadWrite,                      // Reductions
}

#[derive(Debug, Clone)]
pub struct BufferAlloc {
    pub id: BufferId,
    pub shape: Vec<AffineExpr>,     // May be symbolic
    pub dtype: DType,
    pub memory_space: MemorySpace,
}

#[derive(Debug, Clone, Copy)]
pub enum MemorySpace {
    Global,
    Shared,                         // GPU shared memory
    Local,                          // GPU registers / CPU stack
    Constant,
}
```

### 5.5 Statements

```rust
#[derive(Debug, Clone)]
pub enum Stmt {
    /// dst[indices] = src
    Assign {
        dst: MemoryAccess,
        src: Expr,
    },
    /// dst[indices] op= src (for reductions)
    Accumulate {
        dst: MemoryAccess,
        op: ReduceOp,
        src: Expr,
    },
    /// Conditional
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    /// Nested loop
    Loop(Loop, Vec<Stmt>),
    /// Synchronization barrier (GPU)
    Barrier { scope: BarrierScope },
    /// No-op (for legal schedule padding)
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
    Intrinsic { name: String, args: Vec<Expr> },
}

#[derive(Debug, Clone, Copy)]
pub enum BarrierScope {
    Workgroup,                      // __syncthreads()
    Subgroup,                       // Warp-level
    Device,                         // Full device (rare)
}
```

### 5.6 Dependence Representation

```rust
pub struct Dependence {
    pub from: (KernelId, StmtId),
    pub to: (KernelId, StmtId),
    pub kind: DepKind,
    pub distance: Option<Vec<i64>>, // Distance vector if known
    pub relation: DependenceRelation,
}

#[derive(Debug, Clone, Copy)]
pub enum DepKind {
    RAW,    // Read after write (true dependence)
    WAR,    // Write after read (anti-dependence)
    WAW,    // Write after write (output dependence)
}

/// Backend-agnostic dependence relation
/// Represents: { [source_iters] -> [sink_iters] : constraints }
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

## 6. Backend Traits

The scheduling and codegen pipeline is defined via traits that backends implement.

### 6.1 Dependence Analysis

```rust
/// Backend-provided dependence analysis
pub trait DependenceAnalyzer {
    /// Compute dependences between statements in a kernel
    fn analyze_kernel(&self, kernel: &Kernel) -> Result<Vec<Dependence>>;
    
    /// Check if a transformation is legal given dependences
    fn check_legality(
        &self,
        deps: &[Dependence],
        transform: &ScheduleTransform,
    ) -> Result<bool>;
    
    /// Compute the dependence relation between two memory accesses
    fn access_dependence(
        &self,
        write: &MemoryAccess,
        read: &MemoryAccess,
        loops: &[Loop],
    ) -> Result<Option<DependenceRelation>>;
}
```

### 6.2 Schedule Transformations

```rust
/// A schedule transformation that can be applied to LLIR
#[derive(Debug, Clone)]
pub enum ScheduleTransform {
    /// Tile a loop with given factor
    Tile { loop_var: LoopVar, factor: i64 },
    
    /// Interchange two loops
    Interchange { outer: LoopVar, inner: LoopVar },
    
    /// Fuse two adjacent loops with same bounds
    Fuse { loop_a: LoopVar, loop_b: LoopVar },
    
    /// Parallelize a loop
    Parallelize { loop_var: LoopVar, kind: ParallelKind },
    
    /// Unroll a loop
    Unroll { loop_var: LoopVar, factor: usize },
    
    /// Vectorize innermost loop
    Vectorize { loop_var: LoopVar, width: usize },
    
    /// Compute producer at a point in consumer's loop nest
    ComputeAt { producer: KernelId, consumer: KernelId, loop_var: LoopVar },
    
    /// Stage a buffer in faster memory
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
    /// Apply a transformation to a kernel, returning the transformed kernel
    fn apply(
        &self,
        kernel: &Kernel,
        transform: &ScheduleTransform,
    ) -> Result<Kernel>;
    
    /// Compose multiple transformations
    fn apply_sequence(
        &self,
        kernel: &Kernel,
        transforms: &[ScheduleTransform],
    ) -> Result<Kernel> {
        let mut k = kernel.clone();
        for t in transforms {
            k = self.apply(&k, t)?;
        }
        Ok(k)
    }
}
```

### 6.3 Code Generation

```rust
pub trait CodeGenerator {
    type Output;
    
    /// Generate code for a complete LLIR program
    fn generate(&self, program: &LLIRProgram) -> Result<Self::Output>;
    
    /// Generate code for a single kernel
    fn generate_kernel(&self, kernel: &Kernel) -> Result<String>;
}

/// CPU code generator
pub trait CpuCodeGen: CodeGenerator<Output = CpuModule> {
    fn intrinsics(&self) -> &CpuIntrinsics;
}

/// GPU code generator (CUDA, Metal, WebGPU)
pub trait GpuCodeGen: CodeGenerator<Output = GpuModule> {
    fn max_threads_per_block(&self) -> usize;
    fn max_shared_memory(&self) -> usize;
    fn warp_size(&self) -> usize;
}

pub struct CpuModule {
    pub source: String,             // C/Rust source
    pub symbols: Vec<String>,       // Exported kernel names
}

pub struct GpuModule {
    pub source: String,             // CUDA/Metal/WGSL source
    pub entry_points: Vec<GpuEntryPoint>,
}

pub struct GpuEntryPoint {
    pub name: String,
    pub grid_dims: usize,
    pub block_dims: usize,
    pub shared_memory: usize,
}
```

---

## 7. Cost Model & Hardware Abstraction

### 7.1 Hardware Model Trait

```rust
pub trait HardwareModel {
    /// Memory hierarchy (ordered from fastest to slowest)
    fn memory_levels(&self) -> &[MemoryLevel];
    
    /// Compute capabilities
    fn compute(&self) -> &ComputeCapabilities;
    
    /// Estimate cost of executing a kernel
    fn estimate_cost(&self, kernel: &Kernel) -> CostEstimate;
    
    /// Estimate cost of a memory access pattern
    fn memory_cost(&self, access: &MemoryAccess, loops: &[Loop]) -> f64;
}

#[derive(Debug, Clone)]
pub struct MemoryLevel {
    pub name: String,               // "L1", "L2", "Shared", "Global"
    pub size_bytes: usize,
    pub bandwidth_gbps: f64,
    pub latency_cycles: usize,
}

#[derive(Debug, Clone)]
pub struct ComputeCapabilities {
    pub vector_width: HashMap<DType, usize>,    // SIMD width per dtype
    pub peak_flops: HashMap<DType, f64>,        // Peak FLOP/s per dtype
    pub num_cores: usize,
    pub num_threads_per_core: usize,
    // GPU-specific
    pub num_sms: Option<usize>,
    pub warp_size: Option<usize>,
    pub tensor_cores: Option<TensorCoreSpec>,
}

#[derive(Debug, Clone)]
pub struct TensorCoreSpec {
    pub supported_shapes: Vec<(usize, usize, usize)>,   // (M, N, K) shapes
    pub supported_dtypes: Vec<(DType, DType)>,          // (input, output) pairs
    pub throughput: f64,
}
```

### 7.2 Cost Estimation

```rust
#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub compute_cycles: f64,
    pub memory_cycles: f64,
    pub total_cycles: f64,          // max(compute, memory) for roofline
    pub working_set_size: usize,
    pub arithmetic_intensity: f64,
    pub bottleneck: Bottleneck,
}

#[derive(Debug, Clone, Copy)]
pub enum Bottleneck {
    Compute,
    MemoryL1,
    MemoryL2,
    MemoryL3,
    MemoryDRAM,
    MemoryShared,
    MemoryGlobal,
    Latency,
}

impl HardwareModel for GenericCpuModel {
    fn estimate_cost(&self, kernel: &Kernel) -> CostEstimate {
        let flops = count_flops(&kernel.body);
        let (reads, writes) = count_memory_ops(&kernel.reads, &kernel.writes);
        let working_set = estimate_working_set(kernel);
        
        // Determine which cache level working set fits in
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

### 7.3 Predefined Hardware Models

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

## 8. Pipeline Integration

### 8.1 Full Compilation Pipeline

```rust
pub struct Compiler<H: HardwareModel, D: DependenceAnalyzer, C: CodeGenerator> {
    hardware: H,
    dep_analyzer: D,
    codegen: C,
    egraph_config: EGraphConfig,
}

impl<H, D, C> Compiler<H, D, C>
where
    H: HardwareModel,
    D: DependenceAnalyzer,
    C: CodeGenerator,
{
    pub fn compile(&self, hlir: HLIRGraph) -> Result<C::Output> {
        // Phase 1: HLIR optimization (e-graph)
        let optimized_hlir = self.optimize_hlir(hlir)?;
        
        // Phase 2: Region extraction & annotation
        let annotated = self.annotate_regions(optimized_hlir)?;
        
        // Phase 3: Plan IR search (e-graph over schedules)
        let plan = self.search_plans(annotated)?;
        
        // Phase 4: Lower to LLIR
        let llir = self.lower_to_llir(plan)?;
        
        // Phase 5: LLIR-level optimization
        let optimized_llir = self.optimize_llir(llir)?;
        
        // Phase 6: Codegen
        self.codegen.generate(&optimized_llir)
    }
    
    fn optimize_hlir(&self, hlir: HLIRGraph) -> Result<HLIRGraph> {
        // E-graph extraction with cost function based on operation count
        todo!()
    }
    
    fn annotate_regions(&self, hlir: HLIRGraph) -> Result<AnnotatedHLIR> {
        // Pattern recognition via e-graph matching
        // Output: HLIR + SemanticRegion annotations
        todo!()
    }
    
    fn search_plans(&self, annotated: AnnotatedHLIR) -> Result<PlanGraph> {
        // E-graph search over fusion/tiling/parallelization choices
        // Cost function: self.hardware.estimate_cost()
        todo!()
    }
    
    fn lower_to_llir(&self, plan: PlanGraph) -> Result<LLIRProgram> {
        // Region-based lowering: each PlannedRegion → Kernel(s)
        // Reductions stay unsplit until after fusion decisions
        todo!()
    }
    
    fn optimize_llir(&self, llir: LLIRProgram) -> Result<LLIRProgram> {
        // Apply ScheduleTransforms, check legality via dep_analyzer
        todo!()
    }
}
```

### 8.2 E-Graph Configuration

```rust
pub struct EGraphConfig {
    /// Max e-graph size before extraction
    pub max_nodes: usize,
    /// Max iterations of rewrite rules
    pub max_iterations: usize,
    /// Time budget for search (per phase)
    pub timeout: Duration,
    /// Extraction strategy
    pub extraction: ExtractionStrategy,
}

#[derive(Debug, Clone)]
pub enum ExtractionStrategy {
    /// Minimize estimated cost
    MinCost,
    /// Minimize AST size (for debugging)
    MinSize,
    /// Beam search with given width
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

---

## 9. Appendix: Lowering Examples

### 9.1 Matmul Lowering

HLIR (after decomposition):
```
Reduce(Sum, Mul(Expand(A, [M, 1, K]), Expand(B, [1, N, K])), axis=2)
```

Annotated:
```rust
SemanticRegion {
    tag: Contraction { m: M, n: N, k: K },
    nodes: [expand_a, expand_b, mul, reduce],
}
```

Plan IR:
```rust
PlannedRegion {
    schedule: RegionSchedule {
        tiling: Some(TilingSpec {
            tile_sizes: [TileSize::Const(64), TileSize::Const(64), TileSize::Const(8)],
            tile_order: [0, 1, 2],  // M, N, K
        }),
        parallelism: ParallelismSpec::Parallel { axis: 0, num_threads: 8 },
        specialization: Specialization::LoopNest,
    },
}
```

LLIR:
```rust
Kernel {
    body: LoopNest {
        loops: [
            Loop { var: "m_outer", lower: 0, upper: M/64, kind: Parallel },
            Loop { var: "n_outer", lower: 0, upper: N/64, kind: Sequential },
            Loop { var: "k_outer", lower: 0, upper: K/8, kind: Sequential },
            Loop { var: "m_inner", lower: 0, upper: 64, kind: Sequential },
            Loop { var: "n_inner", lower: 0, upper: 64, kind: Vectorized { width: 8 } },
            Loop { var: "k_inner", lower: 0, upper: 8, kind: Reduction { op: Sum } },
        ],
        body: [
            Accumulate {
                dst: C[m_outer*64 + m_inner, n_outer*64 + n_inner],
                op: Sum,
                src: A[m_outer*64 + m_inner, k_outer*8 + k_inner] 
                   * B[k_outer*8 + k_inner, n_outer*64 + n_inner],
            }
        ],
    },
    provenance: Contraction { m: M, n: N, k: K },
}
```

### 9.2 Softmax Lowering (Fused)

HLIR:
```
x_max = Reduce(Max, x, axis=-1)
x_shifted = Sub(x, Expand(x_max))
x_exp = Exp(x_shifted)
x_sum = Reduce(Sum, x_exp, axis=-1)
y = Div(x_exp, Expand(x_sum))
```

Annotated (single region):
```rust
SemanticRegion {
    tag: Softmax { axis: -1 },
    nodes: [reduce_max, sub, exp, reduce_sum, div],
}
```

Plan IR (fused, single-pass online softmax):
```rust
PlannedRegion {
    schedule: RegionSchedule {
        tiling: Some(TilingSpec { tile_sizes: [128], tile_order: [0] }),
        specialization: Specialization::Search,  // Let LLIR search find online algorithm
    },
}
```

---

## 10. Summary

| Layer | Purpose | Search | Representation |
|-------|---------|--------|----------------|
| **HLIR** | Express computation | E-graph: algebraic, pattern recognition | Tensor ops, symbolic shapes |
| **Plan IR** | Schedule decisions | E-graph: fusion, tiling, parallelism | Regions with schedules |
| **LLIR** | Executable form | Local transforms (legality-checked) | Explicit loops, accesses |

**Key principles:**
1. Semantic structure survives until Plan IR commits
2. Reductions unsplit until fusion boundaries are decided
3. E-graphs at both algebraic (HLIR) and scheduling (Plan IR) levels
4. Backend traits abstract over ISL/Halide/custom implementations
5. Hardware model drives cost-based extraction
