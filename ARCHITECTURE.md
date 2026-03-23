# Venum Architecture: Liquid & Solid Execution Modes

## Overview

Venum supports two execution modes with a shared foundation:

- **Liquid** (Tinygrad-style): Lazy per-tensor execution with JIT compilation
- **Solid** (Luminal-style): Ahead-of-time whole-program compilation

Both modes share core abstractions (Graph, Op, ShapeTracker) but differ in scheduling, optimization scope, and execution model.

---

## Design Principles

1. **Maximum Code Reuse**: ~60% of code is shared between Liquid and Solid
2. **Explicit APIs**: `LiquidContext` and `SolidContext` are distinct types
3. **Consistent Tensor API**: Both use `Tensor::new(cx, ...)` pattern
4. **Polyhedral-Ready**: Architecture supports future polyhedral optimization layer

---

## Module Structure

```
src/core/
├── mod.rs
│
├── shared/                      # Foundation (reused by both)
│   ├── mod.rs
│   ├── graph/
│   │   ├── mod.rs
│   │   ├── graph_impl.rs        # Graph struct
│   │   ├── node.rs              # Node struct (buffer: Option<Buffer>)
│   │   ├── node_id.rs           # NodeId(usize)
│   │   ├── op.rs                # Op enum
│   │   └── signature.rs         # GraphSignature
│   ├── shape_tracker.rs
│   ├── dtype.rs                 # DType, Scalar, Buffer
│   ├── schedule/
│   │   ├── mod.rs
│   │   ├── fused_kernel.rs      # FusedKernel, ReduceSpec
│   │   ├── schedule_item.rs     # ScheduleItem
│   │   ├── topo.rs              # topo_sort, consumer analysis
│   │   └── fusion_policy.rs     # FusionPolicy trait
│   ├── optimize/
│   │   ├── mod.rs
│   │   ├── egglog_program.rs
│   │   └── parse.rs
│   └── exec/                    # Interpreter fallbacks
│       ├── mod.rs
│       ├── buffer.rs
│       ├── reduce.rs
│       └── shape.rs
│
├── liquid/                      # Tinygrad-style (renamed from lazy/)
│   ├── mod.rs
│   ├── context.rs               # LiquidContext
│   ├── tensor/
│   │   ├── mod.rs
│   │   ├── structure.rs         # Tensor struct
│   │   ├── constructors.rs      # Tensor::from_slice, etc.
│   │   ├── ops_*.rs
│   │   └── realize.rs           # Per-tensor realize()
│   ├── plan/
│   │   ├── mod.rs
│   │   ├── exec_plan.rs         # ExecutionPlan
│   │   ├── build.rs
│   │   └── buffer_pool.rs       # Dynamic allocation
│   ├── jit/                     # Cranelift per-kernel JIT
│   │   └── ...
│   ├── backend.rs               # Backend trait (per-kernel)
│   ├── kernel.rs                # ExecutableKernel trait
│   ├── fusion_policy.rs         # LiquidFusionPolicy
│   └── lru_cache.rs
│
├── solid/                       # Luminal-style AOT (NEW)
│   ├── mod.rs
│   ├── context.rs               # SolidContext
│   ├── tensor/
│   │   ├── mod.rs
│   │   ├── structure.rs         # Tensor struct (mirrors liquid)
│   │   ├── constructors.rs      # Tensor::placeholder, from_slice, etc.
│   │   └── ops_*.rs
│   ├── program/
│   │   ├── mod.rs
│   │   ├── program.rs           # CompiledProgram
│   │   ├── buffer_plan.rs       # StaticBufferPlan
│   │   └── spec.rs              # TensorSpec
│   ├── pass/
│   │   ├── mod.rs
│   │   ├── manager.rs           # PassManager
│   │   ├── optimize.rs          # OptimizationPass (wraps shared)
│   │   ├── fusion.rs            # GlobalFusionPass
│   │   ├── memory.rs            # MemoryPlanningPass
│   │   └── lowering.rs          # DeviceLoweringPass
│   ├── compile.rs               # compile() entry point
│   ├── runtime.rs               # execute() runtime
│   ├── fusion_policy.rs         # SolidFusionPolicy
│   └── backend.rs               # GlobalBackend trait
│
└── polyhedral/                  # Future: Shared optimization layer
    ├── mod.rs
    ├── domain.rs                # IterationDomain
    ├── access.rs                # AccessRelation (from ShapeTracker)
    ├── dependence.rs            # Dependence analysis
    ├── transform.rs             # Tiling, interchange, fusion
    └── schedule.rs              # PolyhedralSchedule
```

---

## API Design

### Liquid API (Tinygrad-style)

```rust
use venum::liquid::{LiquidContext, Tensor};

// Create context
let cx = LiquidContext::new();

// Create tensors - context is first argument
let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
let b = Tensor::from_slice(&cx, &[4.0, 5.0, 6.0], vec![3]);

// Build computation graph (lazy)
let c = a.add(&b)?;
let d = c.exp();
let e = d.sum(&[0], false)?;

// Execute (JIT compile + run)
let result = e.realize()?;
println!("{:?}", result.to_vec_f32());
```

### Solid API (Luminal-style)

```rust
use venum::solid::{SolidContext, Tensor, compile};

// Create context
let cx = SolidContext::new();

// Create symbolic inputs (placeholders)
let input = Tensor::placeholder(&cx, vec![batch, seq, hidden], DType::F32);
let weights = Tensor::placeholder(&cx, vec![hidden, hidden], DType::F32);

// Or load concrete data for weights
let weights = Tensor::from_slice(&cx, &weight_data, vec![hidden, hidden]);

// Build computation graph
let hidden = input.matmul(&weights)?;
let output = hidden.layer_norm()?;

// Compile entire graph (AOT)
let program = compile(
    &cx,
    &[input.id()],           // Symbolic inputs
    &[output.id()],          // Outputs to compute
)?;

// Execute compiled program (minimal runtime)
let results = program.execute(&[&input_buffer])?;

// Optionally serialize for deployment
program.save("model.vbin")?;
let program = CompiledProgram::load("model.vbin")?;
```

### Consistent Constructor Pattern

Both Liquid and Solid use the same constructor pattern:

```rust
// Pattern: Tensor::method(&context, ...)

// Liquid
let a = Tensor::from_slice(&liquid_cx, &data, shape);
let b = Tensor::constant(&liquid_cx, 1.0, shape);
let c = Tensor::zeros(&liquid_cx, shape, DType::F32);

// Solid
let a = Tensor::from_slice(&solid_cx, &data, shape);
let b = Tensor::constant(&solid_cx, 1.0, shape);
let c = Tensor::zeros(&solid_cx, shape, DType::F32);
let p = Tensor::placeholder(&solid_cx, shape, DType::F32);  // Solid-only
```

---

## Key Abstractions

### Shared Foundation

#### Graph & Node

```rust
// shared/graph/graph_impl.rs
pub struct Graph {
    pub nodes: Vec<Node>,
}

// shared/graph/node.rs
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub buffer: Option<Buffer>,  // None for symbolic inputs (Solid)
}
```

#### FusionPolicy Trait

```rust
// shared/schedule/fusion_policy.rs
pub trait FusionPolicy: Send + Sync {
    /// Can this node be inlined into its consumer's kernel?
    fn can_inline(
        &self,
        graph: &Graph,
        node_id: NodeId,
        consumer_id: NodeId,
        consumer_counts: &HashMap<NodeId, usize>,
    ) -> bool;
}
```

### Liquid-Specific

#### LiquidContext

```rust
// liquid/context.rs
pub struct LiquidContext {
    graph: Arc<Mutex<Graph>>,
    kernel_cache: KernelCache,
    plan_cache: PlanCache,
    buffer_pool: SharedBufferPool,
    backend: Arc<dyn Backend>,
}
```

#### LiquidFusionPolicy

```rust
// liquid/fusion_policy.rs
pub struct LiquidFusionPolicy;

impl FusionPolicy for LiquidFusionPolicy {
    fn can_inline(&self, graph: &Graph, node_id: NodeId, consumer_id: NodeId,
                  consumer_counts: &HashMap<NodeId, usize>) -> bool {
        let node = graph.node(node_id);
        // Conservative: only inline single-consumer nodes
        node.op.is_elementwise()
            && *consumer_counts.get(&node_id).unwrap_or(&0) == 1
            && node.numel() == graph.node(consumer_id).numel()
    }
}
```

### Solid-Specific

#### SolidContext

```rust
// solid/context.rs
pub struct SolidContext {
    graph: Arc<Mutex<Graph>>,
    inputs: Vec<NodeId>,    // Tracked symbolic inputs
    backend: Arc<dyn GlobalBackend>,
}

impl SolidContext {
    pub fn new() -> Self;
    pub fn with_backend(backend: Arc<dyn GlobalBackend>) -> Self;
    
    /// Register a node as a symbolic input
    pub(crate) fn register_input(&self, id: NodeId);
    
    /// Get all registered inputs
    pub fn inputs(&self) -> Vec<NodeId>;
}
```

#### SolidFusionPolicy

```rust
// solid/fusion_policy.rs
pub struct SolidFusionPolicy {
    /// Nodes at compilation boundary (must materialize)
    boundary: HashSet<NodeId>,
}

impl FusionPolicy for SolidFusionPolicy {
    fn can_inline(&self, graph: &Graph, node_id: NodeId, consumer_id: NodeId,
                  consumer_counts: &HashMap<NodeId, usize>) -> bool {
        let node = graph.node(node_id);
        if !node.op.is_elementwise() { return false; }
        if node.numel() != graph.node(consumer_id).numel() { return false; }
        
        // Aggressive: can inline multi-consumer if all in same compilation unit
        !self.boundary.contains(&node_id)
    }
}
```

#### CompiledProgram

```rust
// solid/program/program.rs
pub struct CompiledProgram {
    /// Input specifications
    pub inputs: Vec<TensorSpec>,
    
    /// Output specifications
    pub outputs: Vec<TensorSpec>,
    
    /// Static buffer allocation plan
    pub buffer_plan: StaticBufferPlan,
    
    /// Compiled kernels
    pub kernels: Vec<CompiledKernel>,
    
    /// Execution schedule
    pub schedule: Vec<KernelInvocation>,
}

impl CompiledProgram {
    /// Execute with runtime inputs
    pub fn execute(&self, inputs: &[&Buffer]) -> Result<Vec<Buffer>>;
    
    /// Serialize for caching/deployment
    pub fn save(&self, path: &str) -> Result<()>;
    pub fn load(path: &str) -> Result<Self>;
}
```

#### StaticBufferPlan

```rust
// solid/program/buffer_plan.rs
pub struct BufferSlot {
    pub size: usize,
    pub dtype: DType,
}

pub struct StaticBufferPlan {
    /// Buffer slots (reused across non-overlapping lifetimes)
    pub slots: Vec<BufferSlot>,
    
    /// Maps NodeId to slot index
    pub allocation: HashMap<NodeId, usize>,
    
    /// Liveness: (first_use_step, last_use_step) per node
    pub liveness: HashMap<NodeId, (usize, usize)>,
    
    /// Total memory required
    pub total_bytes: usize,
}
```

#### Pass Infrastructure

```rust
// solid/pass/manager.rs
pub trait GraphPass: Send + Sync {
    fn name(&self) -> &'static str;
    fn run(&self, graph: &mut Graph, roots: &[NodeId]) -> Result<()>;
}

pub struct PassManager {
    passes: Vec<Box<dyn GraphPass>>,
}

impl PassManager {
    pub fn new() -> Self;
    pub fn add<P: GraphPass + 'static>(&mut self, pass: P) -> &mut Self;
    pub fn run(&self, graph: &mut Graph, roots: &[NodeId]) -> Result<()>;
}
```

#### compile() Function

```rust
// solid/compile.rs
pub fn compile(
    cx: &SolidContext,
    inputs: &[NodeId],
    outputs: &[NodeId],
) -> Result<CompiledProgram> {
    // 1. Clone reachable subgraph
    let (mut graph, output_map) = clone_multi_root_subgraph(cx.graph(), outputs);
    
    // 2. Run optimization passes
    let mut pm = PassManager::new();
    pm.add(OptimizationPass::new());      // Egglog algebraic optimization
    pm.add(FusionPass::new());            // Global kernel fusion
    pm.add(MemoryPlanningPass::new());    // Static buffer allocation
    pm.add(LoweringPass::new());          // Device-specific lowering
    pm.run(&mut graph, &mapped_outputs)?;
    
    // 3. Generate code
    let backend = cx.backend();
    backend.compile_program(&graph, inputs, outputs)
}
```

---

## Symbolic Inputs (Solid)

For Solid, inputs are symbolic (shape known, data provided at execution time).

**Representation**: `Node` with `buffer: None`

```rust
// solid/tensor/constructors.rs
impl Tensor {
    /// Create a symbolic input placeholder (Solid-only)
    pub fn placeholder(cx: &SolidContext, shape: Vec<usize>, dtype: DType) -> Self {
        let graph = cx.graph();
        let id = graph.lock().unwrap().add_node(Node {
            op: Op::Load,
            inputs: vec![],
            shape: shape.clone(),
            dtype,
            buffer: None,  // No data - symbolic!
        });
        cx.register_input(id);
        Self { cx: cx.clone(), id, shape, dtype }
    }
}
```

**Alternative Considered**: `Op::Placeholder` variant
- Pro: More explicit in the Op enum
- Con: Adds complexity, Load with None buffer is sufficient

**Alternative Considered**: Separate `InputSpec` type outside Graph
- Pro: Cleaner separation
- Con: Requires more API changes, breaks graph completeness

---

## Multi-Root Handling

Neural networks often have multiple outputs (loss, predictions, attention weights).

**Approach**: `compile(cx, inputs, outputs: &[NodeId])` accepts multiple output roots.

```rust
// Example: Model with multiple outputs
let logits = model.forward(&input)?;
let loss = logits.cross_entropy(&labels)?;
let probs = logits.softmax(-1)?;

// Compile with multiple outputs
let program = compile(
    &cx,
    &[input.id(), labels.id()],
    &[loss.id(), probs.id()],  // Both outputs
)?;

// Execute returns all outputs
let [loss_buf, probs_buf] = program.execute(&[&input_data, &labels_data])?
    .try_into().unwrap();
```

---

## Code Reuse Summary

| Component | Lines | Reuse | Location |
|-----------|-------|-------|----------|
| Graph, Node, NodeId, Op | ~250 | 100% | shared/graph/ |
| ShapeTracker | ~200 | 100% | shared/shape_tracker.rs |
| DType, Scalar, Buffer | ~100 | 100% | shared/dtype.rs |
| Egglog optimization | ~570 | 100% | shared/optimize/ |
| Interpreter fallbacks | ~230 | 100% | shared/exec/ |
| FusedKernel, ScheduleItem | ~250 | 100% | shared/schedule/ |
| topo_sort, consumer analysis | ~100 | 100% | shared/schedule/topo.rs |
| FusionPolicy trait | ~50 | 100% | shared/schedule/fusion_policy.rs |
| **Total Shared** | **~1,750** | | |
| Liquid-specific | ~1,800 | - | liquid/ |
| **New for Solid** | **~1,200** | - | solid/ (estimated) |

---

## Implementation Phases

### Phase 1: Extract `shared/` Module

**Goal**: Factor out reusable components without breaking existing functionality.

**Steps**:
1. Create `shared/` directory structure
2. Move `graph/*` → `shared/graph/*`
3. Move `shape_tracker.rs` → `shared/`
4. Move `dtype.rs` → `shared/`
5. Move `optimize/*` → `shared/optimize/*`
6. Move `exec/*` → `shared/exec/*`
7. Move `GraphSignature` → `shared/graph/signature.rs`
8. Update all `lazy/` imports to use `shared::*`
9. Rename `lazy/` → `liquid/`
10. Update `lib.rs` exports

**Verification**: All existing tests pass, API unchanged.

**Estimated Time**: 2-3 hours

### Phase 2: Extract Scheduling to `shared/schedule/`

**Goal**: Make scheduling logic reusable by both Liquid and Solid.

**Steps**:
1. Create `shared/schedule/`
2. Move `FusedKernel`, `ScheduleItem` → `shared/schedule/`
3. Move `topo_sort`, consumer analysis → `shared/schedule/topo.rs`
4. Create `FusionPolicy` trait in `shared/schedule/fusion_policy.rs`
5. Create `LiquidFusionPolicy` in `liquid/fusion_policy.rs`
6. Parameterize `collect_kernel_inputs` to take `&dyn FusionPolicy`
7. Update `liquid/plan/build.rs` to use `LiquidFusionPolicy`

**Verification**: All existing tests pass, behavior unchanged.

**Estimated Time**: 1-2 hours

### Phase 3: Implement Solid Foundation

**Goal**: Create Solid module structure with basic types.

**Steps**:
1. Create `solid/` directory structure
2. Implement `TensorSpec` (shape, dtype, name)
3. Implement `StaticBufferPlan`
4. Implement `CompiledProgram` structure
5. Implement `SolidContext`
6. Implement Tensor API for Solid (placeholder, constructors, ops)
7. Implement `SolidFusionPolicy`

**Estimated Time**: 3-4 hours

### Phase 4: Implement Pass Infrastructure

**Goal**: Create the compiler passes for global optimization.

**Steps**:
1. Implement `GraphPass` trait
2. Implement `PassManager`
3. Implement `OptimizationPass` (wraps `shared/optimize`)
4. Implement `FusionPass` (uses `shared/schedule` with `SolidFusionPolicy`)
5. Implement `MemoryPlanningPass`
   - Liveness analysis
   - Slot assignment (buffer reuse)
   - Total memory calculation

**Estimated Time**: 2-3 hours

### Phase 5: Implement Codegen & Runtime

**Goal**: Complete the compilation and execution path.

**Steps**:
1. Create `GlobalBackend` trait in `solid/backend.rs`
2. Implement CPU backend (adapt `liquid/jit/` patterns)
3. Implement `compile()` entry point
4. Implement `CompiledProgram::execute()`
5. Add serialization (optional)
6. Add comprehensive tests

**Estimated Time**: 4-5 hours

---

## Testing Strategy

| Phase | Tests |
|-------|-------|
| Phase 1 | All existing tests must pass unchanged |
| Phase 2 | Existing + unit tests for FusionPolicy |
| Phase 3 | Unit tests for TensorSpec, StaticBufferPlan, SolidContext |
| Phase 4 | Unit tests for each pass, integration test for pipeline |
| Phase 5 | End-to-end: compile → execute, compare with Liquid results |

---

## Future: Polyhedral Optimization Layer

The architecture supports adding a shared polyhedral optimization layer:

```
                    ┌──────────────────────┐
                    │  Graph Optimization  │  (Egglog - shared)
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
      LIQUID PATH              │           SOLID PATH
            │                  │                  │
            ▼                  │                  ▼
   ┌─────────────────┐        │         ┌─────────────────┐
   │  Per-Kernel     │        │         │  Global Fusion  │
   │  Scheduling     │        │         │  Passes         │
   └────────┬────────┘        │         └────────┬────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  POLYHEDRAL LAYER    │  (shared, future)
                    │  - Tiling            │
                    │  - Loop interchange  │
                    │  - Vectorization     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Code Generation     │
                    └──────────────────────┘
```

**ShapeTracker → Polyhedral**: The existing `ShapeTracker` already represents affine access functions, making it ideal infrastructure for polyhedral analysis.

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking existing Liquid API | Phase 1 is pure refactor; run all tests |
| FusionPolicy bugs | Default to LiquidFusionPolicy, verify identical behavior |
| Solid codegen complexity | Start with single-kernel, then multi-kernel |
| Memory planning correctness | Validate against Liquid's dynamic allocation |

---

## Open Questions

1. **GPU Backends**: Should Solid initially focus on CPU-only, or target CUDA/Metal from the start?
2. **Serialization Format**: Custom binary format vs. existing standard (FlatBuffers, etc.)?
3. **Dynamic Shapes**: How to handle batch size flexibility in Solid? (compile-time specialization vs. runtime dispatch)
