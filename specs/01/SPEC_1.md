Perfect. Now let me create a **minimal, Bitter Lesson-compliant spec**:

---

# Minimal Hybrid IR Specification
## Simple, General, Auto-Optimizing

**Core Philosophy:**
- **Small RISC op set** (~15 primitives)
- **No domain-specific ops** (no Attention, no LayerNorm nodes)
- **Clean HLIR/LLIR boundary**
- **Let the auto-scheduler discover patterns**

---

## Part I: High-Level IR (HLIR)

### Core Types

```rust
// Symbolic dimension
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Dim {
    Const(i64),
    Sym(String),       // "N", "seq_len"
    Add(Box<Dim>, Box<Dim>),
    Mul(Box<Dim>, i64),
    Div(Box<Dim>, i64),
}

// Tensor type
#[derive(Debug, Clone)]
pub struct TensorType {
    pub shape: Vec<Dim>,
    pub dtype: DType,
    pub strides: Vec<Dim>,  // For view tracking
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32, F16, BF16,
    I32, I64,
    Bool,
}
```

### Operations (RISC-style)

```rust
#[derive(Debug, Clone)]
pub enum Op {
    // ===== Memory =====
    Const { value: Scalar, dtype: DType },
    Load { buffer: BufferId },
    Store { buffer: BufferId, value: NodeId },
    
    // ===== Unary =====
    Neg(NodeId),
    Recip(NodeId),     // 1/x (for division: a/b = a * recip(b))
    Exp(NodeId),
    Log(NodeId),
    Sqrt(NodeId),
    Sin(NodeId),       // For positional encodings
    
    // ===== Binary =====
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Max(NodeId, NodeId),
    
    // ===== Ternary =====
    Where { cond: NodeId, then_val: NodeId, else_val: NodeId },
    
    // ===== Reduction =====
    Reduce {
        input: NodeId,
        axes: Vec<usize>,
        op: ReduceOp,    // Sum, Max
    },
    
    // ===== Views (layout changes) =====
    Reshape { input: NodeId, new_shape: Vec<Dim> },
    Permute { input: NodeId, axes: Vec<usize> },
    Slice { input: NodeId, ranges: Vec<(Dim, Dim)> },
    Expand { input: NodeId, new_shape: Vec<Dim> },  // Broadcast
}

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp { Sum, Max }

// The graph
pub struct Graph {
    nodes: Vec<Node>,
    buffers: Vec<BufferDecl>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub ty: TensorType,
}
```

**That's it.** No Matmul node, no LayerNorm node, no Softmax node. They decompose:

```rust
// Matmul is just: Reduce(Sum, Mul(Expand(A), Expand(B)))
// Softmax is: x_exp = Exp(x - Max(x)); x_exp / Sum(x_exp)
// LayerNorm is: (x - mean) / sqrt(var + eps)
//   where mean = Reduce(Sum, x) / N
//         var = Reduce(Sum, (x - mean)^2) / N
```

### HLIR Passes (Minimal)

```rust
pub trait Pass {
    fn run(&self, graph: Graph) -> Result<Graph>;
}

// 1. Constant folding: Mul(Const(2), Const(3)) -> Const(6)
pub struct ConstantFold;

// 2. Algebraic: x + 0 -> x, x * 1 -> x
pub struct AlgebraicSimplify;

// 3. CSE: Deduplicate identical subgraphs
pub struct CommonSubexpression;

// 4. DCE: Remove unused nodes
pub struct DeadCodeElim;

// 5. View fusion: Permute(Permute(x, p1), p2) -> Permute(x, compose(p1, p2))
pub struct ViewFusion;
```

---

## Part II: Low-Level IR (LLIR)

### Polyhedral Primitives

```rust
// Affine expression: c0 + c1*v1 + c2*v2 + ...
#[derive(Debug, Clone)]
pub struct Aff {
    pub constant: i64,
    pub terms: Vec<(i64, Var)>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Var {
    Iter(String),  // i, j, k (loop indices)
    Param(String), // N, M (symbolic params)
}

// Constraint: aff >= 0 or aff = 0
#[derive(Debug, Clone)]
pub enum Constraint {
    Eq(Aff),
    Ineq(Aff),
}

// Integer set: { [i, j] : 0 <= i < N, 0 <= j < M }
#[derive(Debug, Clone)]
pub struct Domain {
    pub iters: Vec<String>,
    pub params: Vec<String>,
    pub constraints: Vec<Constraint>,
}

// Access map: { [i, j] -> [addr] : addr = stride_i*i + stride_j*j }
#[derive(Debug, Clone)]
pub struct AccessMap {
    pub domain_iters: Vec<String>,
    pub range_dims: Vec<String>,
    pub mapping: Vec<Aff>,
}
```

### Kernel Representation

```rust
pub struct Kernel {
    pub name: String,
    pub domain: Domain,
    pub body: Vec<Stmt>,
    pub reads: Vec<Access>,
    pub writes: Vec<Access>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    // Scalar ops (inside loops)
    ScalarBinary { op: BinOp, lhs: Expr, rhs: Expr },
    ScalarUnary { op: UnOp, arg: Expr },
    Assign { lhs: Expr, rhs: Expr },
}

#[derive(Debug, Clone)]
pub struct Access {
    pub buffer: BufferId,
    pub map: AccessMap,
}

pub struct LLIRGraph {
    pub kernels: Vec<Kernel>,
    pub dependencies: Vec<Dependence>,
}

pub struct Dependence {
    pub from: KernelId,
    pub to: KernelId,
    pub kind: DepKind,
    pub map: AccessMap,  // Which iterations depend on which
}

pub enum DepKind { RAW, WAR, WAW }
```

---

## Part III: The Lowering Boundary

### Lowering Algorithm

```rust
impl Graph {
    pub fn lower(&self) -> Result<LLIRGraph> {
        let mut kernels = Vec::new();
        
        for node in &self.nodes {
            kernels.extend(lower_node(node)?);
        }
        
        let deps = analyze_dependencies(&kernels)?;
        
        Ok(LLIRGraph { kernels, dependencies: deps })
    }
}

fn lower_node(node: &Node) -> Result<Vec<Kernel>> {
    match &node.op {
        // Elementwise: single kernel
        Op::Add(a, b) | Op::Mul(a, b) => {
            let domain = shape_to_domain(&node.ty.shape)?;
            let output_map = strides_to_access(&node.ty.strides, &domain)?;
            let input_a_map = strides_to_access(&a.ty.strides, &domain)?;
            let input_b_map = strides_to_access(&b.ty.strides, &domain)?;
            
            Ok(vec![Kernel {
                domain,
                body: vec![Stmt::Assign {
                    lhs: Expr::Access(output_map),
                    rhs: Expr::Binary {
                        op: match node.op { Op::Add(..) => BinOp::Add, _ => BinOp::Mul },
                        lhs: Box::new(Expr::Access(input_a_map)),
                        rhs: Box::new(Expr::Access(input_b_map)),
                    }
                }],
                reads: vec![Access { buffer: a.buffer, map: input_a_map },
                           Access { buffer: b.buffer, map: input_b_map }],
                writes: vec![Access { buffer: node.buffer, map: output_map }],
            }])
        },
        
        // Reduce: two kernels (init + update)
        Op::Reduce { input, axes, op } => {
            let full_domain = shape_to_domain(&input.ty.shape)?;
            let reduce_iters: Vec<_> = axes.iter().map(|&i| full_domain.iters[i].clone()).collect();
            let outer_iters: Vec<_> = full_domain.iters.iter()
                .filter(|it| !reduce_iters.contains(it))
                .cloned()
                .collect();
            
            // Kernel 1: init to 0 (or -inf for Max)
            let init_kernel = Kernel {
                domain: Domain { iters: outer_iters.clone(), .. },
                body: vec![Stmt::Assign { 
                    lhs: Expr::Access(..),
                    rhs: Expr::Const(init_value(*op))
                }],
                ..
            };
            
            // Kernel 2: reduction loop
            let reduce_kernel = Kernel {
                domain: full_domain,
                body: vec![Stmt::Assign {
                    lhs: Expr::Access(..),  // output[outer_iters]
                    rhs: Expr::Binary {
                        op: match op { ReduceOp::Sum => BinOp::Add, ReduceOp::Max => BinOp::Max },
                        lhs: Box::new(Expr::Access(..)),  // current accumulator
                        rhs: Box::new(Expr::Access(..)),  // input[all_iters]
                    }
                }],
                ..
            };
            
            Ok(vec![init_kernel, reduce_kernel])
        },
        
        // Views: sometimes no kernel (just metadata), sometimes copy
        Op::Permute { input, axes } => {
            // Check if we can propagate layout metadata
            if can_fuse_view(input)? {
                Ok(vec![])  // Zero-copy, just update strides
            } else {
                // Generate copy kernel with permuted access
                Ok(vec![make_copy_kernel(input, node)?])
            }
        },
        
        _ => todo!(),
    }
}
```

### Key Helpers

```rust
// Convert shape [N, M] to domain { [i, j] : 0 <= i < N, 0 <= j < M }
fn shape_to_domain(shape: &[Dim]) -> Result<Domain> {
    let iters: Vec<String> = (0..shape.len()).map(|i| format!("i{}", i)).collect();
    let mut params = Vec::new();
    let mut constraints = Vec::new();
    
    for (idx, dim) in shape.iter().enumerate() {
        let iter_var = Var::Iter(iters[idx].clone());
        
        // 0 <= i
        constraints.push(Constraint::Ineq(Aff {
            constant: 0,
            terms: vec![(1, iter_var.clone())],
        }));
        
        // i < dim  =>  dim - 1 - i >= 0
        match dim {
            Dim::Const(c) => {
                constraints.push(Constraint::Ineq(Aff {
                    constant: c - 1,
                    terms: vec![(-1, iter_var)],
                }));
            },
            Dim::Sym(s) => {
                params.push(s.clone());
                constraints.push(Constraint::Ineq(Aff {
                    constant: -1,
                    terms: vec![(1, Var::Param(s.clone())), (-1, iter_var)],
                }));
            },
            _ => todo!("Complex dimension expressions"),
        }
    }
    
    Ok(Domain { iters, params, constraints })
}

// Convert strides to access map
fn strides_to_access(strides: &[Dim], domain: &Domain) -> Result<AccessMap> {
    // For strides [s0, s1], iters [i, j], map is:
    // { [i, j] -> [addr] : addr = s0*i + s1*j }
    
    let mut terms = Vec::new();
    for (idx, stride) in strides.iter().enumerate() {
        match stride {
            Dim::Const(c) => {
                terms.push((*c, Var::Iter(domain.iters[idx].clone())));
            },
            _ => todo!("Symbolic strides"),
        }
    }
    
    Ok(AccessMap {
        domain_iters: domain.iters.clone(),
        range_dims: vec!["addr".into()],
        mapping: vec![Aff { constant: 0, terms }],
    })
}
```

---

## Part IV: Auto-Scheduler (Minimal)

### Fusion Strategy

```rust
pub truct AutoScheduler {
    cost_model: SimpleCostModel,
}

impl AutoScheduler {
    pub fn schedule(&self, llir: LLIRGraph) -> Result<Vec<Kernel>> {
        // Phase 1: Greedy fusion
        let mut fused = self.fuse_kernels(llir.kernels, &llir.dependencies)?;
        
        // Phase 2: Per-kernel tiling (beam search)
        for kernel in &mut fused {
            self.optimize_kernel(kernel)?;
        }
        
        Ok(fused)
    }
    
    fn fuse_kernels(&self, kernels: Vec<Kernel>, deps: &[Dependence]) -> Result<Vec<Kernel>> {
        let mut result = kernels;
        
        // Greedy: try to fuse each RAW dependency
        for dep in deps {
            if dep.kind != DepKind::RAW { continue; }
            
            let producer = &result[dep.from];
            let consumer = &result[dep.to];
            
            // Analyze dependency pattern
            if is_elementwise(&dep.map) {
                // 1:1 mapping, full fusion
                result = apply_elementwise_fusion(result, dep.from, dep.to)?;
            } else if let Some(tile_info) = detect_tiling(&dep.map) {
                // Producer needs tiling (e.g., Conv->Pool)
                result = apply_compute_at_fusion(result, dep.from, dep.to, tile_info)?;
            }
            // Else: keep separate
        }
        
        Ok(result)
    }
    
    fn optimize_kernel(&self, kernel: &mut Kernel) -> Result<()> {
        // Beam search over tile sizes
        let mut beam = vec![(kernel.clone(), f64::MAX)];
        
        for _ in 0..10 {  // Search depth
            let mut candidates = Vec::new();
            
            for (k, _) in &beam {
                // Try different tile sizes
                for tile_size in [16, 32, 64, 128] {
                    if let Ok(tiled) = apply_tiling(k, tile_size) {
                        let cost = self.cost_model.estimate(&tiled);
                        candidates.push((tiled, cost));
                    }
                }
            }
            
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            beam = candidates.into_iter().take(5).collect();  // Keep top 5
        }
        
        *kernel = beam[0].0.clone();
        Ok(())
    }
}

// Simple analytical cost model
struct SimpleCostModel;

impl SimpleCostModel {
    fn estimate(&self, kernel: &Kernel) -> f64 {
        let flops = count_operations(&kernel.body);
        let memory_accesses = count_memory_ops(&kernel.reads, &kernel.writes);
        
        // Roofline-inspired: max(compute_time, memory_time)
        let compute_time = flops as f64 / 1e12;  // 1 TFLOP/s assumed
        let memory_time = memory_accesses as f64 / 1e12;  // 1 TB/s bandwidth
        
        compute_time.max(memory_time)
    }
}

// Pattern detection
fn is_elementwise(map: &AccessMap) -> bool {
    // Check if map is identity: [i, j] -> [i, j]
    map.mapping.iter().enumerate().all(|(idx, aff)| {
        aff.constant == 0 && 
        aff.terms.len() == 1 &&
        aff.terms[0].0 == 1 &&
        aff.terms[0].1 == Var::Iter(map.domain_iters[idx].clone())
    })
}

fn detect_tiling(map: &AccessMap) -> Option<TileInfo> {
    // Look for patterns like: j = 4*jp + rj
    // (coefficient analysis, similar to Caten's test_ir.py:47-136)
    todo!()
}
```

---

## Part V: ISL Integration

```rust
// Minimal ISL FFI (just what we need)
pub mod isl {
    use std::ffi::CString;
    
    #[link(name = "isl")]
    extern "C" {
        fn isl_ctx_alloc() -> *mut IslCtx;
        fn isl_union_map_apply_range(...) -> *mut IslUnionMap;
        fn isl_schedule_get_map(...) -> *mut IslUnionMap;
        fn isl_ast_build_node_from_schedule(...) -> *mut IslAstNode;
        // ... only what auto-scheduler needs
    }
    
    // Safe wrappers
    pub struct UnionMap(*mut IslUnionMap);
    
    impl UnionMap {
        pub fn compose(&self, other: &UnionMap) -> UnionMap {
            unsafe { 
                UnionMap(isl_union_map_apply_range(self.0, other.0))
            }
        }
    }
}

// Use ISL only for:
// 1. Dependency composition (R ∘ W^-1)
// 2. Legality checking
// 3. AST generation (final codegen)
```

---

## Part VI: Codegen (Minimal)

```rust
pub fn codegen_cuda(kernel: &Kernel) -> Result<String> {
    // Use ISL to generate AST
    let ast = isl::ast_from_kernel(kernel)?;
    
    let mut code = String::new();
    code.push_str("__global__ void kernel(\n");
    
    // Add buffer params
    for buf in &kernel.reads {
        code.push_str(&format!("  float* {},\n", buf.buffer.name));
    }
    for buf in &kernel.writes {
        code.push_str(&format!("  float* {},\n", buf.buffer.name));
    }
    
    code.push_str(") {\n");
    code.push_str(&emit_ast(ast)?);  // ISL generates loops
    code.push_str("}\n");
    
    Ok(code)
}
```

---

## Part VII: Usage Example

```rust
// User builds computation graph
let mut g = Graph::new();

// Inputs
let x = g.parameter([Dim::Sym("B"), Dim::Sym("N"), Dim::Const(768)], DType::F32);

// Simple MLP: x -> ReLU -> Linear -> output
let neg_x = g.unary(Op::Neg(x));
let zeros = g.const_like(x, 0.0);
let relu = g.where_(g.less_than(x, zeros), zeros, x);  // max(0, x)

let w = g.parameter([Dim::Const(768), Dim::Const(3072)], DType::F32);

// Matmul via reduce:
// y[i,j] = sum_k(relu[i,k] * w[k,j])
let relu_expanded = g.expand(relu, [Dim::Sym("B"), Dim::Sym("N"), Dim::Const(1), Dim::Const(768)]);
let w_expanded = g.expand(w, [Dim::Const(1), Dim::Const(1), Dim::Const(3072), Dim::Const(768)]);
let products = g.mul(relu_expanded, w_expanded);
let y = g.reduce(products, vec![3], ReduceOp::Sum);

g.set_outputs(vec![y]);

// Optimize HLIR
let optimized = g
    .apply_pass(ConstantFold)?
    .apply_pass(AlgebraicSimplify)?
    .apply_pass(CommonSubexpression)?
    .apply_pass(DeadCodeElim)?;

// Lower to LLIR
let llir = optimized.lower()?;

// Auto-schedule
let scheduler = AutoScheduler::new();
let scheduled = scheduler.schedule(llir)?;

// Codegen
for kernel in scheduled {
    println!("{}", codegen_cuda(&kernel)?);
}
```

---

## Summary: What Changed

### Removed:
- ❌ MultiHeadAttention op
- ❌ LayerNorm op
- ❌ Softmax op
- ❌ Matmul as primitive (it's `Reduce(Sum, Mul(expand(...)))`)
- ❌ BatchedMatmul
- ❌ Complex auto-scheduler (no ML-based cost models)
- ❌ Symbolic algebra engine (SymPy/Z3)

### Kept (Minimal):
- ✅ 15 ops total (Unary, Binary, Reduce, Views, Memory)
- ✅ Symbolic shapes (just Const/Sym/Add/Mul/Div)
- ✅ Clean HLIR → LLIR boundary
- ✅ Greedy fusion + beam search over tiles
- ✅ Simple analytical cost model
- ✅ ISL for dependency analysis & codegen only

**Total LOC estimate: ~3000 lines of Rust** (vs 10,000+ in the complex version)

---

**Does this match the Bitter Lesson spirit?** 

The auto-scheduler will *discover* that Flash Attention is good (by fusing Q@K, Softmax, @V), not because we hand-coded it, but because the cost model prefers high arithmetic intensity.
