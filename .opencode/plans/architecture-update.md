# Architecture Update: Shared Backend Design

This document contains updates to add to `ARCHITECTURE.md` based on research into Tinygrad and Luminal's optimization approaches.

---

## Section to Add After "Overview": Reference Frameworks

```markdown
## Reference Frameworks

### Tinygrad (Liquid's Inspiration)

Tinygrad uses **BEAM search for per-kernel optimization**:
- Generates multiple kernel variants using `OptOps` (UPCAST, UNROLL, LOCAL, etc.)
- Actually compiles and times each variant on hardware
- Caches the fastest one
- Optimization scope: **local, per-kernel**

From tinygrad docs:
> "Kernel Speed (codegen) - This is what BEAM changes, it searches over a set of equivalent kernels which all perform the same operation and finds the one which performs the fastest."

### Luminal (Solid's Inspiration)

Luminal uses **egglog equality saturation for global optimization**:
- Operates on the entire computation graph at once
- Uses pattern matching to discover complex rewrites (e.g., FlashAttention)
- Compile-time optimization (AOT)
- Optimization scope: **global, whole-program**

From Luminal's Cargo.toml, it depends on `egglog` for equality saturation.
From the README:
> "The best heuristic is no heuristic. We try to search every possible decision... This allows us to automatically derive Flash Attention and other similarly complex rewrites."

### Key Architectural Difference

| Aspect | Tinygrad (Liquid) | Luminal (Solid) |
|--------|-------------------|-----------------|
| Optimization Scope | Per-kernel local | Whole-program global |
| Search Method | BEAM search (compile+time variants) | Egglog equality saturation |
| When | JIT (at runtime) | AOT (at compile time) |
| What's Optimized | Loop schedules, memory access | Graph structure, kernel fusion |
```

---

## New Section to Add: Shared Backend Architecture

Add this section after the "Key Abstractions" section:

```markdown
## Shared Backend Architecture

Liquid and Solid share the core code generation infrastructure through a trait hierarchy.

### Design Rationale

Both modes need to:
1. Convert `FusedKernel` → Cranelift IR
2. Emit loop structures for elementwise/reduce operations
3. Handle ShapeTracker indexing
4. Call math intrinsics (exp, log, sin, etc.)

The difference is:
- **Liquid**: Compiles and finalizes one kernel at a time (JIT)
- **Solid**: Compiles multiple kernels, then links into a single program (AOT)

### Trait Hierarchy

```
                    ┌───────────────────────────────────┐
                    │       CodeGenerator (Shared)      │
                    │  • Cranelift IR generation        │
                    │  • Expression building            │
                    │  • Loop emission                  │
                    │  • Math intrinsics                │
                    └───────────────┬───────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     │                     ▼
    ┌─────────────────────┐        │         ┌─────────────────────┐
    │   LiquidBackend     │        │         │    SolidBackend     │
    │ • Per-kernel JIT    │        │         │ • Multi-kernel AOT  │
    │ • Immediate finalize│        │         │ • Deferred linking  │
    │ • ExecutableKernel  │        │         │ • CompiledProgram   │
    └─────────────────────┘        │         └─────────────────────┘
```

### Trait Definitions

```rust
// shared/codegen/mod.rs

/// Intermediate representation before finalization
pub struct GeneratedKernel {
    /// Cranelift function (not yet compiled to machine code)
    pub function: cranelift::Function,
    /// Number of input buffer pointers
    pub num_inputs: usize,
    /// Debug IR if requested
    pub debug_ir: Option<String>,
    /// Kernel metadata
    pub metadata: KernelMetadata,
}

/// Core code generation capability (shared by Liquid and Solid)
pub trait CodeGenerator: Send + Sync {
    /// Generate Cranelift IR for a single fused kernel
    fn generate_kernel(
        &self,
        graph: &Graph,
        kernel: &FusedKernel,
        capture_ir: bool,
    ) -> Result<GeneratedKernel>;
}

// liquid/backend.rs

/// Liquid-style backend: per-kernel JIT compilation
pub trait LiquidBackend: CodeGenerator {
    /// Compile a kernel to executable form (JIT)
    fn compile_kernel(
        &self,
        graph: &Graph,
        kernel: &FusedKernel,
        capture_ir: bool,
    ) -> Result<Arc<dyn ExecutableKernel>> {
        let generated = self.generate_kernel(graph, kernel, capture_ir)?;
        self.finalize_kernel(generated)
    }
    
    /// Finalize generated IR to executable kernel
    fn finalize_kernel(&self, kernel: GeneratedKernel) -> Result<Arc<dyn ExecutableKernel>>;
}

// solid/backend.rs

/// Solid-style backend: whole-program AOT compilation
pub trait SolidBackend: CodeGenerator {
    /// Compile multiple kernels into a single program
    fn compile_program(
        &self,
        graph: &Graph,
        kernels: &[FusedKernel],
        buffer_plan: &StaticBufferPlan,
    ) -> Result<CompiledProgram> {
        // Generate all kernels using shared CodeGenerator
        let generated: Vec<GeneratedKernel> = kernels.iter()
            .map(|k| self.generate_kernel(graph, k, false))
            .collect::<Result<_>>()?;
        
        // Link into single executable program
        self.link_program(generated, buffer_plan)
    }
    
    /// Link multiple generated kernels into a program
    fn link_program(
        &self,
        kernels: Vec<GeneratedKernel>,
        buffer_plan: &StaticBufferPlan,
    ) -> Result<CompiledProgram>;
}
```

### Code Reuse from Current Implementation

| Current File | Reusable Component | New Location |
|--------------|-------------------|--------------|
| `jit/compile.rs:62-72` | Cranelift ISA setup | `shared/codegen/cranelift_setup.rs` |
| `jit/compile.rs:167-274` | `emit_elementwise_kernel` | `shared/codegen/emit.rs` |
| `jit/compile.rs:286-438` | `emit_reduce_kernel` | `shared/codegen/emit.rs` |
| `jit/expr.rs` | Expression tree → IR | `shared/codegen/expr.rs` |
| `jit/math.rs` | Math intrinsic declarations | `shared/codegen/math.rs` |
| `jit/tracker.rs` | ShapeTracker indexing | `shared/codegen/tracker.rs` |
| `jit/compiled.rs` | `CompiledKernel` struct | `liquid/jit/compiled.rs` |

### CPU Backend Implementation

```rust
// shared/codegen/cpu.rs

/// CPU code generator using Cranelift
pub struct CpuCodeGenerator {
    isa: Arc<dyn TargetIsa>,
}

impl CodeGenerator for CpuCodeGenerator {
    fn generate_kernel(
        &self,
        graph: &Graph,
        kernel: &FusedKernel,
        capture_ir: bool,
    ) -> Result<GeneratedKernel> {
        // Shared implementation: current jit/compile.rs logic
        // Returns Function, not finalized machine code
    }
}

// liquid/backend.rs

pub struct CpuLiquidBackend {
    codegen: CpuCodeGenerator,
}

impl LiquidBackend for CpuLiquidBackend {
    fn finalize_kernel(&self, kernel: GeneratedKernel) -> Result<Arc<dyn ExecutableKernel>> {
        // JIT compile to machine code immediately
        // Return ExecutableKernel with function pointer
    }
}

// solid/backend.rs

pub struct CpuSolidBackend {
    codegen: CpuCodeGenerator,
}

impl SolidBackend for CpuSolidBackend {
    fn link_program(
        &self,
        kernels: Vec<GeneratedKernel>,
        buffer_plan: &StaticBufferPlan,
    ) -> Result<CompiledProgram> {
        // Compile all kernels into single JIT module
        // Set up static buffer bindings
        // Return CompiledProgram
    }
}
```
```

---

## Updated Module Structure

Update the module structure diagram to include `shared/codegen/`:

```markdown
src/core/
├── shared/
│   ├── mod.rs
│   ├── graph/                   # Graph, Node, Op (100% reuse)
│   ├── dtype.rs                 # DType, Scalar, Buffer
│   ├── shape_tracker.rs         # ShapeTracker
│   ├── schedule/                # Scheduling logic
│   │   ├── fused_kernel.rs
│   │   ├── topo.rs
│   │   └── fusion_policy.rs
│   ├── optimize/                # Egglog optimization
│   ├── exec/                    # Interpreter fallbacks
│   └── codegen/                 # NEW: Shared code generation
│       ├── mod.rs
│       ├── cranelift_setup.rs   # ISA, flags, module setup
│       ├── emit.rs              # Loop emission (elementwise, reduce)
│       ├── expr.rs              # Expression tree → Cranelift IR
│       ├── math.rs              # Math intrinsic declarations
│       ├── tracker.rs           # ShapeTracker → address computation
│       └── cpu.rs               # CpuCodeGenerator implementation
│
├── liquid/
│   ├── mod.rs
│   ├── context.rs               # LiquidContext
│   ├── tensor/
│   ├── plan/
│   ├── jit/
│   │   ├── mod.rs
│   │   └── compiled.rs          # CompiledKernel (finalized)
│   ├── backend.rs               # LiquidBackend trait + CpuLiquidBackend
│   └── ...
│
├── solid/
│   ├── mod.rs
│   ├── context.rs               # SolidContext
│   ├── tensor/
│   ├── program/
│   │   ├── program.rs           # CompiledProgram
│   │   └── buffer_plan.rs       # StaticBufferPlan
│   ├── pass/
│   ├── backend.rs               # SolidBackend trait + CpuSolidBackend
│   └── ...
```

---

## Implementation Phase Update

Add to Phase 2 (or create new Phase 2.5):

### Phase 2.5: Extract Shared Codegen

**Goal**: Factor out code generation into shared module.

**Steps**:
1. Create `shared/codegen/` directory
2. Extract Cranelift setup from `jit/compile.rs` → `shared/codegen/cranelift_setup.rs`
3. Extract loop emission → `shared/codegen/emit.rs`
4. Move `jit/expr.rs` → `shared/codegen/expr.rs`
5. Move `jit/math.rs` → `shared/codegen/math.rs`
6. Move `jit/tracker.rs` → `shared/codegen/tracker.rs`
7. Create `CodeGenerator` trait
8. Create `CpuCodeGenerator` implementation
9. Create `LiquidBackend` trait extending `CodeGenerator`
10. Adapt `CpuBackend` → `CpuLiquidBackend` implementing `LiquidBackend`
11. Run all tests to verify no regression

**Estimated Time**: 2-3 hours

---

## Summary

The key insight is that **code generation is the same for both modes** — the difference is:
- **When** finalization happens (immediate vs. deferred)
- **How many** kernels are processed (one vs. many)
- **What** the output is (`ExecutableKernel` vs. `CompiledProgram`)

By extracting `CodeGenerator` as a shared trait, we get:
1. ~800 lines of code reuse from `jit/compile.rs`
2. Single implementation for Cranelift IR generation
3. Easy addition of new backends (CUDA, Metal) that work with both modes
