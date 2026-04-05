# Plan: Dual-Backend Polyhedral API (Native + ISL)

## Overview

Design a **backend-agnostic polyhedral abstraction** that allows swapping between:
1. **Native implementation** (good_lp + Fourier-Motzkin + custom operations) - pure Rust
2. **ISL implementation** (FFI to C library) - battle-tested, feature-complete

The API will expose high-level polyhedral operations while hiding implementation details.

---

## Architecture

### Layer 1: Common Data Structures (Already Exists)
These types are **backend-agnostic** and shared by both implementations:
- `Aff` - affine functions
- `PolyVar` - iteration variables and parameters  
- `Constraint` - equality/inequality constraints
- `Domain` - iteration domains with constraints
- `AccessMap` - memory access functions
- `DependenceRelation` - dependence constraints (in `llir/dependence.rs`)

✅ **Status**: Already implemented in `src/core/poly/domain.rs` and `access_map.rs`

### Layer 2: Polyhedral Engine Trait (NEW)
Abstract interface for polyhedral operations:

```rust
/// Core polyhedral operations backend.
/// Both NativePolyhedralEngine and IslPolyhedralEngine implement this.
pub trait PolyhedralEngine {
    /// Check if a set of affine constraints is satisfiable (has integer solutions).
    /// Used to test if dependencies exist.
    fn is_feasible(&self, domain: &Domain, constraints: &[Constraint]) -> Result<bool>;
    
    /// Compute the inverse of an access map: W^{-1}
    /// { A[f(i)] -> [i] } from { [i] -> A[f(i)] }
    fn invert_map(&self, map: &AccessMap) -> Result<AccessMap>;
    
    /// Compose two access maps: R ∘ W^{-1}
    /// Produces dependence relation from write map W and read map R
    fn compose_maps(&self, read: &AccessMap, write_inv: &AccessMap) -> Result<DependenceRelation>;
    
    /// Project out (eliminate) specified variables from constraints.
    /// Used for computing distance vectors and simplifying dependence relations.
    fn project(&self, domain: &Domain, vars_to_eliminate: &[String]) -> Result<Domain>;
    
    /// Compute lexicographic distance vector for a dependence.
    /// Returns None if no constant distance, Some(vec) otherwise.
    fn compute_distance(&self, relation: &DependenceRelation) -> Result<Option<Vec<i64>>>;
    
    /// Check if a schedule transform preserves dependence legality.
    /// Returns true if transform is legal.
    fn check_transform_legality(
        &self,
        relation: &DependenceRelation,
        transform: &ScheduleTransform,
    ) -> Result<bool>;
}
```

### Layer 3: Backend Implementations (NEW + REFACTOR)

#### 3A. Native Backend (`src/core/poly/engine/native.rs`)
Pure Rust implementation using:
- **Affine arithmetic** (composition, inversion) - custom implementation
- **ILP feasibility** - `good_lp` with a pure-Rust solver backend
- **Fourier-Motzkin elimination** - custom implementation for projection
- **Distance computation** - lexicographic minimization via ILP

```rust
pub struct NativePolyhedralEngine {
    // Configuration for ILP solver
    ilp_config: IlpConfig,
}

impl PolyhedralEngine for NativePolyhedralEngine {
    // Implementation details below
}
```

#### 3B. ISL Backend (`src/core/poly/engine/isl.rs`) 
FFI to ISL C library:
- Serialize `Domain`/`AccessMap` to ISL strings
- Call ISL functions via FFI
- Parse results back to Rust types

```rust
pub struct IslPolyhedralEngine {
    ctx: *mut isl_sys::IslCtx,  // ISL context (not Send)
}

impl PolyhedralEngine for IslPolyhedralEngine {
    // FFI implementation
}
```

### Layer 4: DependenceAnalyzer Implementations (REFACTOR)

Refactor `NativeDependenceAnalyzer` to use `PolyhedralEngine`:

```rust
pub struct PolyhedralDependenceAnalyzer<E: PolyhedralEngine> {
    engine: E,
}

impl<E: PolyhedralEngine> DependenceAnalyzer for PolyhedralDependenceAnalyzer<E> {
    fn analyze_kernel(&self, kernel: &Kernel) -> Result<Vec<Dependence>> {
        // Collect accesses
        // For each write-read pair:
        //   1. invert_map(write) -> W^{-1}
        //   2. compose_maps(read, W^{-1}) -> relation
        //   3. is_feasible(relation) -> exists?
        //   4. compute_distance(relation) -> distance vector
        // Return dependencies
    }
    
    fn check_legality(&self, deps: &[Dependence], transform: &ScheduleTransform) -> Result<bool> {
        // Use engine.check_transform_legality for each dep
    }
    
    fn access_dependence(...) -> Result<Option<DependenceRelation>> {
        // Use engine operations
    }
}
```

**Type aliases for convenience:**
```rust
pub type NativeDependenceAnalyzer = PolyhedralDependenceAnalyzer<NativePolyhedralEngine>;
pub type IslDependenceAnalyzer = PolyhedralDependenceAnalyzer<IslPolyhedralEngine>;
```

---

## File Structure

```
src/core/poly/
├── mod.rs                      # Module exports
├── domain.rs                   # Aff, Constraint, Domain (existing)
├── access_map.rs               # AccessMap (existing)
├── tests.rs                    # Tests (existing)
├── engine/
│   ├── mod.rs                  # PolyhedralEngine trait + exports
│   ├── native/
│   │   ├── mod.rs              # NativePolyhedralEngine
│   │   ├── affine_ops.rs       # Composition, inversion
│   │   ├── ilp.rs              # ILP feasibility via good_lp
│   │   ├── fourier_motzkin.rs  # Projection/quantifier elimination
│   │   └── distance.rs         # Distance computation
│   └── isl/
│       ├── mod.rs              # IslPolyhedralEngine (feature-gated)
│       ├── ffi.rs              # ISL FFI bindings
│       └── serialize.rs        # Convert Rust types to ISL strings
└── analyzer.rs                 # PolyhedralDependenceAnalyzer<E>
```

---

## Implementation Phases

### Phase 1: Trait & Native Scaffolding
**Goal**: Define the trait and create a minimal native implementation

**Tasks**:
1. Create `src/core/poly/engine/mod.rs` with `PolyhedralEngine` trait
2. Create `NativePolyhedralEngine` stub that returns errors for all methods
3. Refactor existing `NativeDependenceAnalyzer` to use new structure:
   - Move to `src/core/poly/analyzer.rs` as `PolyhedralDependenceAnalyzer<E>`
   - Keep type alias `NativeDependenceAnalyzer` for backward compatibility
4. Update `src/core/poly/mod.rs` exports

**Verification**: Existing tests still pass with refactored structure

---

### Phase 2: Affine Operations (Native)
**Goal**: Implement affine map composition and inversion

**Tasks**:
1. **Create `affine_ops.rs`**:
   - `compose_affine(a1: &Aff, a2: &Aff)` - substitute a2 into a1
   - `invert_simple_map(map: &AccessMap)` - solve linear system for inverses
   - Handle 1D, 2D, 3D cases explicitly (common in tensor ops)

2. **Implement `PolyhedralEngine` methods**:
   - `invert_map()` - delegates to `invert_simple_map`
   - `compose_maps()` - use affine composition to create dependence relation

3. **Add `Aff` helper methods** in `domain.rs`:
   - `substitute(var: &PolyVar, expr: &Aff)` - replace variable with expression
   - `negate()` - negate all coefficients
   - `add(other: &Aff)` - add two affine expressions
   - `simplify()` - combine like terms (can use `egglog` here optionally)

**Example**:
```rust
// W: {[i,j] -> [128*i + j]}  (write map)
// W^{-1}: {[a] -> [i,j] : a = 128*i + j}  (invert)
// R: {[i',j'] -> [128*i' + j']}  (read map)
// R ∘ W^{-1}: {[i,j] -> [i',j'] : 128*i + j = 128*i' + j'}  (compose)
```

**Verification**: Unit tests for composition and inversion

---

### Phase 3: ILP Integration (Native)
**Goal**: Use `good_lp` to check constraint feasibility

**Tasks**:
1. **Add `good_lp` dependency** to `Cargo.toml`:
   ```toml
   good_lp = { version = "1.7", default-features = false, features = ["minilp"] }
   ```
   (Use `minilp` for pure Rust, or `highs` for better performance with C++ dep)

2. **Create `ilp.rs`**:
   - `is_integer_feasible(domain: &Domain, constraints: &[Constraint])` 
   - Convert `Aff`/`Constraint` to `good_lp` variables and constraints
   - Call solver with no objective (just feasibility check)
   - Return `Ok(true)` if feasible, `Ok(false)` if infeasible

3. **Implement `is_feasible()` in `NativePolyhedralEngine`**

**Example**:
```rust
// Check if { [i,j] : 128*i + j = 128*i' + j', 0 <= i < 16, 0 <= j < 64, ... } is satisfiable
let feasible = engine.is_feasible(&domain, &dep_constraints)?;
```

**Verification**: Test with known feasible/infeasible constraint systems

---

### Phase 4: Fourier-Motzkin Elimination (Native)
**Goal**: Implement variable projection for simplifying constraints

**Tasks**:
1. **Create `fourier_motzkin.rs`**:
   - `eliminate_variable(constraints: &[Constraint], var: &PolyVar)` 
   - For each pair of lower/upper bounds on `var`, create new constraint
   - Repeat for all variables to eliminate
   - Simplify resulting constraints

2. **Implement `project()` in `NativePolyhedralEngine`**

3. **Optimizations** (optional for Phase 4):
   - Dark shadow / real shadow optimizations
   - Constraint redundancy elimination
   - Early termination if infeasibility detected

**Example**:
```rust
// { [i,j,k] : i+j = k, 0 <= i <= 10, 0 <= j <= 10, 0 <= k <= 20 }
// Project out k:
// { [i,j] : 0 <= i <= 10, 0 <= j <= 10, 0 <= i+j <= 20 }
```

**Verification**: Test projection with known results

---

### Phase 5: Distance Computation (Native)
**Goal**: Compute lexicographic distance vectors for dependencies

**Tasks**:
1. **Create `distance.rs`**:
   - `lexicographic_minimize(relation: &DependenceRelation)` 
   - Set up ILP to minimize (sink[0] - source[0], sink[1] - source[1], ...)
   - Use `good_lp` optimization (not just feasibility)
   - Return constant distance if unique, None if variable

2. **Implement `compute_distance()` in `NativePolyhedralEngine`**

3. **Update `analyze_kernel()`** to use real distances instead of `vec![0; ...]`

**Example**:
```rust
// { [i,j] -> [i',j'] : i' = i+1, j' = j }  =>  distance = [1, 0]
// { [i,j] -> [i',j'] : i' = i, j' = j+k }  =>  distance = None (depends on k)
```

**Verification**: Test with simple loop nests (known distances)

---

### Phase 6: Legality Checking (Native)
**Goal**: Implement transform-specific legality checks using polyhedral operations

**Tasks**:
1. **Implement `check_transform_legality()`** in `NativePolyhedralEngine`:
   - For `Tile`: check if dependence distances are preserved
   - For `Interchange`: check if swapping loops violates positive distances  
   - For `Vectorize`: check no WAW deps with distance[axis] != 0
   - For `Parallelize`: check no positive distances

2. **Refactor `check_legality()` in analyzer** to delegate to engine

**Verification**: Test with transformations that should/shouldn't be legal

---

### Phase 7: LLIR Execution (Separate Track)
**Goal**: Make LLIR loop nests actually executable

This is **orthogonal to polyhedral work** per TODO.md item #1. Can be done in parallel.

**Options**:
- **Interpreter approach** (simpler, faster to implement):
  - Walk `LoopNest` structure, evaluate `Stmt`s directly
  - Handle loop kinds (Sequential, Parallel, Vectorized, etc.)
  
- **Cranelift codegen approach** (spec-compliant, better performance):
  - Generate CLIF IR from `LoopNest`
  - Compile to native code
  - Execute

**Recommendation**: Start with interpreter for validation, add Cranelift later.

---

### Phase 8: ISL Backend (Future)
**Goal**: Add ISL as alternative backend for comparison/validation

**Tasks**:
1. **Add ISL dependency** (feature-gated):
   ```toml
   [dependencies]
   isl-sys = { version = "...", optional = true }
   
   [features]
   isl = ["isl-sys"]
   ```

2. **Create FFI bindings** in `engine/isl/ffi.rs`:
   - Context management
   - Map operations (apply_range, reverse, is_empty)
   - Minimal API per SPEC.md §7.3

3. **Create `IslPolyhedralEngine`**:
   - Serialize `Aff`/`Domain`/`AccessMap` to ISL string format
   - Call ISL functions
   - Parse results back to Rust

4. **Testing**: Compare `NativePolyhedralEngine` vs `IslPolyhedralEngine` results

---

## Dependency Management

### `Cargo.toml` Updates

```toml
[dependencies]
# Existing dependencies...
good_lp = { version = "1.7", default-features = false, features = ["minilp"] }

# ISL support (optional, feature-gated)
isl-sys = { version = "0.x", optional = true }

[features]
default = ["native-poly"]
native-poly = []  # Pure Rust polyhedral (good_lp + Fourier-Motzkin)
isl = ["isl-sys"]  # ISL via FFI
```

### Feature Gates in Code

```rust
// src/core/poly/engine/mod.rs
pub mod native;

#[cfg(feature = "isl")]
pub mod isl;

pub trait PolyhedralEngine { ... }

// src/core/poly/analyzer.rs
pub type NativeDependenceAnalyzer = PolyhedralDependenceAnalyzer<native::NativePolyhedralEngine>;

#[cfg(feature = "isl")]
pub type IslDependenceAnalyzer = PolyhedralDependenceAnalyzer<isl::IslPolyhedralEngine>;
```

---

## Testing Strategy

### Unit Tests (Per Phase)
- **Affine ops**: Test composition, inversion with known results
- **ILP**: Test feasibility with simple/complex constraint systems
- **Fourier-Motzkin**: Test projection with known polytopes
- **Distance**: Test with loop nests from matmul, reductions, etc.

### Integration Tests
- **End-to-end lowering** with `NativeDependenceAnalyzer` enabled
- Compare `NoOpDependenceAnalyzer` vs `NativeDependenceAnalyzer` (should catch illegal transforms)
- **Cross-backend validation** (when ISL is ready): same input → same output

### Benchmark Tests (Optional)
- Time `NativePolyhedralEngine` vs `IslPolyhedralEngine` on real workloads
- Identify performance bottlenecks in native implementation

---

## Migration Path

### Immediate (Phase 1-3)
Current code using `NoOpDependenceAnalyzer` continues to work. Once native backend has basic feasibility checking:

```rust
// src/core/runner.rs - change from:
&NoOpDependenceAnalyzer

// to:
&NativeDependenceAnalyzer::default()
```

### Medium Term (Phase 4-6)
Full native polyhedral engine operational. Users can choose:
```rust
let analyzer = NativeDependenceAnalyzer::default();
// or
let analyzer = NativeDependenceAnalyzer::with_config(config);
```

### Long Term (Phase 8)
ISL available as alternative:
```rust
#[cfg(feature = "isl")]
let analyzer = IslDependenceAnalyzer::new();

#[cfg(not(feature = "isl"))]
let analyzer = NativeDependenceAnalyzer::default();
```

---

## Questions / Decisions Needed

1. **ILP Solver Choice**:
   - **`minilp`** (pure Rust, LP only, may be insufficient for integer constraints)
   - **`highs`** via `good_lp` (ILP support, C++ dependency but better than ISL)
   - **Custom branch-and-bound** on top of `minilp` (pure Rust but more work)
   
   **Recommendation**: Start with `highs` for correctness, consider pure Rust later if C++ dep is unacceptable.

2. **Fourier-Motzkin Complexity**:
   - Full implementation with optimizations is ~500-1000 LOC
   - Conservative version (no optimizations) is ~200 LOC but may be slow
   
   **Recommendation**: Start simple, optimize if performance issues arise.

3. **LLIR Execution Priority**:
   - Should we prioritize LLIR interpreter (TODO #1) before polyhedral work?
   - Or do them in parallel?
   
   **Recommendation**: These are orthogonal - suggest parallel tracks if you have time, otherwise LLIR first per TODO.md.

---

## Summary

This plan gives you:
- ✅ **Backend-agnostic API** via `PolyhedralEngine` trait
- ✅ **Pure Rust path** with native implementation (Phases 1-6)
- ✅ **ISL compatibility** for future (Phase 8)
- ✅ **Incremental delivery** - each phase adds functionality
- ✅ **Backward compatibility** - existing code keeps working

**Estimated Effort**:
- Phase 1-2 (Trait + Affine Ops): 2-3 days
- Phase 3 (ILP): 1-2 days  
- Phase 4 (Fourier-Motzkin): 3-5 days
- Phase 5-6 (Distance + Legality): 2-3 days
- **Total for native backend**: ~2-3 weeks
