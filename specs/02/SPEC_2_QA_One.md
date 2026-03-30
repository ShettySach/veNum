## Where Symbolic Shapes Live

**HLIR** is fully symbolic. `Dim::Sym("N")`, `Dim::Sym("seq_len")`, products like `Mul(Sym("N"), Sym("seq_len"))` are all valid. Every tensor type carries a `Vec<Dim>`, and the graph is parameterized over `HLIRGraph::symbols: HashSet<Symbol>`. No concrete sizes anywhere.

**Plan IR** is *mostly* symbolic but with cracks. `TilingSpec::tile_sizes: Vec<TileSize>` can be `TileSize::Const(64)` — the first concrete values enter here, representing hardware-fixed decisions (cache line size, SIMD width). Loop bounds in the planned schedule are still expressed in terms of symbolic dims. `SymbolBinding` either leaves a symbol as `Parameter` (runtime-provided) or replaces it with a `Derived` affine expression.

**LLIR** is the critical boundary. `AffineExpr` is `c0 + c1*v1 + c2*v2 + ...` — strictly affine in loop variables and *parameters*. Those parameters are `Var::Param(Symbol)`. So loop bounds like `upper: AffineExpr { constant: 0, terms: [(1, Param("N"))] }` are still symbolic — `N` is a runtime parameter, not a compile-time constant. This is the polyhedral model's standard treatment: symbolic parameters are universally quantified constants from the loop nest's perspective.

So the answer is: **LLIR can and should remain parametric**. The loop structure, access patterns, tiling, and parallelism strategy are all determined at compile time. The actual runtime values of `N`, `seq_len`, etc. are substituted at dispatch time.

---

## What Actually Requires Concrete Shapes

Very little, if the design is careful:

**Compile time (shape-independent):**
- Loop structure and nesting order
- Tiling factors (hardware-fixed or search-determined)
- Vectorization width
- Access function coefficients (`A[m_outer*64 + m_inner, k_outer*8 + k_inner]`)
- Fusion topology

**Dispatch time (symbolic params bound to concrete values):**
- Loop upper bounds (`upper: M/64` → `upper: 512/64 = 8`)
- Buffer allocation sizes
- Grid/block dimensions for GPU launch

**Genuinely requires recompilation:**
- `dtype` changes — the lowered code is dtype-specialized
- `Rank` changes — the loop nests have fixed dimensionality
- *Static* tile size decisions that were shaped around a specific size (e.g., if search chose `tile_m=64` because it assumed `M >= 64` — a `M=32` at runtime breaks the assumption)
- Specializations that used concrete shape predicates (e.g., an unrolled epilogue assuming `M % 64 == 0`)

---

## The Boundary Problem: `BufferAlloc::shape`

This is where your spec has a gap. `BufferAlloc::shape: Vec<AffineExpr>` is correct — intermediate buffer sizes are affine in the symbolic parameters. But the codegen backend needs to emit an actual allocation. For CPU that's a `malloc` or stack allocation; for GPU that's a `cudaMalloc` or shared memory declaration.

For shared memory in particular, CUDA requires the size to be a **compile-time constant** in `__shared__ float buf[SIZE]`. This forces concretization of anything going into shared memory at compile time. Your spec's `MemoryPlacement::SharedMemory { size: Dim }` carries a `Dim` — the codegen trait will need to evaluate that `Dim` to a constant before emitting GPU code, which means shared memory allocation sizes must be resolvable without runtime `Symbol` values.

---

## LLM Inference: Does Variable `seq_len` Force Recompilation?

Not necessarily, and this is exactly the question that separates good inference compilers from naive ones. The answer depends on what's parametric vs. what got concretized during search.

A typical transformer forward pass has shapes roughly:

```
Q, K, V: [batch, seq_len, num_heads, head_dim]
Attention weights: [batch, num_heads, seq_len, seq_len]
```

If your LLIR loop nests have `seq_len` as a `Var::Param`, the compiled kernel accepts it as a runtime integer and the loops run `for i in 0..seq_len`. No recompilation. This is what llama.cpp does — its kernels are parameterized over sequence length natively.

**But the e-graph search may have implicitly assumed a shape.** If during Plan IR search the cost model evaluated tile sizes with a concrete `seq_len=2048` in mind, and chose `tile_seq=128` because `2048 % 128 == 0`, then at `seq_len=100` that tiling is either illegal (if you asserted divisibility) or requires a tail-handling epilogue that wasn't generated. The spec's `TileSize::Const(64)` is the problem: once a concrete tile size is committed, you've implicitly constrained the valid input domain.

The standard solutions are:

1. **Tile with guards.** Generate the tiled loop plus an epilogue for the remainder. The LLIR `If` statement exists for exactly this. Now the kernel is valid for any `seq_len`, just with a scalar epilogue.

2. **Parametric tiling.** Keep `TileSize::SearchParam("tile_seq")` unresolved through LLIR, bind it at dispatch. The search result is stored as metadata; at dispatch you pick the best known tile size for the actual `seq_len`. This is essentially a dispatch table over tile configurations — what TVM does with its tuned artifact format.

3. **Shape bucketing at the framework level.** Not a compiler concern — the runtime pads or selects from pre-compiled kernels for `seq_len ∈ {128, 256, 512, 1024, 2048, ...}`. PyTorch's `torch.compile` does this; it recompiles on shape change up to a configurable limit, then falls back.

For Venum, option 2 fits the search-first philosophy best: the compiled artifact is a *family* of kernels parameterized by tile size, and the dispatch layer picks the tile configuration with the best measured performance for the actual shapes. This means `SearchParam` needs to survive not just through Plan IR but into the compiled artifact as a dispatch key.

---

## Concrete Recommendation for the Spec

You need an explicit **concretization model** — a documented answer to "at what point does each symbol become concrete, and who provides the value?" Something like:

```rust
pub enum SymbolResolution {
    /// Fixed at compile time (dtype, rank)
    CompileTime(i64),
    /// Fixed at search/tune time (tile sizes, after cost model search)
    TuneTime(i64),
    /// Bound at kernel dispatch (batch, seq_len, N, M, K)
    RuntimeParameter,
    /// Computed from runtime params (P_NM = N * M after reshape)
    RuntimeDerived { expr: AffineExpr },
}
```

The split between `TuneTime` and `RuntimeParameter` is the key distinction. `tile_m` is tune-time: fixed after search, but potentially re-searched for a new hardware target. `seq_len` is runtime: never fixed, must be valid for all values (or a range you explicitly commit to supporting).

LLIR should only contain `CompileTime` and `RuntimeParameter` symbols. `TuneTime` symbols must be resolved before LLIR emission — they produce concrete `i64` loop steps and buffer sizes, with guards generated for the non-divisible case.
