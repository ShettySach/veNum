# venum — Implementation Notes

## Compilation Model

**Ahead-of-time, global search.** The full computation graph is compiled once
before any inference runs. Schedule search operates over the entire `PlanGraph`
jointly — not per-operator — so fusion decisions and tile size choices see the
whole picture. This is the Luminal model, and it is the right one for inference:
pay the compilation cost once, amortize over many forward passes.

This rules out tinygrad-style lazy per-kernel scheduling where each kernel is
optimized in isolation. The tradeoff is accepted: longer compile time, better
runtime.

## Backend Strategy

The `CodeGenerator`, `DependenceAnalyzer`, and `ScheduleTransformer` traits in
the spec are the backend interface. Backends are pluggable; the LLIR they receive
is fully concrete (no unresolved search params, no symbolic tile sizes).

**First backend: CPU via Cranelift.**
Cranelift is a good fit — it is a Rust-native code generator with a stable IR,
no LLVM dependency, and direct object file emission. The CPU backend walks the
`LLIRProgram`, emits Cranelift IR (CLIF) for each kernel, and uses
`cranelift-object` to produce a native `.o` that is linked and loaded at the end
of compilation. `cranelift-codegen` handles register allocation, instruction
selection, and calling convention.

`AbstractVectorOp` nodes are lowered in the CPU backend by `lower_vector_op`:
initially to scalar fallbacks, then progressively to AVX2/AVX-512 intrinsics as
that path matures.

**Later backends (in rough priority order):** CUDA (PTX emission via string
codegen or NVVM IR), Metal (MSL string codegen via `metal-rs`), WebGPU (WGSL).
None of these require changes to LLIR or the spec.

## Crate Structure

`poly` holds the ISL FFI and is the only crate with a non-Rust dependency.
Everything else is pure Rust. ISL is linked as a static library via `build.rs`.

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `petgraph` | DAG representation for HLIR and Plan IR graphs |
| `inkwell` / PTX | CUDA codegen (later) |
| ISL crate (C, via FFI) | Polyhedral dependence analysis and legality checking |


`cranelift` and `egg` are already in dependencies

Rust native ISL alternatives for affine expressions (Maybe later) - 
-  good_lp

## Rust-Specific Notes

- **Interned strings for symbols.** `Symbol` should be an interned string type
  (e.g., backed by a global `DashMap<Arc<str>, usize>`) so that `Dim::Sym`
  comparison is a pointer comparison, not a heap allocation.

- **`Arc`-shared immutable schedule nodes.** `ScheduleNode` values in the search
  tree should be `Arc`-wrapped with path-copying on mutation. This makes beam
  search branching cheap: cloning a candidate is O(depth), not O(graph size).

- **`petgraph::StableGraph` for HLIR.** Node and edge indices must remain stable
  across graph mutations (CSE, DCE). `StableGraph` provides this; `Graph` does not.

- **ISL is not `Send`.** `IslDependenceAnalyzer` holds an `isl_ctx` which is
  single-threaded. Wrap it in a `thread_local!` or confine it to the lowering
  thread. Search candidates can be evaluated in parallel; ISL calls happen only
  during legality checks, which are serialized per candidate anyway.
