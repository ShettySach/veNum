# Native Polyhedral Implementation Plan

## Goal

Build the polyhedral layer described in [SPEC.md](file:///home/sword/Desktop/code/venum/specs/04/SPEC.md) as a real part of the compiler pipeline, not just as helper syntax around affine expressions.

Today, the repository already has the beginnings of the right vocabulary:

- [`src/core/poly/domain.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/domain.rs) defines `Aff`, `Constraint`, `Domain`, and `PolyVar`
- [`src/core/poly/access_map.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/access_map.rs) defines `AccessMap`
- [`src/core/poly/native.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/native.rs) exposes a `NativeDependenceAnalyzer`

But the current implementation is not yet polyhedral in the sense intended by the spec:

- dependence analysis only compares equalized indices for same-buffer accesses and returns a synthetic zero distance
- loop bounds are mostly concrete placeholders in lowering rather than derived from symbolic affine domains
- legality checks do not reason from a true dependence polyhedron
- search does not consume legality/cost facts derived from polyhedral structure
- LLIR optimization is still a placeholder in [`src/core/compile.rs`](file:///home/sword/Desktop/code/venum/src/core/compile.rs#L29-L31)

This plan describes what is needed for a native Rust implementation and how to stage it.

## What Polyhedrality Means Here

The spec says LLIR is where polyhedral dependence analysis lives. Concretely, that means the compiler should be able to represent and operate on:

1. Iteration domains
2. Access maps
3. Dependence relations
4. Schedule legality predicates
5. Schedule transforms as rewrites over affine loop nests

### Minimal Theory We Need

#### Iteration Domain

An iteration domain is the integer set of loop instances where a statement executes.

Example:

```text
S[i, j] : 0 <= i < M and 0 <= j < N
```

This is already close to [`Domain`](file:///home/sword/Desktop/code/venum/src/core/poly/domain.rs), but today `shape_to_domain` only handles a very small subset well and collapses non-constant dimensions too aggressively.

#### Access Map

An access map maps a statement instance to a memory location.

Example:

```text
S[i, j] -> A[i, j]
S[i, j] -> B[i, k]
```

This is what [`AccessMap`](file:///home/sword/Desktop/code/venum/src/core/poly/access_map.rs) wants to model, but the current code only stores affine outputs and does not yet support the operations we need on maps.

#### Dependence Relation

A dependence relation connects a producer iteration to a consumer iteration when both access the same memory cell and ordering matters.

Example:

```text
{ S[i, k] -> T[i, j, k] : same A[i, k] element participates in both sides }
```

For legality, the key question is not just whether two accesses alias, but whether there exists a pair of source/sink iterations that violates a proposed reordering.

#### Projection / Elimination

Projection removes intermediate variables from a constraint system to derive a smaller relation or set.

We need it for:

- collapsing existential variables introduced during dependence construction
- simplifying composed relations
- deriving per-loop legality summaries

Fourier-Motzkin elimination is enough for a first native implementation because we only need affine inequalities over integers, with modest problem sizes.

#### Distance / Direction Information

For transforms like interchange and parallelization, we need a compact summary of whether a dependence flows forward in a loop dimension.

We do not need full Pluto-style scheduling up front. A first useful target is:

- exact constant distance when derivable cheaply
- otherwise direction-style summaries such as `<`, `=`, `>` per loop axis
- fallback to conservative “unknown, therefore illegal” when proof is incomplete

That gives us sound legality without requiring a complete scheduling solver in phase one.

## What We Need In The Codebase

### 1. A Real Polyhedral Core

The current [`src/core/poly`](file:///home/sword/Desktop/code/venum/src/core/poly/mod.rs) module has data structures, but not an engine. We need native implementations for:

- affine normalization and simplification
- domain construction from LLIR loop nests
- access-map construction from `MemoryAccess`
- relation composition
- feasibility checking
- variable elimination / projection
- legality queries over relations

### 2. A Better Boundary Between LLIR And Poly

Polyhedral analysis should operate on LLIR loop nests and memory accesses, but it should not depend on ad hoc pattern matching over raw `Stmt`s every time.

We need a normalized analysis view, for example:

```rust
pub struct StatementInstance {
    pub stmt_id: usize,
    pub domain: Domain,
    pub reads: Vec<AccessMap>,
    pub writes: Vec<AccessMap>,
}
```

This becomes the bridge from [`src/core/llir`](file:///home/sword/Desktop/code/venum/src/core/llir/mod.rs) into the polyhedral engine.

### 3. Affine Normalization For Shapes And Indices

The spec allows symbolic and runtime-derived dimensions, but the current lowering path still does things like defaulting non-constant upper bounds to `1` in [`build_base_kernel`](file:///home/sword/Desktop/code/venum/src/core/lower/mod.rs#L59-L72). That blocks meaningful polyhedral reasoning.

We need a normalization pass that:

- converts HLIR `Dim` expressions into affine LLIR bounds when possible
- introduces explicit runtime parameters for `RuntimeDerived` symbols from the spec
- rejects or materializes genuinely non-affine cases before they reach the affine engine

### 4. Dependence Analysis That Produces Real Relations

The current analyzer in [`src/core/poly/native.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/native.rs) creates equality constraints for matching indices, but it does not:

- distinguish statement domains for source and sink instances
- encode execution order between source and sink
- derive precise distance or direction summaries
- reason about write-write and write-read with schedule context

The new analyzer must construct a proper dependence problem:

- source domain constraints
- sink domain constraints
- memory-equality constraints for aliased accesses
- execution-order constraints for original schedule
- optional projection to remove memory coordinates or temporary variables

### 5. Legality Integrated Into Lowering And LLIR Optimization

Right now legality is checked opt-by-opt in [`src/core/lower/legality.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/legality.rs), but the transform mapping is lossy and in one case incorrect: `PadTo` is treated like an interchange surrogate.

We need legality checks that answer questions such as:

- can loop `i` become parallel?
- can loops `i` and `j` be interchanged?
- can a loop be vectorized without crossing a loop-carried dependence?
- can producer and consumer be merged at a given loop depth?

These should consume dependence summaries, not handcrafted heuristics.

### 6. Search Inputs Derived From Polyhedral Facts

The scheduler in [`src/core/schedule/search.rs`](file:///home/sword/Desktop/code/venum/src/core/schedule/search.rs) is search-first, which matches the spec, but the search space still needs legal pruning informed by the polyhedral layer.

Specifically, the polyhedral layer should provide:

- transform legality
- fusibility checks between HLIR nodes after lowering to candidate kernels
- reuse/locality features for cost modeling
- reduction-specific facts for `GroupReduce`

### 7. Tests At The Right Level

The current tests in [`src/core/poly/tests.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/tests.rs) mostly verify scaffolding. We need a compact but high-value regression suite around:

- domain construction
- affine composition
- dependence existence
- parallel legality rejection/acceptance
- interchange legality
- reduction legality
- symbolic parameter handling

## Recommended Design

### Keep It As A Submodule First

Recommendation: keep this as a submodule under [`src/core/poly`](file:///home/sword/Desktop/code/venum/src/core/poly/mod.rs), not a separate crate yet.

Why:

- the native implementation will share `LLIR`, `HLIR`, and scheduling types heavily
- there are currently no heavyweight non-Rust dependencies forcing isolation
- iteration while the design is still unstable will be much faster inside the main crate
- a native engine is compiler-internal, not yet a reusable public library boundary

When to split to a subcrate later:

- if an ISL backend is added behind a feature flag and brings build-system complexity
- if compile times become painful
- if we want solver backends and poly utilities reusable outside `venum`
- if we want a stable internal API between compiler frontend and backend crates

Recommendation summary:

- now: submodule
- later, if ISL/solver backends grow: subcrate `venum-poly`

## Proposed Module Layout

```text
src/core/poly/
  mod.rs
  affine/
    mod.rs
    normalize.rs
    simplify.rs
    substitute.rs
  sets/
    mod.rs
    domain.rs
    relation.rs
    project.rs
  analysis/
    mod.rs
    extract.rs
    dependence.rs
    legality.rs
  native/
    mod.rs
    fm.rs
    simplex.rs
    distance.rs
  tests.rs
```

### Responsibilities

- `affine`: algebra on affine expressions and normalization from HLIR/LLIR forms
- `sets`: integer-set and relation operations independent of compiler policy
- `analysis`: compiler-facing extraction from LLIR and legality questions
- `native`: concrete algorithms used by the native backend

This is a better fit than today’s flat layout because it separates math primitives from compiler integration.

## Implementation Plan

### Phase 0: Tighten The IR Contract

Before adding algorithms, make the IR expose the data the polyhedral layer needs.

Tasks:

- add a normalized way to extract statement instances from LLIR bodies
- make loop bounds retain affine expressions instead of collapsing dynamic cases to constants
- represent runtime parameters and runtime-derived symbols explicitly in LLIR bounds/accesses
- make `MemoryAccess` extraction preserve full rank and affine index structure

Deliverable:

- an LLIR-to-poly extraction path that is deterministic and testable without running scheduling

### Phase 1: Affine Algebra And Normalization

Tasks:

- canonicalize affine expressions by sorting/combining terms
- implement `add`, `sub`, scalar multiply, substitution, coefficient lookup
- implement conversion from [`AffineExpr`](file:///home/sword/Desktop/code/venum/src/core/llir/affine.rs) to poly `Aff`
- normalize HLIR `Dim` into affine form or an explicit “not affine” outcome

Theory:

- affine expressions form a closed algebra under addition, subtraction, and substitution by affine terms
- canonical form is required so equality and simplification are stable

Deliverable:

- a reliable affine kernel used everywhere else

### Phase 2: Domain And Access Extraction

Tasks:

- build `Domain` directly from `LoopNest.loops`
- extract per-statement read/write access maps from `Stmt`
- assign stable statement ids during extraction
- encode predicates from `If` guards when those guards are affine; conservatively ignore or split otherwise

Theory:

- each statement has its own iteration domain, even when nested in the same loop nest
- control predicates refine the domain

Deliverable:

- `Vec<StatementInstance>` for any kernel

### Phase 3: Native Relation Operations

Tasks:

- implement map equality composition for memory-alias constraints
- support projection/elimination via Fourier-Motzkin for inequality systems
- keep equalities in normalized form and reduce them before elimination
- add cheap redundancy cleanup to prevent blow-up

Theory:

- dependence construction is mostly set intersection plus projection
- projection is the core operation needed to remove memory coordinates and temporary variables

Deliverable:

- the ability to build a dependence relation from source/sink statements and accesses

### Phase 4: Feasibility And Order Checking

Tasks:

- implement a native feasibility solver for small affine integer systems
- start with a pragmatic hybrid:
  - equality simplification and bound propagation first
  - bounded branch-and-check or simplex-style rational check plus integer validation second
- only add a full external LP/MILP dependency if the native path proves too weak

Theory:

- legality queries reduce to emptiness of affine integer sets/relations
- most compiler examples are small enough that a simple native solver can work initially if we simplify aggressively first

Deliverable:

- `is_empty` / `is_feasible` primitives over domains and relations

### Phase 5: Dependence Analyzer Rewrite

Replace [`NativeDependenceAnalyzer`](file:///home/sword/Desktop/code/venum/src/core/poly/native.rs) with a pipeline that:

1. extracts normalized statements from a kernel
2. pairs candidate write/read or write/write accesses by buffer
3. builds dependence constraints from domains, access equality, and original execution order
4. checks feasibility
5. computes a conservative direction or distance summary
6. emits `Dependence` values backed by real relations

Deliverable:

- a sound dependence analyzer for affine LLIR kernels

### Phase 6: Transform Legality Queries

Tasks:

- implement `can_parallelize(loop_var)` from dependence direction data
- implement `can_interchange(outer, inner)` by checking lexicographic order preservation
- implement `can_vectorize(loop_var)` conservatively using innermost dependence information and stride/alignment constraints
- implement legality for reduction transforms, especially `GroupReduce`

Deliverable:

- legality checks in [`src/core/lower/legality.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/legality.rs) that are backed by polyhedral facts rather than placeholders

### Phase 7: Polyhedral LLIR Optimization

Tasks:

- implement the LLIR optimization hook in [`src/core/compile.rs`](file:///home/sword/Desktop/code/venum/src/core/compile.rs#L29-L31)
- start with one or two sound transforms only:
  - interchange when proven legal and cost-improving
  - cache-read/write placement when reuse is proven
- keep transformation count small; let search remain the main decision-maker

Deliverable:

- actual LLIR optimization using dependence-guided transforms

### Phase 8: Feed Search And Fusion

Tasks:

- expose fusibility checks and legality summaries to the scheduler
- use polyhedral facts to reject illegal opt sequences early
- enrich `KernelContext` or a sibling analysis structure with reuse and dependence data
- use those features in the hardware cost model, especially for memory traffic estimates

Deliverable:

- search remains heuristic and beam-based, but is constrained by sound legality information

## Algorithm Choices For A Native First Version

### What To Implement Natively

- affine normalization and substitution
- domain/access extraction
- equality solving and simplification
- Fourier-Motzkin projection for inequalities
- conservative direction/distance derivation
- emptiness/feasibility for small systems

### What Not To Overbuild Immediately

- full Presburger arithmetic library
- general piecewise quasi-affine schedules
- automatic schedule synthesis like Pluto
- non-affine reasoning beyond “normalize or reject”

That keeps the scope aligned with the spec: search-first scheduling with a polyhedral legality foundation, not a full polyhedral optimizer framework.

## Concrete Repository Changes

### Files To Rework

- [`src/core/poly/native.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/native.rs)
- [`src/core/poly/domain.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/domain.rs)
- [`src/core/poly/access_map.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/access_map.rs)
- [`src/core/lower/mod.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/mod.rs)
- [`src/core/lower/legality.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/legality.rs)
- [`src/core/compile.rs`](file:///home/sword/Desktop/code/venum/src/core/compile.rs)
- [`src/core/llir/affine.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/affine.rs)
- [`src/core/traits.rs`](file:///home/sword/Desktop/code/venum/src/core/traits.rs)

### Files To Add

- `src/core/poly/analysis/extract.rs`
- `src/core/poly/analysis/dependence.rs`
- `src/core/poly/analysis/legality.rs`
- `src/core/poly/affine/normalize.rs`
- `src/core/poly/sets/relation.rs`
- `src/core/poly/native/fm.rs`
- `src/core/poly/native/distance.rs`

## Suggested Milestones

### Milestone 1

Affine loop bounds and access extraction work for elementwise and reduction kernels.

### Milestone 2

Dependence existence and parallel legality work soundly for affine kernels.

### Milestone 3

Interchange legality and one real LLIR optimization are enabled.

### Milestone 4

Search uses legality pruning from the polyhedral layer.

## Risks And How To Manage Them

### Constraint Explosion

Fourier-Motzkin can blow up quickly.

Mitigation:

- keep problems small and local to one kernel
- simplify equalities first
- use conservative early exits
- project only when needed

### Symbolic Shapes Becoming Non-Affine In Practice

Mitigation:

- introduce explicit normalization to runtime-derived params
- fail closed on truly non-affine cases
- keep the non-affine fallback path outside the native poly engine

### Overcoupling Search And Poly Analysis

Mitigation:

- make poly analysis produce facts and legality queries
- keep actual schedule selection in search
- do not turn the poly layer into a second scheduler

## Final Recommendation

Implement the native polyhedral layer as an internal submodule under `src/core/poly` first.

The first target should be sound affine dependence analysis and transform legality for LLIR kernels, not full automatic schedule synthesis. Once that foundation exists, plug it into:

- lowering legality
- LLIR optimization
- search pruning
- cost-model features

If the native layer later grows a separate backend matrix, optional solver integrations, or an ISL bridge, then split it into a dedicated subcrate. Right now, a subcrate would add boundary friction before the design is mature enough to benefit from it.
