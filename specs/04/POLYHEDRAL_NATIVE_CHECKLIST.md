# Native Polyhedral Implementation Checklist

This is the execution companion to [POLYHEDRAL_NATIVE_PLAN.md](file:///home/sword/Desktop/code/venum/specs/04/POLYHEDRAL_NATIVE_PLAN.md).

The goal here is not more theory. The goal is implementation order:

- what to change first
- which files move together
- what each step must prove before the next one starts

## Working Decision

Implement this as an internal submodule under [`src/core/poly`](file:///home/sword/Desktop/code/venum/src/core/poly/mod.rs) first.

Do not split to a subcrate yet.

Reason:

- the code is still shaping its core interfaces
- the poly layer depends directly on LLIR, lowering, and scheduling details
- we want fast iteration and easy refactors across module boundaries

## Target End State

By the end of this checklist, the compiler should be able to:

1. extract affine statement domains from LLIR
2. extract affine read/write access maps per statement
3. build real dependence relations for affine kernels
4. answer legality questions for parallelize, interchange, vectorize, and reduction-aware transforms conservatively and soundly
5. use those legality answers during lowering and LLIR optimization
6. provide legality and reuse facts back to schedule search

## Order Of Work

## Step 1: Fix The Affine Foundations

### Why First

Everything else depends on affine expressions behaving predictably. Right now there are two parallel affine vocabularies:

- [`src/core/llir/affine.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/affine.rs)
- [`src/core/poly/domain.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/domain.rs)

They are too minimal and too disconnected for real analysis.

### Files

- [`src/core/llir/affine.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/affine.rs)
- [`src/core/poly/domain.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/domain.rs)
- `src/core/poly/affine/mod.rs`
- `src/core/poly/affine/normalize.rs`
- `src/core/poly/affine/substitute.rs`

### Tasks

- add canonicalization helpers for affine terms:
  - combine like terms
  - sort terms deterministically
  - drop zero coefficients
- add arithmetic helpers:
  - `add`
  - `sub`
  - `scale`
  - `substitute`
  - `coefficient_of`
- add conversion between LLIR affine expressions and poly affine expressions
- make equality and debug output stable enough for tests

### Done When

- the same affine expression always normalizes to the same representation
- there is one obvious path from `AffineExpr` to `Aff`
- unit tests cover canonicalization and substitution

### Suggested Tests

- combining duplicate loop terms
- substitution of one loop variable with another affine form
- symbolic parameter retention
- normalization of `x - x + 3`

## Step 2: Preserve Affine Information In LLIR

### Why Next

The analysis cannot work if lowering throws affine structure away.

The most important current issue is in [`src/core/lower/mod.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/mod.rs#L59-L72), where non-constant dimensions effectively collapse to `1` for loop upper bounds.

### Files

- [`src/core/lower/mod.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/mod.rs)
- [`src/core/hlir/dim.rs`](file:///home/sword/Desktop/code/venum/src/core/hlir/dim.rs)
- [`src/core/poly/domain.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/domain.rs)
- [`src/core/llir/loop_nest.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/loop_nest.rs)

### Tasks

- introduce a conversion path from affine `Dim` to LLIR `AffineExpr`
- explicitly classify dimensions as:
  - affine and directly lowerable
  - runtime-derived affine parameters
  - non-affine and therefore not admissible to the native affine engine
- stop defaulting dynamic loop bounds to fake constants
- preserve dynamic affine bounds in `Loop.upper`
- keep reduction bounds affine too

### Done When

- a symbolic extent like `N` becomes an LLIR param, not `1`
- affine expressions like `N + 7` can appear in loop bounds
- genuinely non-affine cases fail clearly or go through a separate fallback path

### Suggested Tests

- loop bound from a single runtime symbol
- loop bound from affine `Add(Const, Sym)`
- rejection of unsupported non-affine forms

## Step 3: Extract A Polyhedral View From LLIR

### Why Next

Before doing dependence analysis, we need a stable analysis representation for statements, domains, and accesses.

### Files

- `src/core/poly/analysis/mod.rs`
- `src/core/poly/analysis/extract.rs`
- [`src/core/llir/stmt.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/stmt.rs)
- [`src/core/llir/memory.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/memory.rs)
- [`src/core/poly/access_map.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/access_map.rs)

### Tasks

- define a normalized analysis struct, for example:

```rust
pub struct StatementInstance {
    pub stmt_id: usize,
    pub domain: Domain,
    pub reads: Vec<AccessMap>,
    pub writes: Vec<AccessMap>,
}
```

- walk `LoopNest` recursively and accumulate active loop bounds
- refine domains with affine `If` guards when possible
- extract read/write accesses from:
  - `Assign`
  - `Accumulate`
  - vector ops that imply loads/stores
- give each statement a stable id

### Done When

- any affine kernel can be converted into `Vec<StatementInstance>`
- extracted access maps preserve full rank and indices
- extraction does not rely on ad hoc legality code paths

### Suggested Tests

- elementwise kernel with two loads and one store
- reduction kernel with accumulator read/write
- nested `If` that contributes affine constraints

## Step 4: Introduce Relation And Set Operations

### Why Next

Once extraction exists, we need the math operations that let us combine domains and access equalities into dependences.

### Files

- `src/core/poly/sets/mod.rs`
- `src/core/poly/sets/relation.rs`
- `src/core/poly/sets/project.rs`
- [`src/core/poly/access_map.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/access_map.rs)
- [`src/core/poly/domain.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/domain.rs)

### Tasks

- define an internal relation/set representation richer than the current raw lists
- implement relation construction from:
  - source domain
  - sink domain
  - memory-equality constraints
  - source-before-sink execution constraints
- add projection/elimination APIs
- keep equalities and inequalities distinct internally

### Done When

- the code can construct a dependence relation object without yet deciding feasibility
- memory-equality composition is centralized in one place
- projection has a stable interface even if internals remain conservative initially

### Suggested Tests

- equal-access dependence relation for same-buffer accesses
- projected relation removes temporary memory coordinates
- relation build for two-dimensional access maps

## Step 5: Implement Native Projection

### Why Next

Projection is the first real algorithmic step that makes the relation engine useful.

### Files

- `src/core/poly/native/mod.rs`
- `src/core/poly/native/fm.rs`
- `src/core/poly/sets/project.rs`

### Tasks

- implement Fourier-Motzkin elimination for affine inequalities
- simplify equalities before elimination
- add light redundancy cleanup:
  - deduplicate identical constraints
  - drop trivially true inequalities
  - detect obvious contradictions early
- keep the implementation deliberately conservative if integer exactness is unclear

### Done When

- projected systems remain small enough for typical kernel analyses
- obvious infeasible projections are rejected
- elimination is covered by focused unit tests

### Suggested Tests

- eliminate one variable from simple box constraints
- eliminate a variable from a coupled system
- infeasible constraint system detected during elimination

## Step 6: Add Native Feasibility Checking

### Why Next

Legality depends on emptiness queries. This is where the native engine starts answering actual compiler questions.

### Files

- `src/core/poly/native/simplex.rs`
- `src/core/poly/native/mod.rs`
- `src/core/poly/sets/relation.rs`

### Tasks

- implement a pragmatic feasibility checker for small affine integer systems
- start simple:
  - equality simplification
  - interval/bound propagation
  - rational feasibility check or bounded search for leftover small systems
- make the API explicitly conservative if exact proof is unavailable
- expose `is_feasible` and `is_empty`

### Done When

- dependence existence can be answered soundly for small affine kernels
- the engine returns conservative failure instead of unsound success

### Suggested Tests

- satisfiable rectangular domain
- infeasible lower-vs-upper bound conflict
- equality forcing one legal solution
- symbolic-parameter feasibility where parameters remain unconstrained

## Step 7: Rewrite Dependence Analysis Around The New Core

### Why Next

At this point the math layer is useful enough to replace the placeholder analyzer in [`src/core/poly/native.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/native.rs).

### Files

- [`src/core/poly/native.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/native.rs)
- `src/core/poly/analysis/dependence.rs`
- [`src/core/traits.rs`](file:///home/sword/Desktop/code/venum/src/core/traits.rs)
- [`src/core/llir/dependence.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/dependence.rs)

### Tasks

- replace index-equality-only dependence building with statement-pair analysis
- model:
  - write-read
  - write-write
  - optionally read-write if needed for later legality consumers
- attach the real relation to each `Dependence`
- derive distance or direction summaries conservatively
- keep the public `DependenceAnalyzer` trait stable if possible

### Done When

- `analyze_kernel` is driven by extracted statements and feasibility queries
- dependencies no longer default to synthetic zero vectors
- dependence output is sound for affine elementwise and reduction kernels

### Suggested Tests

- loop-carried RAW dependence
- independent elementwise accesses with no dependence
- reduction-carried dependence

## Step 8: Implement Real Legality Queries

### Why Next

Only now do we have enough information to answer schedule legality correctly.

### Files

- `src/core/poly/analysis/legality.rs`
- [`src/core/lower/legality.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/legality.rs)
- [`src/core/traits.rs`](file:///home/sword/Desktop/code/venum/src/core/traits.rs)

### Tasks

- implement legality helpers for:
  - `can_parallelize`
  - `can_interchange`
  - `can_vectorize`
  - reduction-aware checks for `GroupReduce`
- stop using lossy transform substitutions where the transform kind does not match the opt semantics
- make unknown cases conservative

### Done When

- `Parallelize` is rejected exactly when a carried dependence exists on that loop
- `Interchange` legality follows lexicographic ordering constraints
- vectorization legality does not silently ignore loop-carried memory dependences
- `PadTo` and `GroupReduce` are checked with their own semantics

### Suggested Tests

- positive-distance dependence blocks parallelization
- fully pointwise loop allows parallelization
- legal interchange on independent loops
- illegal interchange on carried dependence

## Step 9: Wire Legality Back Into Lowering

### Why Next

The lowerer already checks legality incrementally. Once the legality layer is real, we should upgrade that path instead of inventing another one.

### Files

- [`src/core/lower/mod.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/mod.rs)
- [`src/core/lower/apply_opt.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/apply_opt.rs)
- [`src/core/lower/legality.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/legality.rs)

### Tasks

- keep incremental legality checking after each opt application
- ensure opt-to-transform mapping is exact rather than approximate
- add any missing loop metadata needed by legality checks
- verify `GroupReduce` structure against reduction legality facts

### Done When

- the lowerer rejects illegal opt sequences for the right reason
- reduction transforms are checked against reduction dependencies rather than generic parallel heuristics

### Suggested Tests

- illegal opt sequence fails during lowering
- legal tiled+parallelized pointwise kernel survives lowering
- illegal group reduction on non-reduce loop still fails clearly

## Step 10: Turn On LLIR Polyhedral Optimization

### Why Next

The spec wants LLIR optimization to be dependence-guided. The hook already exists and is currently empty in [`src/core/compile.rs`](file:///home/sword/Desktop/code/venum/src/core/compile.rs#L29-L31).

### Files

- [`src/core/compile.rs`](file:///home/sword/Desktop/code/venum/src/core/compile.rs)
- new `src/core/optimize/` module or keep under `src/core/poly/analysis/`
- [`src/core/traits.rs`](file:///home/sword/Desktop/code/venum/src/core/traits.rs)

### Tasks

- pick one or two transformations only for the first pass:
  - interchange
  - cache read/write placement
- run legality before committing any transform
- keep optimization optional and easy to disable while the engine matures

### Done When

- `optimize_llir` is no longer a no-op
- at least one transformation uses dependence-derived legality

### Suggested Tests

- LLIR optimization preserves outputs on an end-to-end kernel
- optimization refuses illegal interchange

## Step 11: Feed Search With Polyhedral Facts

### Why Last

Search should consume legality facts only after the analysis is trustworthy.

### Files

- [`src/core/schedule/search.rs`](file:///home/sword/Desktop/code/venum/src/core/schedule/search.rs)
- [`src/core/schedule/candidates.rs`](file:///home/sword/Desktop/code/venum/src/core/schedule/candidates.rs)
- [`src/core/cost/hardware.rs`](file:///home/sword/Desktop/code/venum/src/core/cost/hardware.rs)
- new analysis glue under `src/core/poly/analysis/`

### Tasks

- expose cheap legality summaries to schedule search
- prune impossible opts early
- add reuse/locality features for cost estimation where available
- do not make search depend on heavyweight full-kernel reanalysis for every branch if avoidable

### Done When

- candidate generation can reject obviously illegal opts early
- schedule search uses dependence-driven pruning without changing its beam-search nature

### Suggested Tests

- candidate list excludes illegal parallelization on carried-dependence loops
- legal pointwise candidates remain available

## File-By-File Change Queue

This is the shortest practical order to edit files without fighting the compiler constantly.

### Queue A: Foundational Types

1. [`src/core/llir/affine.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/affine.rs)
2. [`src/core/poly/domain.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/domain.rs)
3. [`src/core/poly/mod.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/mod.rs)
4. add `src/core/poly/affine/mod.rs`
5. add `src/core/poly/affine/normalize.rs`
6. add `src/core/poly/affine/substitute.rs`

### Queue B: LLIR Preservation

1. [`src/core/lower/mod.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/mod.rs)
2. [`src/core/llir/loop_nest.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/loop_nest.rs)
3. [`src/core/hlir/dim.rs`](file:///home/sword/Desktop/code/venum/src/core/hlir/dim.rs)

### Queue C: Extraction Layer

1. add `src/core/poly/analysis/mod.rs`
2. add `src/core/poly/analysis/extract.rs`
3. [`src/core/poly/access_map.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/access_map.rs)
4. [`src/core/llir/memory.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/memory.rs)
5. [`src/core/llir/stmt.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/stmt.rs)

### Queue D: Set/Relation Engine

1. add `src/core/poly/sets/mod.rs`
2. add `src/core/poly/sets/relation.rs`
3. add `src/core/poly/sets/project.rs`
4. add `src/core/poly/native/mod.rs`
5. add `src/core/poly/native/fm.rs`
6. add `src/core/poly/native/simplex.rs`

### Queue E: Analysis Integration

1. add `src/core/poly/analysis/dependence.rs`
2. add `src/core/poly/analysis/legality.rs`
3. [`src/core/poly/native.rs`](file:///home/sword/Desktop/code/venum/src/core/poly/native.rs)
4. [`src/core/llir/dependence.rs`](file:///home/sword/Desktop/code/venum/src/core/llir/dependence.rs)
5. [`src/core/traits.rs`](file:///home/sword/Desktop/code/venum/src/core/traits.rs)

### Queue F: Compiler Wiring

1. [`src/core/lower/legality.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/legality.rs)
2. [`src/core/lower/apply_opt.rs`](file:///home/sword/Desktop/code/venum/src/core/lower/apply_opt.rs)
3. [`src/core/compile.rs`](file:///home/sword/Desktop/code/venum/src/core/compile.rs)
4. [`src/core/schedule/search.rs`](file:///home/sword/Desktop/code/venum/src/core/schedule/search.rs)
5. [`src/core/schedule/candidates.rs`](file:///home/sword/Desktop/code/venum/src/core/schedule/candidates.rs)
6. [`src/core/cost/hardware.rs`](file:///home/sword/Desktop/code/venum/src/core/cost/hardware.rs)

## Verification Gates

Do not move to the next stage without these gates.

### Gate 1: Affine Soundness

- unit tests for normalization and substitution pass
- symbolic affine bounds survive lowering

### Gate 2: Extraction Soundness

- statement extraction produces expected domains/accesses for elementwise and reduction kernels

### Gate 3: Dependence Soundness

- analyzer finds carried dependences where expected
- analyzer reports no false independence on known dependent examples

### Gate 4: Legality Soundness

- illegal transforms are rejected conservatively
- legal pointwise transforms still pass

### Gate 5: Pipeline Integration

- end-to-end compiler tests still pass
- at least one end-to-end test uses the real dependence analyzer rather than [`NoOpDependenceAnalyzer`](file:///home/sword/Desktop/code/venum/src/core/dep.rs)

## First Deliverable Slice

If this is implemented incrementally, the best first slice is:

1. Step 1: affine foundations
2. Step 2: preserve affine bounds in LLIR
3. Step 3: statement extraction
4. partial Step 7: dependence existence without full distance computation

That gives a useful checkpoint quickly:

- the compiler can represent affine structure properly
- the analyzer can answer whether affine dependences exist
- legality can start conservatively with existence-based answers even before richer distance summaries land

## What To Avoid

- do not start with full automatic schedule synthesis
- do not introduce a heavyweight solver abstraction before the internal data model is stable
- do not let non-affine cases silently degrade into fake affine constants
- do not wire search to poly analysis before the legality layer is sound

## Recommended PR Breakdown

If you want to land this in reviewable chunks, use roughly this split:

1. affine cleanup and LLIR affine preservation
2. LLIR statement/access extraction
3. relation representation plus projection
4. native feasibility and dependence analysis rewrite
5. legality rewrite in lowering
6. LLIR optimization enablement
7. search and cost-model integration

That sequence keeps each PR meaningful and testable while preserving the design direction from the main plan.
