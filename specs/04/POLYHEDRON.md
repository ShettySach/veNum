# Polyhedral Capabilities Specification

This document defines what polyhedral analysis capabilities the Venum compiler
needs, why it needs them, and where the implementation currently stands.

It complements:

- `specs/04/SPEC.md`
- `specs/04/POLYHEDRAL_NATIVE_PLAN.md`
- `specs/04/POLYHEDRAL_NATIVE_CHECKLIST.md`

The goal here is practical: describe the engine quality bar for a deep learning
compiler and track implemented vs missing capabilities.

## 1. Scope

Polyhedral analysis in Venum is an LLIR-level capability for affine kernels.
It must support:

1. domain and access extraction
2. dependence construction and feasibility queries
3. legality checks for schedule transforms
4. feedback to schedule search and LLIR optimization

Out of scope for this phase:

- full automatic schedule synthesis (Pluto-style global scheduling)
- general non-affine reasoning beyond reject/fallback paths
- mandatory external solver dependency

## 2. Core Theory (Minimal but Complete)

### 2.1 Iteration Domain

Each statement `S` has an integer iteration domain:

```text
DS = { i in Z^n : A*i + B*p + c >= 0, Aeq*i + Beq*p + ceq = 0 }
```

- `i` are loop iterators
- `p` are symbolic runtime parameters

Example:

```text
S[i, j] : 0 <= i < M and 0 <= j < N
```

### 2.2 Access Map

Each memory access is an affine map from statement iterators to memory
coordinates:

```text
R_S : i -> f(i, p)
W_S : i -> g(i, p)
```

Example:

```text
S[i, j] reads A[i, j]
T[i, j] writes B[i, j + 1]
```

### 2.3 Dependence Relation

Dependence exists from source statement instance `x` to sink statement instance
`y` when:

1. source and sink access same memory location
2. source executes before sink in original order
3. both instances are in their domains

Formal shape:

```text
Delta_{S->T} = { x -> y : x in DS and y in DT and W_S(x) = R_T(y) and x <lex y }
```

### 2.4 Legality

A transform is legal if it preserves dependence order.

- Parallelize loop `k`: no carried dependence with positive distance on `k`
- Interchange `(i, j)`: transformed dependence vectors stay lexicographically
  non-negative
- Vectorize loop `k`: no dependence that violates vector lane ordering model

### 2.5 Projection and Feasibility

Most legality queries reduce to emptiness checks over affine constraint systems.

- Projection removes existential variables
- Feasibility decides if a system has integer solutions

Proof-oriented requirement:

- never claim feasible unless proven
- use `Unknown` when proof is unavailable

## 3. Required Capabilities for DL Compiler Quality

This section defines what the compiler needs for a mature, production-grade
polyhedral layer.

### C1. Canonical Affine Algebra

Need:

- deterministic normalization (combine terms, sort terms, drop zeros)
- algebra ops: add/sub/scale/substitute/coefficient lookup
- conversion between LLIR affine and poly affine forms

Example:

```text
2*i + 3*i - j + j + 4  ->  5*i + 4
```

### C2. Affine-Preserving Lowering Contract

Need:

- HLIR `Dim` to LLIR affine conversion for affine expressions
- explicit rejection/fallback path for non-affine dims
- no silent collapse of symbolic/non-affine bounds into fake constants

Example:

```text
N + 7  ->  Param(N) + 7
Sym*Sym -> non-affine path (not silently 1)
```

### C3. Statement Extraction

Need:

- stable statement instances with domain, reads, writes
- affine guard refinement for `If`
- vector memory semantics (gather/scatter loads/stores)

Example:

```text
if (i < N) { A[i] = B[i]; } else { A[i] = C[i]; }
```

Then and else must get different domain constraints.

### C4. Relation System

Need:

- distinct equality and inequality stores
- source/sink-prefixed variable spaces
- memory equality composition centralized
- lexicographic order modeling

### C5. Projection Engine

Need:

- exact elimination where supported (equality substitution, FM)
- strict handling when exact elimination unavailable
- no lossy relaxation in soundness-critical paths

### C6. Feasibility and Emptiness

Need:

- contradiction detection
- exact simplification and bounded checks where possible
- proof-oriented result type (`Feasible`, `Infeasible`, `Unknown`)

### C7. Dependence Analysis Pipeline

Need:

- statement-pair dependence construction
- RAW, WAW, and WAR support as needed by legality clients
- dependence relation payload, not synthetic placeholders

### C8. Legality Queries

Need:

- `can_parallelize`, `can_interchange`, `can_vectorize`
- reduction-aware legality for `GroupReduce`
- conservative unknown handling

### C9. Lowering Integration

Need:

- legality checked after each opt application
- exact opt->transform mapping
- actionable legality failure reasons

### C10. Search and Cost Integration

Need:

- carried dependence facts to prune candidate generation
- reuse/locality signals for cost model
- avoid full heavy re-analysis per candidate branch when possible

### C11. LLIR Polyhedral Optimization

Need:

- at least one real dependence-backed transform in optimize phase
- legality-checked transformations only

## 4. Current Implementation Status (As of this branch)

Status labels:

- Implemented: available and used
- Partial: available but limited/approximate
- Missing: not implemented yet

| Capability | Status | Notes |
|---|---|---|
| C1 Canonical affine algebra | Implemented | `src/core/poly/domain.rs`, `src/core/llir/affine.rs` provide canonicalization, substitution, conversions |
| C2 Affine-preserving lowering contract | Partial | Poly side rejects non-affine via `shape_to_domain_checked`, but lowerer still has non-affine `Dim` fallback to constant `1` in `src/core/lower/mod.rs` |
| C3 Statement extraction | Partial | Statement instances and loop-domain extraction are implemented; affine binary guards and gather/scatter memory effects are now modeled in `src/core/poly/analysis/extract.rs`; boolean composition and full affine normalization of predicates are still limited |
| C4 Relation system | Partial | Constraint system and dependence relation building are present in `src/core/poly/sets/relation.rs`; lexicographic order modeled via order slices |
| C5 Projection engine | Partial | Exact-only elimination path added via `try_project_out_exact` in `src/core/poly/sets/project.rs`; unsupported cases remain unresolved as `None` |
| C6 Feasibility/emptiness | Partial | Proof-oriented feasibility in `src/core/poly/native/feasibility.rs`; returns `Unknown` in many symbolic or hard systems |
| C7 Dependence analysis pipeline | Partial | Kernel dependence uses extracted statements and feasibility in `src/core/poly/analysis/dependence.rs`; currently focuses on RAW/WAW paths and conservative distance extraction |
| C8 Legality queries | Partial | Implemented in `src/core/poly/analysis/legality.rs`; conservative, but direction/distance precision is limited |
| C9 Lowering integration | Implemented | Incremental legality checks after each opt in `src/core/lower/legality.rs` and `src/core/lower/mod.rs` |
| C10 Search/cost integration | Partial | Candidate pruning can use `carried_dep_axes`, but search currently seeds empty carried-dependence info in `src/core/schedule/beam.rs` |
| C11 LLIR polyhedral optimization | Partial | `optimize_llir` currently performs legality-checked interchange only (`src/core/compile.rs`) |

## 5. What Has Been Implemented Recently

Recent critical improvements now in code:

1. exact-only projection for soundness-critical elimination
2. proof-oriented feasibility (`Unknown` instead of optimistic success)
3. lexicographic order slices in dependence construction
4. `access_dependence` now builds real constraints and filters infeasible pairs
5. affine `If` guard refinement in extraction
6. gather/scatter memory effects extracted into read/write maps

These significantly improve soundness over placeholder scaffolding.

## 6. Remaining Work to Reach Mature Engine Quality

### 6.1 Precision and Solver Strength

- stronger exact integer feasibility for symbolic systems
- less frequent `Unknown` where proofs are practical
- better distance and direction derivation than current mostly-unknown fallback

### 6.2 Extraction Completeness

- richer affine predicate extraction (`and`, `or`, nested normal forms)
- better modeling for complex vector index expressions
- robust alias assumptions and disambiguation model

### 6.3 Dependence Coverage

- WAR modeling where required by legality consumers
- improved statement ordering and schedule-dimension reasoning
- relation simplification and canonicalization for stable comparison

### 6.4 Pipeline Integration

- remove non-affine-to-constant fallback in lowering contract
- feed carried dependence facts into real search contexts (not empty defaults)
- integrate reuse/locality features into hardware cost model inputs

### 6.5 Validation Quality Bar

- differential tests against a trusted engine on affine kernels
- property-based tests for projection and feasibility operations
- larger end-to-end legality regression suite with known positive/negative cases

## 7. Worked Examples

### Example A: Pointwise Add (Parallel Legal)

```text
for i in [0, N):
  C[i] = A[i] + B[i]
```

- Memory equality enforces same `i`
- Distance is zero on `i`
- Parallelize on `i` is legal

### Example B: Reduction Carry (Parallel Illegal on Reduce Axis)

```text
for i in [0, M):
  acc = 0
  for k in [0, K):
    acc += A[i, k]
```

- Dependence carries along `k` through accumulator updates
- Parallelize on `k` is illegal unless transformed into reduction-safe form

### Example C: Interchange Guarded by Dependence

```text
S[i, j]: A[i, j] = A[i, j-1] + 1
```

- Dependence along `j`
- Swapping `i` and `j` can violate lexicographic order
- Interchange rejected when transformed distance would be negative

### Example D: Branch Domain Refinement

```text
if (i < N) then A[i] = B[i] else A[i] = 0
```

- Then domain gets `N - 1 - i >= 0`
- Else domain gets `i - N >= 0`
- Dependence queries can reason per branch domain

## 8. Acceptance Criteria for "Mature"

The polyhedral layer is considered mature when all are true:

1. no known unsound success path for legality decisions
2. non-affine input paths are explicit and never silently affine-faked
3. search consumes carried-dependence facts from real analysis
4. legality for parallelize/interchange/vectorize/group-reduce is backed by
   dependence relations with high-confidence direction info
5. end-to-end benchmarks show stable legality behavior and meaningful schedule
   quality improvements on representative DL kernels

## 9. Summary

Venum now has a sound foundational native polyhedral core with conservative
behavior and real relation-based legality checks for affine kernels.

It is no longer placeholder-level, but it is not yet a fully mature polyhedral
engine. The remaining work is mostly about precision, completeness, and deeper
integration with search and cost modeling.
