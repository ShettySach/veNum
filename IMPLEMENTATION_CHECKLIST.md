# Shape-Aware Egglog Refactor Checklist

This checklist turns [SPEC.md](file:///home/sword/Desktop/code/venum/SPEC.md) into an implementation sequence against the current codebase.

## Tracking Rules

- Mark a checkbox complete only when code, tests, and call sites are updated.
- If a step changes IR semantics, add or update tests in the same change.
- Do not carry Rust-side canonicalization shortcuts into the new system.
- Keep the old [src/core/hlir/egraph.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph.rs) alive only until parity is proven.

## 0. Baseline And Guardrails

- [ ] Re-read [SPEC.md](file:///home/sword/Desktop/code/venum/SPEC.md) and lock the current acceptance criteria into the work plan.
- [ ] Inventory all current entry points that call `egglog_algebraic` in [src/core/hlir/egraph.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph.rs).
- [ ] Snapshot current behavior with focused regression tests for:
  - [ ] add-zero elimination
  - [ ] mul-one elimination
  - [ ] true barrier input simplification through `Reduce`
  - [ ] reshape/permute/expand simplification already covered today
- [ ] Add explicit regression tests that fail under the current design but are required by the new spec:
  - [x] scalar plus tensor add
  - [x] scalar plus tensor mul
  - [x] `(x * 2 + x) + (x * 2 + x) -> 6 * x`
  - [ ] illegal reshape sinking when only final target shape matches
  - [ ] exact `F16` and `BF16` scalar folding

## 1. HLIR Prep

### Core IR

- [x] Add `Op::Broadcast { input: NodeId, shape: Vec<Dim> }` to [src/core/hlir/op.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/op.rs).
- [x] Update `Op::name()` in [src/core/hlir/op.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/op.rs) for `Broadcast`.
- [x] Update `Op::inputs()` in [src/core/hlir/op.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/op.rs) for `Broadcast`.

### Graph Construction

- [x] Add `HLIRGraph::broadcast(&mut self, input: NodeId, shape: Vec<Dim>) -> NodeId` to [src/core/hlir/graph.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/graph.rs).
- [x] Define shape inference for `Broadcast` in [src/core/hlir/graph.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/graph.rs).
- [x] Define layout/stride inference for `Broadcast` in [src/core/hlir/graph.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/graph.rs).
- [ ] Decide and document whether `Expand` remains as same-rank-only or becomes a thin wrapper over `Broadcast`.

### Remapping And Plumbing

- [x] Update `remap_op_inputs` in [src/core/hlir/optimize.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/optimize.rs) to support `Broadcast`.
- [ ] Update any HLIR pretty-printing, debugging, or traversal code that assumes the current op set.
- [ ] Ensure [src/core/hlir/mod.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/mod.rs) re-exports any new module structure cleanly.

### Completion Criteria

- [x] A scalar broadcast can be represented in HLIR without `Reshape + Expand`.
- [x] Existing HLIR tests still pass after adding `Broadcast`.

## 2. Replace Single-File Egraph Implementation

- [x] Create `src/core/hlir/egraph/` directory.
- [ ] Add:
  - [x] [src/core/hlir/egraph/mod.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/mod.rs)
  - [x] [src/core/hlir/egraph/region.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/region.rs)
  - [x] [src/core/hlir/egraph/encode.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/encode.rs)
  - [x] [src/core/hlir/egraph/facts.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/facts.rs)
  - [x] [src/core/hlir/egraph/schema.egg](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/schema.egg)
  - [x] [src/core/hlir/egraph/extract.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/extract.rs)
  - [x] [src/core/hlir/egraph/decode.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/decode.rs)
  - [x] [src/core/hlir/egraph/cost.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/cost.rs)
  - [x] [src/core/hlir/egraph/tests.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/tests.rs)
- [x] Add a temporary compatibility shim so callers can move from `egglog_algebraic` to `canonicalize_algebraic` incrementally.
- [x] Update module wiring in [src/core/hlir/mod.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/mod.rs).

### Completion Criteria

- [x] The new module tree builds with stubbed internals.
- [x] All existing imports resolve without forcing a full refactor in one commit.

## 3. Region Planning And Symbolic Barriers

### Region Planner

- [x] Implement `RegionPlan` in [src/core/hlir/egraph/region.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/region.rs).
- [x] Implement algebraic-op classification in one place in [src/core/hlir/egraph/region.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/region.rs).
- [x] Traverse roots and partition algebraic regions from barrier nodes.
- [x] Record barrier nodes in topological order for later rebuild.
- [ ] Record symbolic leaf refs for:
  - [x] loads
  - [x] barrier outputs

### Constraints

- [x] Do not materialize any output graph nodes during planning.
- [x] Do not encode barriers as prebuilt opaque subtrees.

### Tests

- [x] Region planner correctly isolates a `Reduce` between two algebraic regions.
- [x] Shared barrier outputs are recorded once and reused.
- [x] Topological order of barrier rebuild inputs is deterministic.

## 4. Exact Scalar Infrastructure

### Scalar Arena

- [x] Add `ScalarArena` and `ScalarId` in a new Rust module, either [src/core/hlir/egraph/facts.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/facts.rs) or a dedicated helper module if it grows.
- [x] Intern scalar constants exactly, keyed by dtype and exact bit pattern.
- [x] Add exact `neg`, `add`, and `mul` memoized closures over scalar ids.
- [x] Add exact zero and one predicates per dtype.

### Fix Current Scalar Weaknesses

- [x] Stop using `Scalar::to_f64()` as the canonicalization path for scalar arithmetic.
- [x] Stop using `Scalar::from_f64()` for exact half/bfloat canonicalization unless it is fixed first in [src/core/hlir/types.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/types.rs).
- [x] If needed, add exact `F16` and `BF16` conversion helpers to [src/core/hlir/types.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/types.rs).

### Tests

- [x] `I64` scalar folding remains exact.
- [x] `F16` scalar folding preserves exact bit patterns.
- [x] `BF16` scalar folding preserves exact bit patterns.

## 5. Shape, Axes, And Layout Fact Emission

### Shape Interning

- [x] Add canonical shape interning in [src/core/hlir/egraph/facts.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/facts.rs).
- [x] Add canonical axes interning in [src/core/hlir/egraph/facts.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/facts.rs).
- [x] Define how symbolic `Dim` expressions are serialized into egglog-compatible shape facts.

### Required Facts

- [x] Emit `shape-of(expr)` facts.
- [x] Emit `dtype-of(expr)` facts.
- [x] Emit `layout-of(expr)` facts.
- [x] Emit `rank-of(shape)` facts.
- [x] Emit `same-shape` facts.
- [x] Emit `same-numel` and `reshape-ok` facts.
- [x] Emit `broadcast-ok` facts.
- [x] Emit `expand-ok` facts.
- [x] Emit `permute-ok` facts.
- [x] Emit `axes-compose` facts.
- [x] Emit `axes-identity` facts.
- [x] Emit binary shape legality facts for `Add`, `Mul`, `Max`, and `Min`.

### Completion Criteria

- [ ] Egglog can distinguish legality from mere shape-id equality.
- [ ] The illegal reshape-sinking counterexample is blocked by missing facts, not by a Rust-side special case.

## 6. Typed Egglog Schema

### Schema Setup

- [x] Implement the new typed schema in [src/core/hlir/egraph/schema.egg](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/schema.egg).
- [x] Add datatypes for:
  - [x] `Expr`
  - [x] `Shape`
  - [x] `Axes`
  - [x] `DType`
  - [x] `Layout`
  - [x] `ScalarExpr`

### Functions And Relations

- [x] Add all required semantic functions from [SPEC.md](file:///home/sword/Desktop/code/venum/SPEC.md).
- [x] Add all required semantic relations from [SPEC.md](file:///home/sword/Desktop/code/venum/SPEC.md).
- [x] Ensure the schema has a place for symbolic leaf refs that are source-node based, not output-node based.

### Validation

- [x] Add a small schema-only smoke test that parses and runs the egglog program without using the full HLIR pipeline.

## 7. Structural Encoder

### Direct Encoding

- [x] Implement structural encoding in [src/core/hlir/egraph/encode.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/encode.rs).
- [x] Encode every algebraic op directly as its typed `Expr` constructor.
- [x] Encode `Broadcast` directly.
- [x] Encode scalar constants as rank-0 `Const` terms.
- [x] Encode non-scalar constants as tensor `Const` terms.
- [x] Encode barrier outputs as `Leaf` refs.

### Explicit Non-Behavior

- [x] Do not eliminate identity reshape/permute/expand in Rust.
- [x] Do not convert multiply-by-constant into a special expr node in Rust.
- [x] Do not emit HLIR output nodes during encode.

### Tests

- [x] Encoding a graph with identity shape ops still emits those terms structurally.
- [x] Scalar-plus-tensor input graphs encode to scalar const plus tensor expr, not a pre-broadcasted output graph.

## 8. Core Rewrites And Canonical Form

### Identity And Basic Algebra

- [x] Add typed add-zero and mul-one rules.
- [x] Add typed mul-zero rules.
- [x] Add double-negation and double-recip rules.
- [x] Gate non-IEEE identities like `Exp(Log(x)) -> x` behind an explicit `fast_math` flag.

### View Rules

- [x] Add reshape-collapse rules guarded by `reshape-ok`.
- [x] Add permute composition rules guarded by `axes-compose`.
- [x] Add expand-collapse rules guarded by `expand-ok`.
- [x] Add broadcast-collapse rules guarded by `broadcast-ok`.

### Pointwise Motion

- [x] Add unary-through-view rules guarded by legality facts.
- [x] Add binary-through-view rules guarded by legality facts.
- [x] Verify that no binary shape-motion rule depends only on identical output shape ids.

### Canonical Ordering

- [x] Define stable operand ordering in [src/core/hlir/egraph/schema.egg](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/schema.egg) or [src/core/hlir/egraph/cost.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/cost.rs), depending on implementation strategy.
- [x] Flatten nested `Add` internally.
- [x] Flatten nested `Mul` internally.
- [x] Sort flattened operands deterministically.

### Tests

- [x] `x + 0` and `0 + x` canonicalize identically.
- [x] `x * 1` and `1 * x` canonicalize identically.
- [x] Operand reordering in `Add(Add(a, b), c)` and `Add(c, Add(b, a))` yields the same extracted form.

## 9. Scalar Broadcast Semantics

### Canonicalization Rules

- [x] Add scalar-plus-tensor rewrite rules that introduce `Broadcast` explicitly.
- [x] Add scalar-times-tensor rewrite rules that introduce `Broadcast` explicitly.
- [x] Add canonical cast-before-broadcast rules for dtype promotion.
- [x] Add zero/one canonical tensor identities in terms of `Broadcast(Const(..., []), shape)`.

### Tests

- [x] `Add(x, Const(2, []))` canonicalizes to `Add(x, Broadcast(Const(2, []), shape(x)))`.
- [x] `Mul(x, Const(2, []))` canonicalizes to `Mul(Broadcast(Const(2, []), shape(x)), x)` or the chosen canonical order.
- [x] Mixed-dtype scalar-plus-tensor ops cast the scalar before broadcast.

## 10. Linear Combination Analysis

### Generic Like-Term Collection

- [x] Implement `base-of(expr)` logic for recognized linear tensor terms.
- [x] Implement `coeff-of(expr)` logic for recognized linear tensor terms.
- [x] Make coefficient arithmetic use the exact scalar arena.
- [x] Group `Add` operands by canonical `base-of`.
- [x] Fold exact scalar coefficients for identical bases.
- [x] Re-emit grouped terms without introducing `EScale` or `EBias` expr nodes.

### Required Coverage

- [x] `x + x -> 2 * x`
- [x] `x * 2 + x -> 3 * x`
- [x] `(x * 2 + x) + (x * 2 + x) -> 6 * x`
- [x] `x + 1 + 2 + 3 -> x + 6`

### Guardrails

- [ ] Only collect terms when scalar coefficient semantics are exact and legal.
- [ ] Do not collect unlike bases.
- [ ] Do not fold through non-linear ops.

## 11. Extraction And Cost Model

### Extraction Plumbing

- [x] Implement one shared e-graph session per canonicalization run in [src/core/hlir/egraph/extract.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/extract.rs).
- [x] Saturate to fixpoint or an explicit bounded strategy that reports when the bound is hit.
- [x] Extract all region roots together.
- [x] Share decode memo across all extracted roots.

### Cost Model

- [x] Implement custom costs in [src/core/hlir/egraph/cost.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/cost.rs).
- [x] Price `Broadcast`, `Expand`, and `Permute` as cheap view-ish ops.
- [x] Penalize materialized non-scalar constants that could stay scalar-plus-broadcast.
- [x] Penalize explicit dtype conversion chains.
- [ ] Penalize awkward layouts if the resulting expression is otherwise equivalent.

### Tests

- [x] Extraction prefers `Broadcast(Const([], ...), shape)` over materialized full-shape tensor constants when equivalent.
- [ ] Extraction does not choose a syntactically cheap form that decodes to a larger HLIR than an available alternative.

## 12. Structural Decoder And Barrier Rebuild

### Decoder

- [x] Implement structural decode in [src/core/hlir/egraph/decode.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/decode.rs).
- [x] Decode `Broadcast` to `Op::Broadcast`, not to `Reshape + Expand`.
- [x] Rebuild flattened `Add` and `Mul` into balanced binary trees.
- [x] Preserve sharing with a single decode memo across all roots.

### Barrier Rebuild

- [x] Rebuild barriers using symbolic leaf remapping, not materialized opaque subtrees.
- [x] Ensure barrier inputs resolve from:
  - [x] decoded algebraic outputs
  - [x] already rebuilt barriers
  - [x] preserved source leaves

### Explicit Non-Behavior

- [x] Do not simplify during decode.
- [x] Do not invent shape-op sequences that the e-graph never saw.

### Tests

- [x] `Reduce(Add(x, 0))` rebuilds to `Reduce(x)` through the new symbolic-barrier path.
- [x] An algebraic parent above a barrier sees the remapped barrier output, not a stale opaque subtree.

## 13. Migration And Cleanup

- [x] Switch all internal call sites from the old [src/core/hlir/egraph.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph.rs) entry point to the new [src/core/hlir/egraph/mod.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph/mod.rs) API.
- [x] Remove the old [src/core/hlir/algebra.egg](file:///home/sword/Desktop/code/venum/src/core/hlir/algebra.egg) once parity and correctness are verified.
- [x] Remove the old [src/core/hlir/egraph.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph.rs) once no callers remain.
- [x] Update any docs or comments that refer to the old affine-node design.

## 14. Final Verification

- [x] Run the full HLIR test suite.
- [x] Run the new egraph-specific test suite.
- [x] Add at least one randomized equivalence test over small tensor shapes for supported algebraic ops.
- [x] Manually inspect extracted graphs for:
  - [x] scalar-plus-tensor add
  - [x] scalar-plus-tensor mul
  - [x] `6 * x` normalization
  - [x] legal reshape sinking
  - [x] blocked illegal reshape sinking
- [x] Confirm no Rust-side encode/decode shortcut remains for:
  - [x] identity reshape
  - [x] identity permute
  - [x] identity expand
  - [x] scalar multiply/add special cases

## Done Definition

- [x] The new canonicalizer is shape-aware from the start.
- [x] The e-graph owns legality and canonicalization.
- [x] Scalar-tensor arithmetic is explicit in HLIR via `Broadcast`.
- [x] Like-term collection works without `EScale` and `EBias`.
- [x] Barrier reconstruction uses symbolic leaves instead of opaque materialization.
- [x] Extraction cost matches the emitted HLIR closely enough that there is no known systematic mismatch.
