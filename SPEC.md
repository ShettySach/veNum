# Shape-Aware Egglog Canonicalization Spec

## Goal

Replace the current hybrid optimizer in [src/core/hlir/egraph.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph.rs) with a shape-aware, type-aware, egglog-first canonicalization pipeline.

The new design makes egglog responsible for:

- equivalence
- legality
- canonical form
- scalar arithmetic
- affine term collection
- extraction over legal alternatives

Rust is responsible for:

- partitioning HLIR into algebraic regions and barriers
- structurally encoding terms and facts into egglog
- extracting chosen terms
- structurally decoding them back to HLIR

Rust must not perform semantic canonicalization during encode or decode.

## Lessons From The Current Design

The current approach in [src/core/hlir/egraph.rs](file:///home/sword/Desktop/code/venum/src/core/hlir/egraph.rs) and [src/core/hlir/algebra.egg](file:///home/sword/Desktop/code/venum/src/core/hlir/algebra.egg) has four structural problems.

1. Legality is not represented inside egglog.
2. Shape, dtype, and layout are passed as opaque ids instead of semantic facts.
3. Canonicalization is split between egglog and Rust encode/decode.
4. Extraction cost does not match the HLIR that decode actually emits.

The replacement design fixes those directly.

## Non-Goals

- scheduling
- fusion planning
- lowering to kernels
- backend-specific heuristics
- proving arbitrary nonlinear tensor identities

This spec covers canonicalization of algebraic HLIR regions only.

## Scope

Canonicalize the algebraic subset of HLIR:

- `Const`
- `Load`
- `Neg`, `Recip`, `Exp`, `Log`, `Sqrt`, `Sin`
- `Cast`
- `Add`, `Mul`, `Max`, `Min`
- `Reshape`, `Permute`, `Expand`

Treat all other ops as barriers:

- `Reduce`
- `Slice`
- `Concat`
- `Cmp`
- `Where`
- `Store`
- any future non-elementwise or effectful op

Barrier ops are not encoded as opaque materialized subgraphs. They are encoded as symbolic leaf references and rebuilt after extraction.

## Required Refactor

Replace the current single-file implementation with this module layout.

```text
src/core/hlir/egraph/
  mod.rs
  region.rs
  encode.rs
  facts.rs
  schema.egg
  extract.rs
  decode.rs
  cost.rs
  tests.rs
```

### `mod.rs`

Exports:

```rust
pub fn canonicalize_algebraic(
    graph: &HLIRGraph,
    roots: &[NodeId],
) -> (HLIRGraph, HashMap<NodeId, NodeId>);
```

This replaces `egglog_algebraic`.

### `region.rs`

Responsibilities:

- walk `roots`
- partition the graph into algebraic regions separated by barriers
- record barrier nodes in topological order
- record symbolic leaf references for barrier outputs and loads
- record region roots to extract together

Data structures:

```rust
pub struct RegionPlan {
    pub region_roots: Vec<NodeId>,
    pub barriers: Vec<BarrierNode>,
    pub symbolic_leaves: Vec<LeafRef>,
}

pub enum LeafRef {
    Load(NodeId),
    BarrierOutput(NodeId),
}

pub struct BarrierNode {
    pub src_id: NodeId,
    pub inputs: Vec<NodeId>,
}
```

Constraint:

- no subtree materialization during planning
- no output graph construction during planning

### `encode.rs`

Responsibilities:

- encode algebraic source nodes into egglog `Expr`
- encode all source constants into typed scalar and tensor facts
- intern each source node exactly once
- emit no identity elimination, no peepholes, no lowering tricks

Constraint:

- `Reshape`, `Permute`, `Expand`, `Cast`, scalar constants, and broadcast forms are encoded structurally even if they are identities

### `facts.rs`

Responsibilities:

- emit semantic facts for shape, dtype, rank, layout, scalar-ness, broadcast compatibility, reshape legality, and permutation composition
- compute and intern canonical descriptors for shapes and axes

### `extract.rs`

Responsibilities:

- evaluate roots in one e-graph session
- saturate to fixpoint or configured node budget
- extract all region roots with a shared cost model
- preserve sharing across roots during decode

### `decode.rs`

Responsibilities:

- reconstruct HLIR from extracted terms
- rebuild barriers using remapped algebraic inputs
- decode structurally only

Constraint:

- no simplification during decode
- no scalar-broadcast lowering tricks hidden from egglog

### `cost.rs`

Responsibilities:

- define extraction costs aligned to final HLIR cost
- include penalties for materializing ops, casts, broadcasts, and non-contiguous layouts

## One HLIR Change

Add an explicit broadcast op.

```rust
Op::Broadcast { input: NodeId, shape: Vec<Dim> }
```

Reason:

- scalar-to-tensor operations are first-class semantics, not a decode hack
- `Const([])` broadcasting should be visible to egglog and to extraction cost
- current `reshape([] -> [1,1,...]) + expand` is an implementation artifact, not a canonical IR primitive

`Broadcast` semantics:

- allows rank expansion by inserting leading singleton dimensions
- allows singleton expansion in any dimension
- preserves dtype
- produces strided output with zero strides where expansion occurs

`Expand` remains as the shape op for same-rank expansion of already rank-aligned tensors. `Broadcast` is the general broadcast op. If desired, `Expand` may later be removed and subsumed by `Broadcast`, but this spec does not require that.

## Egglog Schema

The schema is not just a datatype for expressions. It is a typed term language plus semantic facts.

### Sorts

```lisp
(datatype DType
  F32 F16 BF16 F64 I8 I16 I32 I64 U8 U16 U32 U64 Bool)

(datatype Layout
  Contiguous
  Broadcasted
  Strided)

(datatype Shape
  (ShapeCons i64 Shape)
  ShapeNil)

(datatype Axes
  (AxesCons i64 Axes)
  AxesNil)

(datatype ScalarExpr
  (SConst i64)
  (SAdd ScalarExpr ScalarExpr)
  (SMul ScalarExpr ScalarExpr)
  (SNeg ScalarExpr))

(datatype Expr
  (Leaf i64)
  (Const ScalarExpr Shape DType)
  (Neg Expr)
  (Recip Expr)
  (Exp Expr)
  (Log Expr)
  (Sqrt Expr)
  (Sin Expr)
  (Cast Expr DType)
  (Add Expr Expr)
  (Mul Expr Expr)
  (Max Expr Expr)
  (Min Expr Expr)
  (Reshape Expr Shape)
  (Permute Expr Axes)
  (Expand Expr Shape)
  (Broadcast Expr Shape))
```

Notes:

- `ScalarExpr` carries ids into a Rust-side scalar arena for exact typed scalar values and exact scalar arithmetic.
- `Const` is a tensor constant, not just a scalar. A scalar constant is `Const(v, ShapeNil, dtype)`.
- `Leaf` payload is a symbolic source-node reference, not an output-node id.

### Semantic Functions And Relations

Required functions:

```lisp
(function shape-of (Expr) Shape)
(function dtype-of (Expr) DType)
(function layout-of (Expr) Layout)
(function scalar-of (Expr) ScalarExpr)
(function rank-of (Shape) i64)
(function numel-id-of (Shape) i64)
(function base-of (Expr) Expr)
(function coeff-of (Expr) ScalarExpr)
```

Required relations:

```lisp
(relation scalar-shape (Shape))
(relation same-shape (Shape Shape))
(relation same-numel (Shape Shape))
(relation reshape-ok (Shape Shape))
(relation broadcast-ok (Shape Shape))
(relation expand-ok (Shape Shape))
(relation permute-ok (Shape Axes Shape))
(relation axes-compose (Axes Axes Axes))
(relation axes-identity (Axes))
(relation unary-pointwise (Expr))
(relation binary-pointwise (Expr))
(relation scalar-expr (Expr))
(relation tensor-expr (Expr))
(relation legal-add-shapes (Shape Shape Shape))
(relation legal-mul-shapes (Shape Shape Shape))
```

Required scalar facts emitted from Rust:

- exact scalar dtype
- exact scalar value id
- scalar zero/one predicates
- exact add/mul/neg closure in the scalar arena

The e-graph must never recover scalar values via `to_f64` round-trips.

## Type And Shape Invariants

Every encoded `Expr` must have these facts available:

- `shape-of(expr)`
- `dtype-of(expr)`
- `layout-of(expr)`

Every rewrite that moves or re-associates shape ops must be guarded by these facts. No rewrite may rely on output-shape equality alone when source-shape legality matters.

Examples:

- `Add(Reshape(x, s), Reshape(y, s)) -> Reshape(Add(x, y), s)` is only legal if `shape-of(x)` and `shape-of(y)` are binary-compatible before reshape, not merely if both reshapes target `s`.
- `Mul(Broadcast(a, s), Broadcast(b, s)) -> Broadcast(Mul(a, b), s)` is only legal if the scalar or singleton broadcast semantics are provably equivalent.

## Canonical Form

The canonical form is defined by egglog, not by Rust.

### Structural Canonicalization

1. collapse identity view ops only via egglog rules
2. compose view ops where legality facts allow it
3. push view ops outward across pointwise ops where legality facts allow it
4. push casts to a consistent side where legality facts allow it
5. order commutative operands deterministically
6. flatten associative `Add` and `Mul` into internal n-ary forms for extraction

### Algebraic Canonicalization

1. exact constant folding
2. add-zero and mul-one elimination
3. mul-zero elimination where dtype and shape remain legal
4. collection of like terms in linear tensor expressions
5. grouping of scalar biases into one broadcasted constant when legal

## Linear Combination Analysis Without `EScale` Or `EBias`

Yes, this spec handles:

```text
(x * 2 + x) + (x * 2 + x) -> 6 * x
```

It does not do it by inventing ad hoc expression nodes like `EScale` or `EBias` inside the main `Expr` language.

Instead it uses a generic linear-combination analysis implemented with `base-of` and `coeff-of` plus canonical `Add` flattening.

### Required Behavior

For any expression recognized as a tensor-linear term:

- `base-of(expr)` returns the canonical tensor basis term
- `coeff-of(expr)` returns the exact scalar coefficient

Definitions:

- `base-of(x) = x`, `coeff-of(x) = 1`
- `base-of(Mul(Const(c, [], dt), x)) = base-of(x)` when `Const(c, [], dt)` is scalar
- `coeff-of(Mul(Const(c, [], dt), x)) = c * coeff-of(x)` when scalar
- `base-of(Broadcast(Const(c, [], dt), shape-of(x)))` is not a basis term for `x`; it is a pure constant tensor
- `Add` canonicalization groups terms with equal `base-of`

Example derivation:

```text
x * 2           -> basis x, coeff 2
x               -> basis x, coeff 1
x * 2 + x       -> basis x, coeff 3
(x * 2 + x) + (x * 2 + x)
                 -> basis x, coeff 3 + 3
                 -> basis x, coeff 6
                 -> Mul(Broadcast(Const(6, [], dtype(x)), shape-of(x)), x)
```

This is a generic scalar-ring normalization, not a custom `x * c` opcode.

### Implementation Requirement

The linear-combination analysis is partial.

It applies only when:

- scalar coefficients are rank-0 tensors or recognized scalar expressions
- the tensor basis term is identical after canonicalization
- dtype promotion rules for the scalar coefficient are exact and legal

If analysis does not apply, the term remains in normal `Add` and `Mul` form.

## Scalar Plus Tensor

Yes, this spec handles adding scalars to tensors.

### Canonical Semantics

A scalar is represented as a rank-0 tensor:

```text
Const(2, [], F32)
```

Adding it to a tensor is represented canonically as:

```text
Add(x, Broadcast(Const(2, [], dtype(x)), shape-of(x)))
```

not as a decode-time trick.

### Required Rules

1. if one `Add` operand is rank-0 and the other is tensor-shaped, rewrite to `Broadcast` on the scalar side
2. if one `Mul` operand is rank-0 and the other is tensor-shaped, rewrite to `Broadcast` on the scalar side
3. adjacent broadcasts collapse when target shape matches
4. `Broadcast(Const(0, [], dt), s)` is the canonical zero tensor of shape `s`
5. `Broadcast(Const(1, [], dt), s)` is the canonical multiplicative identity tensor of shape `s`

### Required DType Rule

When scalar and tensor dtypes differ, the scalar must be cast before broadcast according to the same promotion table used by HLIR binary ops.

Canonical order:

```text
Broadcast(Cast(Const(...), promoted_dtype), target_shape)
```

not:

```text
Cast(Broadcast(Const(...), target_shape), promoted_dtype)
```

unless the latter is cheaper and proven equivalent by a specific rewrite.

## Rewrites

The schema must include these rewrite families.

### Identity And Involution

```text
Add(x, zero(shape(x), dtype(x))) -> x
Mul(x, one(shape(x), dtype(x))) -> x
Mul(x, zero(shape(x), dtype(x))) -> zero(shape(x), dtype(x))
Neg(Neg(x)) -> x
Recip(Recip(x)) -> x
```

`Exp(Log(x)) -> x` and `Log(Exp(x)) -> x` are not enabled by default. They require a `fast_math` configuration gate.

### View Rules

```text
Reshape(Reshape(x, s1), s2) -> Reshape(x, s2)           if reshape-ok(shape(x), s1) and reshape-ok(s1, s2)
Permute(Permute(x, p1), p2) -> Permute(x, p3)          if axes-compose(p1, p2, p3)
Expand(Expand(x, s1), s2) -> Expand(x, s2)             if expand-ok(shape(x), s1) and expand-ok(s1, s2)
Broadcast(Broadcast(x, s1), s2) -> Broadcast(x, s2)    if broadcast-ok(shape(x), s1) and broadcast-ok(s1, s2)
```

### Pointwise Motion

```text
Unary(View(x)) -> View(Unary(x))
Binary(View(x), View(y)) -> View(Binary(x, y))
```

These rules must be generated only for combinations whose legality facts are present.

### Commutative And Associative Canonicalization

Implement internal n-ary normalization for `Add` and `Mul`.

Required behavior:

- flatten nested `Add`
- flatten nested `Mul`
- sort operands by a stable total order
- fold scalar constants inside each flattened bag
- rebuild as a balanced tree only during decode

The stable total order is:

1. tensor constants
2. scalar broadcasts
3. leaves
4. unary expressions
5. casts
6. view ops
7. composite pointwise expressions

Within each class, order by structural hash.

## Cost Model

Use a custom cost model, not raw constructor count.

Base costs:

- `Leaf`: 0
- `Const`: 0 for scalar, 1 for tensor constant
- unary pointwise op: 1
- binary pointwise op: 1
- `Cast`: 1
- `Reshape`: 0
- `Permute`: 0.25
- `Expand`: 0.25
- `Broadcast`: 0.25

Penalties:

- `+2` if resulting layout is non-contiguous and not broadcast-only
- `+1` for each explicit dtype conversion away from the root dtype
- `+2` for materializing a non-scalar constant tensor if it could have stayed scalar-plus-broadcast

Extraction is performed over all region roots in one session. Decode must memoize across all extracted roots so that shared subexpressions stay shared.

## Rust Scalar Arena

Introduce an exact scalar arena in Rust.

```rust
pub struct ScalarArena {
    values: Vec<ScalarValue>,
    add_cache: HashMap<(ScalarId, ScalarId), ScalarId>,
    mul_cache: HashMap<(ScalarId, ScalarId), ScalarId>,
    neg_cache: HashMap<ScalarId, ScalarId>,
}

pub struct ScalarValue {
    pub dtype: DType,
    pub bits: Scalar,
}
```

Requirements:

- exact arithmetic for integer types
- exact bit-pattern-preserving support for `F16` and `BF16`
- exact zero and one predicates per dtype
- no lossy `f64` intermediary during canonicalization

## Encode Rules

Encoding must be direct.

Examples:

- `Op::Const` -> `Const(scalar_id, shape_id, dtype)`
- `Op::Load` -> `Leaf(load_leaf_id)`
- `Op::Add(a, b)` -> `Add(expr(a), expr(b))`
- `Op::Broadcast` -> `Broadcast(expr(input), shape_id)`
- barrier output -> `Leaf(barrier_leaf_id)`

Disallowed encode behavior:

- replacing identity reshape with its input in Rust
- replacing identity permute with its input in Rust
- translating scalar-tensor multiply to a custom node in Rust
- materializing barrier subtrees into the output graph during encode

## Decode Rules

Decode must reconstruct only the extracted term.

Examples:

- `Broadcast(Const(c, [], dt), s)` decodes to `Op::Broadcast`, not `Reshape + Expand`
- flattened `Add` and `Mul` are rebuilt as balanced binary trees
- barrier leaves resolve through the region remap table

Disallowed decode behavior:

- eliminating identities not visible to egglog
- inventing view ops to implement scalar broadcast if `Op::Broadcast` exists
- folding scalar arithmetic in Rust after extraction

## Barrier Handling

Barriers are rebuilt after algebraic extraction.

Algorithm:

1. build `RegionPlan` with barrier nodes in topological order
2. encode algebraic regions with symbolic barrier-output leaves
3. saturate and extract all algebraic roots
4. decode all extracted roots into a new HLIR graph
5. rebuild barriers by remapping each input through either:
   - decoded algebraic output
   - already rebuilt barrier output
   - preserved leaf mapping

No barrier output may be materialized before extraction.

## Tests

The new test suite must include:

1. scalar plus tensor add:

```text
Add(Load([M,N]), Const(2, [])) -> Add(Load([M,N]), Broadcast(Const(2, []), [M,N]))
```

2. scalar plus tensor mul:

```text
Mul(Load([M,N]), Const(2, [])) -> Mul(Broadcast(Const(2, []), [M,N]), Load([M,N]))
```

3. like-term collection:

```text
(x * 2 + x) + (x * 2 + x) -> x * 6
```

4. broadcasted bias accumulation:

```text
x + 1 + 2 + 3 -> x + 6
```

5. legal reshape sinking with compatible pre-reshape shapes

6. illegal reshape sinking rejected when only target shape matches

7. permute composition

8. cast motion preserving dtype legality

9. barrier input simplification through `Reduce`

10. global extraction preserves DAG sharing across two roots

11. exact `F16` and `BF16` scalar folding

12. property tests comparing old and new graph evaluation on random small shapes for supported ops

## Implementation Sequence

Implement in this order.

1. add `Op::Broadcast` to HLIR and update shape/layout inference
2. create the new `src/core/hlir/egraph/` module tree
3. implement `region.rs` with symbolic barrier leaves
4. implement `ScalarArena` and exact scalar facts
5. implement `schema.egg` with typed `Expr`, `Shape`, `Axes`, and scalar facts
6. emit structural encode plus all legality facts
7. implement view, broadcast, and commutative canonicalization rules
8. implement linear-combination analysis via `base-of` and `coeff-of`
9. implement custom extraction cost model
10. implement structural decode and barrier rebuild
11. add the full test matrix
12. delete the old `src/core/hlir/egraph.rs` implementation once parity is reached

## Acceptance Criteria

The refactor is complete when all of the following are true.

1. `canonicalize_algebraic` performs no semantic simplification in Rust encode or decode.
2. All shape-sensitive rewrites are guarded by egglog facts.
3. Scalar-tensor arithmetic is represented explicitly through `Broadcast`.
4. `(x * 2 + x) + (x * 2 + x)` canonicalizes to `6 * x` without dedicated `EScale` or `EBias` expr nodes.
5. Extraction cost matches emitted HLIR structure.
6. Barrier reconstruction uses symbolic leaves and never pre-materializes opaque subgraphs.
7. `F16` and `BF16` scalar canonicalization is exact.
8. The canonical form is stable across operand ordering of commutative expressions.
