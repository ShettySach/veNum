# Native Polyhedral Examples (Current State)

This document shows concrete examples of what the native polyhedral layer can
already do today.

The examples are grounded in current tests under `src/core/poly/tests.rs`.

## 1) Affine If Guard Refinement

What it demonstrates:

- extraction can parse affine `if` predicates
- `then` and `else` domains are refined with different constraints

Representative pattern:

```text
for i0 in [0, 16):
  if 0 < 8:
    S_then(i0)
  else:
    S_else(i0)
```

Even with this simple predicate, extraction now builds distinct domain facts:

- then-side includes `7 >= 0` (from strict `<` lowering)
- else-side includes `-8 >= 0` (negated guard path)

Reference test:

- `extract_if_affine_guard_refines_then_and_else_domains`

## 2) Vector Gather/Scatter Memory Effects

What it demonstrates:

- gather contributes a read access to its base buffer
- scatter contributes a write access to its base buffer

Representative pattern:

```text
S0: dst[i0] = gather(base=buf7, indices=...)
S1: dst[i0] = scatter(base=buf9, indices=..., value=...)
```

Extraction now records these as real accesses in statement instances:

- gather: `reads` includes buffer `7`
- scatter: `writes` includes buffer `9`

Reference tests:

- `extract_vector_gather_records_base_buffer_read`
- `extract_vector_scatter_records_base_buffer_write`

## 3) Access Dependence Is No Longer Placeholder-Empty

What it demonstrates:

- `NativeDependenceAnalyzer::access_dependence` now builds a constrained
  relation from domains, memory equalities, and lexicographic order slices
- impossible dependence cases return `None`

Representative pattern (feasible):

```text
write: A[i0]
read:  A[i0]
```

Result:

- returns `Some(DependenceRelation)`
- relation contains non-empty `Eq` and `Ge` constraints

Representative pattern (infeasible under source-before-sink):

```text
write: A[i0]
read:  A[i0 + 1]
```

Result:

- returns `None`

Reference tests:

- `access_dependence_emits_real_constraints_for_same_cell`
- `access_dependence_returns_none_for_reverse_time_shift`

## 4) Proof-Oriented Feasibility

What it demonstrates:

- feasibility checker is conservative and proof-oriented
- when exact elimination is unavailable, it returns `Unknown` instead of
  claiming `Feasible`

Representative pattern:

```text
2*x + y = 0
x >= 0
y >= 1
```

Result:

- feasibility is `Unknown` (not optimistic `Feasible`)

Reference test:

- `feasibility_non_unit_equality_with_bounds_is_unknown`

## 5) Non-Affine Shape Rejection for Poly Domain Build

What it demonstrates:

- non-affine dims are rejected in checked path rather than collapsed into fake
  affine singletons

Representative pattern:

```text
shape: [Div(8, 2)]
```

Result:

- `shape_to_domain_checked(...)` returns an error

Reference test:

- `shape_to_domain_rejects_non_affine_dim`

## Quick Commands

Run focused examples:

```bash
cargo test -q extract_if_affine_guard_refines_then_and_else_domains -- --nocapture
cargo test -q extract_vector_gather_records_base_buffer_read -- --nocapture
cargo test -q access_dependence_emits_real_constraints_for_same_cell -- --nocapture
cargo test -q feasibility_non_unit_equality_with_bounds_is_unknown -- --nocapture
```

Run all poly tests:

```bash
cargo test -q core::poly::tests -- --nocapture
```

## Current Boundaries

These examples are intentionally modest and represent current capability, not
the final mature target:

- feasibility still returns `Unknown` for many symbolic/non-unit systems
- guard parsing supports affine comparisons but not full boolean normalization
- distance/direction summaries remain conservative
