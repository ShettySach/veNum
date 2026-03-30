> Dim::Mul(Box<Dim>, i64) is too restrictive. The Dim type supports multiplying a dimension by a scalar constant, but not symbol × symbol. Stride computations after Reshape require this — if you reshape [N, M] into [N*M]

I understand, but would this clash with Presburger arithmetic and affinity? Can this cause me to miss some fusions and optimizations in the higher layer? Any solutions?

> More importantly, there is no Argmax or Argmin

Argmax, Argmin are not associative. Can I do reductions on non-associative functions like these?

Keep in mind I need to balance higher level fusion and polyhedrality.
With these lenses, once again state any valid improvements.

--- 

### On `Dim::Mul` and Presburger Arithmetic

The concern is real but the framing needs sharpening. `Dim::Mul(Box<Dim>, i64)` — a symbolic dimension scaled by a *constant* — is fine. Multiplying a variable by a constant is affine by definition. The actual problem is more specific: after `Reshape([N, M] -> [N*M])`, the loop bound of the flattened domain becomes `0 <= i < N*M`. That is a product of two *parameters*, which is non-linear. Standard Presburger arithmetic — and ISL's affine map representation — requires that parameter terms appear linearly. `N*M` as a bound is outside that.

The practical consequence for ISL: it either rejects the constraint outright or falls back to a conservative over-approximation of the iteration domain. Over-approximated domains produce over-approximated dependence relations, which cascade directly into missed fusion — the scheduler sees phantom dependencies between kernels that are actually independent, and refuses to fuse them.

The strides themselves are fine. After reshape, the stride of a dimension with size `M` (a parameter) is representable as `Aff { constant: 0, terms: [(1, Var::Param("M"))] }` — that is linear in the parameters. It is only the *bounds* that become problematic.

**Solutions, in order of how well they preserve fusion power:**

The cleanest is **parameter interning before lowering**: before `lower()` runs, introduce a fresh named parameter for any product of symbolic dims that appears as a loop bound — `P_NM = N * M` — and substitute throughout. The domain constraint becomes `0 <= i < P_NM`, which is linear. ISL sees a normal Presburger constraint. Dependence analysis is exact. Fusion decisions are not degraded. The mapping back to the original symbolic dims is preserved in a side table.

The second option is to use **ISL's quasi-affine (`isl_pw_aff`) extensions**, which handle piecewise affine expressions and some non-linear cases by case-splitting. This preserves expressiveness but makes dependence analysis approximate in ways that are hard to predict — you get partial fusion rather than no fusion, but you cannot reason about exactly which fusions will be blocked.

The third option is to **restrict `Reshape` to cases where the resulting bounds remain affine** — i.e., only allow reshape when involved dimensions are constants, or when the reshape is a pure no-op at the affine level (e.g., inserting or removing a size-1 dimension). All other reshapes are materialized as copy kernels with concrete strides. This preserves strict affinity at the cost of more copies, which is acceptable for a CPU-first compiler in early stages.

For now, option one — parameter interning — is the right call. It keeps the `Dim` type as-is, avoids the quasi-affine complexity, and produces clean ISL domains.

---

### On Argmax / Argmin and Non-Associativity

The associativity framing is slightly off. Argmax *is* associative — just not over scalars. Lift the reduction element to a `(value, index)` pair, and the combining operation is:

```
combine(a, b) = if a.value >= b.value then a else b
```

This is associative and commutative over pairs. The problem is not associativity — it is that your current type system has no tuple type, and your `ReduceOp` operates on scalar elements only.

The deeper issue for polyhedrality is the conditional update. In loop form:

```
best_val = -inf; best_idx = -1
for i in 0..N:
    if x[i] > best_val: best_val = x[i]; best_idx = i
```

The update to `best_idx` is data-dependent. That is not an affine recurrence. ISL cannot tile this loop freely — splitting it across iterations requires a merge step with its own conditional, which breaks the standard polyhedral reduction form. Concretely: you can tile a `Sum` reduction and parallelize the partial sums because partial sums combine linearly. You cannot do the same for `Argmax` without a two-phase approach.

The two-phase decomposition is the correct solution:

1. `max_val = Reduce(Max, x)` — affine, tileable, fusable with adjacent elementwise ops via standard polyhedral analysis
2. `argmax = find first index where x[i] == max_val` — a linear scan, not polyhedral, emitted as a separate non-affine kernel

Phase 1 participates fully in the polyhedral layer. Phase 2 is a special-case kernel outside it. This is what most production compilers do — MLIR's linalg dialect handles this by treating `argmax` as a named op that lowering does not attempt to decompose into the affine layer.

For LLM inference specifically: argmax over vocabulary at decode time is a single call over a large flat buffer. It is not in a hot tiled loop. Library BLAS/SIMD routines handle it well without polyhedral analysis, and the cost of keeping it out of the polyhedral layer is negligible.

---

### Valid Improvements Through These Two Lenses

**1. Normalize non-linear bounds before lowering.**  
Add a `NormalizeDims` pass between HLIR and the Plan IR that interns products of symbolic parameters as fresh named params. This keeps ISL domains strictly affine and preserves full dependence precision through lowering. Without it, any program with a `Reshape` over symbolic shapes degrades to conservative fusion.

**2. Separate affine and non-affine reductions at the type level.**  
Split `ReduceOp` into `AffineReduce { Sum, Max, Min, Prod }` and `IndexedReduce { Argmax, Argmin }`. The scheduler routes them differently: `AffineReduce` enters the polyhedral layer, participates in tiling and fusion analysis, and has its init/update lowering delayed until after fusion planning. `IndexedReduce` is lowered immediately to a two-phase kernel — an `AffineReduce(Max)` followed by a non-affine index scan — and the Max phase is re-exposed to the polyhedral layer as a normal reduction. This preserves as much fusion surface as possible while being honest about what the polyhedral layer can and cannot do.

**3. Delay init/update splitting for `AffineReduce` until after region planning.**  
The current spec splits `Reduce` into init + update kernels inside `lower_node`, which runs per-node, before the scheduler sees the graph. This is the specific mechanism by which Softmax and LayerNorm lose their fusion surface — the three passes of a Softmax all contain reductions, and once those reductions are split, the producer-consumer chain between them looks like six kernels (three inits, three updates) rather than three semantic stages. The fix: `lower_node` for `AffineReduce` emits a single unsplit reduction kernel tagged with `SemanticTag::Reduction`. The init/update split is applied by the scheduler *after* fusion regions are decided, only to the kernels that remain unfused. This is the highest-leverage single change for semantic fusion quality.

**4. Carry broadcast strides explicitly.**  
`Expand` should set stride-0 on broadcast dimensions at HLIR type-check time, and `strides_to_access` should emit those as affine terms with zero coefficient — i.e., no loop variable appears in the address computation for that dimension. This is strictly necessary for the access maps to be correct, and also allows ISL to detect that two kernels over broadcast dimensions have no data dependence on those axes — which enables tiling along them freely.

**5. Enforce that `lower()` processes nodes in topological order.**  
This is a correctness requirement, not an optimization. The HLIR graph construction should maintain topological invariants, or `lower()` should sort before iterating. An incorrect kernel order produces an incorrect dependence graph — ISL will compute dependences on a schedule that is already wrong.
