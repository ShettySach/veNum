## Review: Venum IR Design — Gaps and Recommended Fixes

---

### What the current design gets right

The two-IR split is sound. The HLIR correctly avoids bloating itself with domain ops — Matmul, Softmax, LayerNorm and friends stay out of the IR proper, which keeps the representation clean and the lowering path predictable. The LLIR preserves exactly what polyhedral analysis needs: explicit iteration domains, affine access maps, dependence structure, and enough metadata for legality checking, loop tiling, interchange, and elementwise or simple producer-consumer fusion. The decomposition strategy is coherent and the lowering direction is right.

The scheduler being greedy and dependency-driven is also not inherently wrong — it is the right tool for the optimization surface the LLIR exposes. The problem is not the scheduler; it is what the scheduler is given to work with.

---

### Core problem

Lowering eagerly, node-by-node, before scheduling destroys the semantic structure that makes deep fusion possible. Once Matmul decomposes into a `Reduce(Mul(Expand(...), Expand(...)))` subgraph and Softmax becomes `Exp(x - Max(x)) / Sum(...)`, those patterns are structurally gone from the IR. They are not hidden or encoded differently — they do not exist anymore as addressable units. A greedy dependency-driven scheduler sees a DAG of primitive operations with correct dependence edges. It cannot reliably reconstruct higher-level patterns from that, not because it is a bad scheduler, but because the information required to do so has been discarded by the time it runs.

The consequence is that you get good **local fusion** — elementwise chains, simple producer-consumer pairs, view elision where aliasing is obvious — but you do not get **semantic fusion**:

- **GEMM + bias + activation**: without knowing the reduce-multiply is a contraction, you cannot make the tiling and packing decisions that make this fast, nor can you reliably identify that the following add and elementwise are fusable into the output epilogue.
- **Softmax**: the numerically stable form is a three-pass computation (max reduction, exp+subtract, sum reduction, divide). Fusing those into one pass is the entire optimization. You cannot do it if each primitive is lowered independently.
- **LayerNorm**: same structure. Two reductions (mean, variance) over the same input followed by a normalization. Fusing them requires knowing they share an input and are part of a normalization chain.
- **Attention**: the entire FlashAttention family of optimizations depends on recognizing the QK^T + softmax + V pattern as a unit and tiling it in a way that keeps the intermediate softmax off DRAM entirely. This is architecturally impossible if Q, K, V matmuls and the softmax lower into independent kernel regions.
- **Conv + pool compute-at style placement**: fusion here depends on layout knowledge and on recognizing that the pooling window aligns with the convolution output in a way that allows producer values to be computed on demand. That reasoning requires both semantic identity and layout metadata to survive lowering.
- **Layout-sensitive optimizations generally**: when strides become abstract or are carried only implicitly, the compiler cannot make packing or transposition decisions. Those decisions silently default to whatever the lowering order happened to produce.

---

### Required fixes

**1. Semantic provenance tags**

Every lowered kernel region must carry a tag recording its origin: contraction, normalization chain, reduction chain, elementwise graph. This is the minimum. Without it, the LLIR is structurally isomorphic — every contraction looks like every other reduction, every normalization chain looks like a sequence of elementwise ops and reductions — and the scheduler has no basis for treating them differently.

Provenance tags are not just metadata for debugging. They are the mechanism by which HLIR-level decisions survive into the LLIR scheduling pass. If a region is tagged as a contraction, the scheduler knows to apply tiling strategies appropriate to GEMM-like access patterns, to look for fusable epilogues, and to consider library dispatch as a candidate. Without the tag, none of that is triggered.

**2. Pattern-recovery pass before lowering**

Add a rewrite pass at HLIR level that fires before lowering begins. It should recognize at minimum:

- `Reduce(Mul(Expand(...), Expand(...)))` → contraction / matmul-like
- `Exp(x - Max(x)) / Sum(...)` → softmax-like
- mean + variance chains → LayerNorm-like
- `Q @ K^T` followed by softmax followed by `@ V` → attention-like

The output of this pass is not a new op. It is a **region annotation** — a marker that says "this subgraph is a semantic unit" without changing the underlying primitive structure. Lowering can still decompose it into the same primitives it would have produced anyway; the difference is that it now does so as a unit, preserving the provenance tag and the region boundary, rather than independently for each node.

This pass does not require putting Matmul or Softmax back as first-class HLIR ops and it does not compromise the primitive-first philosophy. It is purely a recognition and annotation step.

**3. Region-based lowering**

Lower connected subgraphs together, not node-by-node. The current per-node eager lowering makes fusion decisions implicitly — by the time the scheduler runs, the question of whether two ops belong in the same kernel has already been answered by the lowering order, not by the scheduler. That is the wrong place for that decision.

Region-based lowering means: identify candidate fusion regions at HLIR level (using the pattern-recovery pass and standard producer-consumer analysis), lower each region as a unit into a single LLIR subgraph, and only then hand that subgraph to the polyhedral scheduler. This preserves the producer-consumer chains that matter for fusion and gives the scheduler a coherent region to tile and schedule rather than a pile of already-separated kernels.

**4. Richer view and layout metadata**

The following must survive lowering intact:

- **Aliasing relationships** between tensors — which views alias which backing allocations, and with what offset and stride relationship
- **Symbolic strides** — not just concrete values, but the symbolic expressions that let the compiler reason about whether two access patterns are compatible for fusion without materializing the full concrete layout
- **Broadcast semantics** — which dimensions are broadcast-expanded and from where, so that the scheduler can avoid materializing broadcast dimensions and instead handle them in the access map
- **Slice and permute provenance** — where a view came from, so that layout-sensitive transformations can be applied without introducing copies

Without this, layout-aware fusions silently degrade. The compiler cannot tell whether fusing two ops would require a copy or not, so it defaults to the conservative option. In practice this means most layout-sensitive fusions either do not fire or produce a kernel that copies the intermediate into a contiguous buffer, which often costs more than not fusing at all.

**5. Delay reduction lowering**

Reductions must remain as a single abstract operation through the entire fusion planning phase. Splitting a reduction into its init and update kernels before fusion decisions are made is the most common source of missed fusions in compilers of this style, because once split, the update kernel looks like an independent elementwise op and the structural relationship to the init is lost.

The rule should be: a reduction is lowered to init/update only after the fusion region it belongs to has been finalized and committed. Not before. This applies both to simple reductions and to the multi-pass reductions in Softmax and LayerNorm.

**6. Fusion recognizers, not hard-coded ops**

Do not reintroduce Matmul or Softmax as primary IR nodes. The goal is not to encode a fixed set of patterns the compiler knows about — that is exactly the approach the bitter lesson argues against. The goal is to have a mechanism by which arbitrary patterns can be recognized and scheduled as units.

The right structure is a set of **fusion recognizers**: pattern-matching rules that identify subgraphs matching known shapes and annotate them as candidate fusion regions. This is what tinygrad's `PatternMatcher` and `UPat` infrastructure does — it matches over the UOp graph and fires rewrite rules, but the rules are data, not hardcoded logic. The recognizer for matmul is a rule that can be added, modified, or replaced without touching the core scheduler. That is the right level of abstraction.

---

### Recommended architecture

The two-IR design is worth keeping, but it needs a third object between them:

```
HLIR → Plan IR → LLIR
```

The **Plan IR** is not a user-facing IR and does not need its own type system or syntax. It is an internal representation of the output of HLIR-level search: which ops are fused into which regions, which views remain symbolic versus materialized, which reductions are still inside a region versus split out, and which candidate specialization (loop nest vs library call vs custom kernel) was selected. It is a committed plan — a frozen set of decisions — that is then handed to the polyhedral lowering pass, which turns it into explicit iteration domains, access maps, and dependence edges.

This matters because search and lowering have different requirements. Search needs flexibility — it needs to be able to evaluate multiple candidate plans without committing to a representation. Polyhedral lowering needs commitment — it needs a fixed structure to analyze. Mixing them means either the search is constrained by what the LLIR can represent, or the lowering is complicated by having to handle partially-committed plans. The Plan IR separates those concerns cleanly.

Search is then stratified by layer:

**HLIR / Plan IR search:**
- Fusion boundary selection — which ops fuse into which regions
- Semantic rewrites and pattern recognition — contraction identification, normalization chain recognition
- Layout and materialization choices — which views get materialized, which remain symbolic
- Operator specialization candidates — does this contraction become a loop nest, a GEMM call, or a custom kernel

**LLIR search:**
- Tile sizes and tile shapes
- Loop order and interchange
- Unrolling factors and vectorization decisions
- Compute-at placement within already-chosen regions — where an operation's output is computed relative to its consumer's loop nest

This matches how tinygrad and Luminal actually work. Tinygrad is not purely primitive: its UOp set includes `CONTRACT`, `REDUCE`, `WMMA`, and `SHAPED_WMMA` alongside low-level primitives. Its `PatternMatcher` fires rewrite rules over the UOp graph before scheduling, and its scheduler uses BEAM search and heuristics over the grouped UOp graph — not over the raw primitives. Luminal exposes `matmul` at the graph API level and is explicitly building toward search-first kernel discovery, with FlashAttention as a motivating target. Neither framework stays purely primitive all the way down, and neither hardcodes a fixed backend implementation for any op.

---

### On Matmul specifically

The representation that preserves the bitter lesson while still enabling optimization is a **contraction region** — not a primitive op, not a library call alias, not a magic node the scheduler has special cases for:

```
Y[b,m,n] = sum_k X[b,m,k] * W[k,n]
```

This form carries enough information to:

- Identify the operation as a matrix multiplication or batched contraction
- Preserve full polyhedral legality for tiling along m, n, k and loop interchange
- Decide whether bias addition or an activation function can be fused into the output epilogue without an additional pass
- Decide whether to emit a tiled loop nest, a GEMM dispatch, or a custom microkernel — based on the cost model, not hardcoded rules
- Carry layout constraints (row-major vs column-major, packing requirements) as metadata on the region rather than as assumptions baked into the lowered loops

The compiler is still choosing the implementation via search. Matmul as a contraction region is not the implementation — it is the semantic anchor that makes the search tractable. Without it, the scheduler is searching over a space of primitive loop nests that happen to compute a matrix multiplication, which is both a larger search space and a worse one, because the structure that makes GEMM-specific tiling strategies correct is not visible.

---

### Cost model

Replace hardcoded hardware numbers with a `HardwareModel` struct carrying:

- Per-level cache sizes (L1, L2, L3)
- Per-level bandwidth (L1↔L2, L2↔L3, L3↔DRAM)
- Vector width and number of vector units
- Peak FLOP/s at each vector width

Tile cost should be evaluated per cache level: does the tile's working set fit in L1? If not, L2? The roofline should be computed per level — a tile that fits in L2 has a different arithmetic intensity ceiling than one that spills to DRAM, and treating them the same produces systematically wrong cost estimates for memory-bound kernels, which is most of what a CPU-first compiler will encounter. The current global roofline will tend to over-tile (producing tiles that are optimal for compute intensity but too large for any real cache level) or under-tile (avoiding tiles that would be profitable if the working set fit in L2).
## Review: Venum IR Design — Gaps and Recommended Fixes

---

### What the current design gets right

The two-IR split is sound. The HLIR correctly avoids bloating itself with domain ops — Matmul, Softmax, LayerNorm and friends stay out of the IR proper, which keeps the representation clean and the lowering path predictable. The LLIR preserves exactly what polyhedral analysis needs: explicit iteration domains, affine access maps, dependence structure, and enough metadata for legality checking, loop tiling, interchange, and elementwise or simple producer-consumer fusion. The decomposition strategy is coherent and the lowering direction is right.

The scheduler being greedy and dependency-driven is also not inherently wrong — it is the right tool for the optimization surface the LLIR exposes. The problem is not the scheduler; it is what the scheduler is given to work with.

---

### Core problem

Lowering eagerly, node-by-node, before scheduling destroys the semantic structure that makes deep fusion possible. Once Matmul decomposes into a `Reduce(Mul(Expand(...), Expand(...)))` subgraph and Softmax becomes `Exp(x - Max(x)) / Sum(...)`, those patterns are structurally gone from the IR. They are not hidden or encoded differently — they do not exist anymore as addressable units. A greedy dependency-driven scheduler sees a DAG of primitive operations with correct dependence edges. It cannot reliably reconstruct higher-level patterns from that, not because it is a bad scheduler, but because the information required to do so has been discarded by the time it runs.

The consequence is that you get good **local fusion** — elementwise chains, simple producer-consumer pairs, view elision where aliasing is obvious — but you do not get **semantic fusion**:

- **GEMM + bias + activation**: without knowing the reduce-multiply is a contraction, you cannot make the tiling and packing decisions that make this fast, nor can you reliably identify that the following add and elementwise are fusable into the output epilogue.
- **Softmax**: the numerically stable form is a three-pass computation (max reduction, exp+subtract, sum reduction, divide). Fusing those into one pass is the entire optimization. You cannot do it if each primitive is lowered independently.
- **LayerNorm**: same structure. Two reductions (mean, variance) over the same input followed by a normalization. Fusing them requires knowing they share an input and are part of a normalization chain.
- **Attention**: the entire FlashAttention family of optimizations depends on recognizing the QK^T + softmax + V pattern as a unit and tiling it in a way that keeps the intermediate softmax off DRAM entirely. This is architecturally impossible if Q, K, V matmuls and the softmax lower into independent kernel regions.
- **Conv + pool compute-at style placement**: fusion here depends on layout knowledge and on recognizing that the pooling window aligns with the convolution output in a way that allows producer values to be computed on demand. That reasoning requires both semantic identity and layout metadata to survive lowering.
- **Layout-sensitive optimizations generally**: when strides become abstract or are carried only implicitly, the compiler cannot make packing or transposition decisions. Those decisions silently default to whatever the lowering order happened to produce.

---

### Required fixes

**1. Semantic provenance tags**

Every lowered kernel region must carry a tag recording its origin: contraction, normalization chain, reduction chain, elementwise graph. This is the minimum. Without it, the LLIR is structurally isomorphic — every contraction looks like every other reduction, every normalization chain looks like a sequence of elementwise ops and reductions — and the scheduler has no basis for treating them differently.

Provenance tags are not just metadata for debugging. They are the mechanism by which HLIR-level decisions survive into the LLIR scheduling pass. If a region is tagged as a contraction, the scheduler knows to apply tiling strategies appropriate to GEMM-like access patterns, to look for fusable epilogues, and to consider library dispatch as a candidate. Without the tag, none of that is triggered.

**2. Pattern-recovery pass before lowering**

Add a rewrite pass at HLIR level that fires before lowering begins. It should recognize at minimum:

- `Reduce(Mul(Expand(...), Expand(...)))` → contraction / matmul-like
- `Exp(x - Max(x)) / Sum(...)` → softmax-like
- mean + variance chains → LayerNorm-like
- `Q @ K^T` followed by softmax followed by `@ V` → attention-like

The output of this pass is not a new op. It is a **region annotation** — a marker that says "this subgraph is a semantic unit" without changing the underlying primitive structure. Lowering can still decompose it into the same primitives it would have produced anyway; the difference is that it now does so as a unit, preserving the provenance tag and the region boundary, rather than independently for each node.

This pass does not require putting Matmul or Softmax back as first-class HLIR ops and it does not compromise the primitive-first philosophy. It is purely a recognition and annotation step.

**3. Region-based lowering**

Lower connected subgraphs together, not node-by-node. The current per-node eager lowering makes fusion decisions implicitly — by the time the scheduler runs, the question of whether two ops belong in the same kernel has already been answered by the lowering order, not by the scheduler. That is the wrong place for that decision.

Region-based lowering means: identify candidate fusion regions at HLIR level (using the pattern-recovery pass and standard producer-consumer analysis), lower each region as a unit into a single LLIR subgraph, and only then hand that subgraph to the polyhedral scheduler. This preserves the producer-consumer chains that matter for fusion and gives the scheduler a coherent region to tile and schedule rather than a pile of already-separated kernels.

**4. Richer view and layout metadata**

The following must survive lowering intact:

- **Aliasing relationships** between tensors — which views alias which backing allocations, and with what offset and stride relationship
- **Symbolic strides** — not just concrete values, but the symbolic expressions that let the compiler reason about whether two access patterns are compatible for fusion without materializing the full concrete layout
- **Broadcast semantics** — which dimensions are broadcast-expanded and from where, so that the scheduler can avoid materializing broadcast dimensions and instead handle them in the access map
- **Slice and permute provenance** — where a view came from, so that layout-sensitive transformations can be applied without introducing copies

Without this, layout-aware fusions silently degrade. The compiler cannot tell whether fusing two ops would require a copy or not, so it defaults to the conservative option. In practice this means most layout-sensitive fusions either do not fire or produce a kernel that copies the intermediate into a contiguous buffer, which often costs more than not fusing at all.

**5. Delay reduction lowering**

Reductions must remain as a single abstract operation through the entire fusion planning phase. Splitting a reduction into its init and update kernels before fusion decisions are made is the most common source of missed fusions in compilers of this style, because once split, the update kernel looks like an independent elementwise op and the structural relationship to the init is lost.

The rule should be: a reduction is lowered to init/update only after the fusion region it belongs to has been finalized and committed. Not before. This applies both to simple reductions and to the multi-pass reductions in Softmax and LayerNorm.

**6. Fusion recognizers, not hard-coded ops**

Do not reintroduce Matmul or Softmax as primary IR nodes. The goal is not to encode a fixed set of patterns the compiler knows about — that is exactly the approach the bitter lesson argues against. The goal is to have a mechanism by which arbitrary patterns can be recognized and scheduled as units.

The right structure is a set of **fusion recognizers**: pattern-matching rules that identify subgraphs matching known shapes and annotate them as candidate fusion regions. This is what tinygrad's `PatternMatcher` and `UPat` infrastructure does — it matches over the UOp graph and fires rewrite rules, but the rules are data, not hardcoded logic. The recognizer for matmul is a rule that can be added, modified, or replaced without touching the core scheduler. That is the right level of abstraction.

---

### Recommended architecture

The two-IR design is worth keeping, but it needs a third object between them:

```
HLIR → Plan IR → LLIR
```

The **Plan IR** is not a user-facing IR and does not need its own type system or syntax. It is an internal representation of the output of HLIR-level search: which ops are fused into which regions, which views remain symbolic versus materialized, which reductions are still inside a region versus split out, and which candidate specialization (loop nest vs library call vs custom kernel) was selected. It is a committed plan — a frozen set of decisions — that is then handed to the polyhedral lowering pass, which turns it into explicit iteration domains, access maps, and dependence edges.

This matters because search and lowering have different requirements. Search needs flexibility — it needs to be able to evaluate multiple candidate plans without committing to a representation. Polyhedral lowering needs commitment — it needs a fixed structure to analyze. Mixing them means either the search is constrained by what the LLIR can represent, or the lowering is complicated by having to handle partially-committed plans. The Plan IR separates those concerns cleanly.

Search is then stratified by layer:

**HLIR / Plan IR search:**
- Fusion boundary selection — which ops fuse into which regions
- Semantic rewrites and pattern recognition — contraction identification, normalization chain recognition
- Layout and materialization choices — which views get materialized, which remain symbolic
- Operator specialization candidates — does this contraction become a loop nest, a GEMM call, or a custom kernel

**LLIR search:**
- Tile sizes and tile shapes
- Loop order and interchange
- Unrolling factors and vectorization decisions
- Compute-at placement within already-chosen regions — where an operation's output is computed relative to its consumer's loop nest

This matches how tinygrad and Luminal actually work. Tinygrad is not purely primitive: its UOp set includes `CONTRACT`, `REDUCE`, `WMMA`, and `SHAPED_WMMA` alongside low-level primitives. Its `PatternMatcher` fires rewrite rules over the UOp graph before scheduling, and its scheduler uses BEAM search and heuristics over the grouped UOp graph — not over the raw primitives. Luminal exposes `matmul` at the graph API level and is explicitly building toward search-first kernel discovery, with FlashAttention as a motivating target. Neither framework stays purely primitive all the way down, and neither hardcodes a fixed backend implementation for any op.

---

### On Matmul specifically

The representation that preserves the bitter lesson while still enabling optimization is a **contraction region** — not a primitive op, not a library call alias, not a magic node the scheduler has special cases for:

```
Y[b,m,n] = sum_k X[b,m,k] * W[k,n]
```

This form carries enough information to:

- Identify the operation as a matrix multiplication or batched contraction
- Preserve full polyhedral legality for tiling along m, n, k and loop interchange
- Decide whether bias addition or an activation function can be fused into the output epilogue without an additional pass
- Decide whether to emit a tiled loop nest, a GEMM dispatch, or a custom microkernel — based on the cost model, not hardcoded rules
- Carry layout constraints (row-major vs column-major, packing requirements) as metadata on the region rather than as assumptions baked into the lowered loops

The compiler is still choosing the implementation via search. Matmul as a contraction region is not the implementation — it is the semantic anchor that makes the search tractable. Without it, the scheduler is searching over a space of primitive loop nests that happen to compute a matrix multiplication, which is both a larger search space and a worse one, because the structure that makes GEMM-specific tiling strategies correct is not visible.

---

### Cost model

Replace hardcoded hardware numbers with a `HardwareModel` struct carrying:

- Per-level cache sizes (L1, L2, L3)
- Per-level bandwidth (L1↔L2, L2↔L3, L3↔DRAM)
- Vector width and number of vector units
- Peak FLOP/s at each vector width

Tile cost should be evaluated per cache level: does the tile's working set fit in L1? If not, L2? The roofline should be computed per level — a tile that fits in L2 has a different arithmetic intensity ceiling than one that spills to DRAM, and treating them the same produces systematically wrong cost estimates for memory-bound kernels, which is most of what a CPU-first compiler will encounter. The current global roofline will tend to over-tile (producing tiles that are optimal for compute intensity but too large for any real cache level) or under-tile (avoiding tiles that would be profitable if the working set fit in L2).

---

### Summary

The spec is clean and the two-IR split is the right foundation. The gap is specific: node-by-node eager lowering makes fusion decisions before the scheduler runs, the greedy scheduler cannot recover the semantic structure that was discarded, and the result is a compiler that produces fast elementwise and simple producer-consumer fusion but cannot approach the semantic fusion quality of tinygrad or Luminal. The fixes are also specific: a thin Plan IR between HLIR and LLIR, a pre-lowering pattern-recovery and region annotation pass, delayed reduction lowering, richer layout metadata propagation, and a per-cache-level cost model. None of this requires abandoning the primitive-first philosophy or hardcoding op-specific backend logic — it just requires that the compiler make its major structural decisions before it discards the information those decisions depend on.
---

### Summary

The spec is clean and the two-IR split is the right foundation. The gap is specific: node-by-node eager lowering makes fusion decisions before the scheduler runs, the greedy scheduler cannot recover the semantic structure that was discarded, and the result is a compiler that produces fast elementwise and simple producer-consumer fusion but cannot approach the semantic fusion quality of tinygrad or Luminal. The fixes are also specific: a thin Plan IR between HLIR and LLIR, a pre-lowering pattern-recovery and region annotation pass, delayed reduction lowering, richer layout metadata propagation, and a per-cache-level cost model. None of this requires abandoning the primitive-first philosophy or hardcoding op-specific backend logic — it just requires that the compiler make its major structural decisions before it discards the information those decisions depend on.
