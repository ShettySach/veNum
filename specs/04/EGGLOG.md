# Egglog Growth Plan

You can grow Egglog meaningfully without letting it eat scheduling. The trick is to keep a hard contract:

Egglog owns semantic equivalence of compute graphs.
Search owns fusion, materialization, tiling, vectorization, parallelization, and memory placement.
Lowering owns turning a chosen compute graph plus chosen schedule into LLIR.

I would grow it in three stages.

## Stage 1: Keep Today’s Contract, Tighten It

This is basically what you already have in src/core/hlir/optimize.rs and src/core/hlir/egraph.rs.

Use Egglog for:

Algebraic simplification.
Shape/view motion that is schedule-neutral, like reshape sinking.
Decomposition normalization, so equivalent frontends end up in similar HLIR.

Do not let Egglog contain:

Tile sizes.
Loop order.
Thread/block structure.
Shared/local/global memory choices.
Backend-specific rewrites.

This alone is already a good design.

## Stage 2: Let Egglog Produce Compute Alternatives, Not Schedules

Right now you extract a single simplified graph. The next step is not “put scheduling into the e-graph.” It is:

Let Egglog represent multiple semantically equivalent compute graphs.
Extract a small top-K set of compute alternatives.
Hand each alternative to schedule search.

Conceptually:

optimize_hlir(hlir) -> Vec<ComputeAlternative>
search(alternative_i) -> ranked schedules
lower(best alternative + best schedule)

Examples of alternatives that still belong in Egglog:

Sub(a, b) vs Add(a, Neg(b)).
Different reassociations of elementwise/reduction-adjacent arithmetic.
Naive softmax expression vs stabilized softmax expression.
Direct-slice convolution form vs a more compact windowed equivalent, if you eventually choose to encode both.

These are still compute-level choices, not schedule-level choices.

This gets you a lot closer to the “Luminal flavor” without collapsing boundaries.

## Stage 3: Add Rewrite Classes, Not Rewrite Chaos

As the rule set grows, I would explicitly classify rules into buckets.

Canonicalization rules. These reduce representational noise. Example: redundant reshape removal, commutativity normalization.
Decomposition rules. These eliminate derived ops into primitives.
Algorithmic-equivalence rules. These change compute structure but preserve semantics. Example: stabilized softmax vs naive softmax.
Layout-neutral motion rules. These move Reshape, Expand, maybe Permute when semantics are obvious.

And I would explicitly ban two classes from Egglog:

Schedule-bearing rules. Anything that implies tiles, blocking, vector widths, thread grouping, or shared memory.
Hardware-conditioned rules. Anything whose desirability depends on GPU warp size, SIMD width, cache line size, or shared memory budget.

That line keeps the architecture understandable.

## What This Would Look Like In Practice

A clean future pipeline could be:

- Frontend builds HLIR with only semantic ops and derived-op decompositions.
- Egglog builds an equivalence class of compute-only graphs.
- Extract a tiny frontier of compute alternatives.
- For each compute alternative, run fusion and Opt search.
- Lower the winning (compute graph, schedule decision) pair to LLIR.
- Execute real LLIR/codegen, not HLIR interpretation.
