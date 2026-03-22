# Venum Optimization Plan

## Overview
This document outlines a comprehensive optimization strategy for the Venum lazy tensor engine, targeting transformer inference workloads (GPT-2 and larger models). The plan is organized into phases based on certainty of benefit and implementation complexity.

---

## Phase 1: Obvious Optimizations (High Confidence, High Impact)

These optimizations have clear benefits with minimal risk and should be implemented first.

### 1.1 Remove Unnecessary BufferPool Zeroing

**Current Issue**: The buffer pool zeros every reused buffer even though JIT-compiled kernels overwrite all elements.

**Implementation**:
- Remove the iterator that zeros buffer contents in `BufferPool::acquire`
- Keep the vector allocation path unchanged (already allocates zeroed memory)
- This is a one-line deletion

**Expected Impact**: 
- Eliminates O(numel) memset operation on every intermediate buffer reuse
- Particularly impactful for large attention intermediates in transformers

**Risk**: None. JIT kernels write every element before reads occur.

---

### 1.2 Implement LRU Cache with Size Limits

**Current Issue**: `KernelCache` and `PlanCache` grow unbounded in long-running inference servers.

**Implementation**:
- Create a new `lru_cache.rs` module implementing a generic LRU cache with configurable capacity
- The cache should maintain a HashMap for O(1) lookups and a linked list (or VecDeque) for LRU ordering
- On insertion when at capacity, evict the least recently used entry
- On access (get), move the accessed key to the most-recently-used position
- Replace raw HashMap types in `context.rs` with this LRU implementation
- Set default capacity: 1000 kernels, 500 execution plans (tunable via Context constructor)
- Add a `Context::with_cache_sizes(kernel_capacity, plan_capacity)` constructor for custom sizing

**Expected Impact**:
- Bounded memory usage regardless of workload diversity
- For typical GPT-2 inference: most kernels reused across batches, so high hit rate expected
- Prevents memory leaks in long-running services

**Risk**: Minimal. May need to tune cache sizes based on real workload profiling.

---

### 1.3 Precompute Kernel Metadata During Scheduling

**Current Issue**: Several properties are recomputed during JIT compilation even though they're deterministic from the schedule.

**Implementation**:
- Add two new fields to `FusedKernel` struct:
  - `has_noncontiguous_trackers: bool` - whether any input tracker requires multi-dimensional indexing
  - `input_index_map: HashMap<NodeId, usize>` - maps node IDs to their position in the input_buffers array
- Compute these fields during schedule building in `build_schedule`:
  - Check all trackers during the fusion process to set the boolean flag
  - Build the index map directly from the input_buffers vector
- Remove the runtime computation in `compile_kernel` that currently builds these on demand
- Update JIT compilation to read these precomputed values

**Expected Impact**:
- Eliminates repeated iteration over tracker lists during compilation
- Clearer separation of concerns: scheduling finalizes all fusion decisions

**Risk**: None. Purely moving computation earlier in the pipeline.

---

### 1.4 Remove Dead Code and Redundant Fields

**Current Issue**: Several fields and functions are marked with `allow(dead_code)` or are genuinely unused.

**Implementation**:
- Audit all `#[allow(dead_code)]` attributes:
  - Remove the attribute where the code IS actually used (false positives)
  - Delete the code where it's genuinely unused
- Once HashMap optimization is complete (Phase 3), remove these now-redundant fields from FusedKernel:
  - `num_tracked_inputs` (can use `input_trackers.len()`)
  - `num_absorbed_shape_ops` (can use `shape_source_map.len()`)
- Delete the `build_input_index_map` function (replaced by precomputed field)
- Inline the trivial `id_to_index` function (just returns `node_id.0`)

**Expected Impact**:
- Cleaner codebase, reduced maintenance burden
- Slight reduction in struct sizes

**Risk**: None. This is pure cleanup.

---

## Phase 2: Extended Egglog Optimization (High Impact, Moderate Risk)

These optimizations extend the egglog equality saturation engine to handle shape operations, enabling optimization across a much larger subset of typical computation graphs.

### 2.1 Add Shape Operation Support to Egglog

**Current Issue**: `is_optimize_safe` rejects any graph containing shape operations, preventing optimization of most real-world transformer graphs which heavily use reshape, permute, and transpose operations.

**Implementation**:

**Step 1 - Define Shape Operations in Egglog**
- Extend the egglog program in `egglog_program.rs` to define new function symbols for shape ops:
  - `Reshape(expr, shape)` - reshape to new dimensions
  - `Permute(expr, axes)` - reorder dimensions
  - `Transpose(expr, dim1, dim2)` - swap two dimensions
  - `Expand(expr, shape)` - broadcast to larger shape
  - `Squeeze(expr)` - remove singleton dimensions

**Step 2 - Add Rewrite Rules**
- Implement algebraic rules for shape operations:
  - **Reshape Identity**: `reshape(x, original_shape(x)) = x`
  - **Reshape Composition**: `reshape(reshape(x, s1), s2) = reshape(x, s2)`
  - **Permute Identity**: `permute(x, [0,1,2,...]) = x`
  - **Permute Composition**: `permute(permute(x, p1), p2) = permute(x, compose(p1, p2))`
  - **Transpose as Permute**: Express transpose in terms of permute for unified handling
  - **Commute with Elementwise**: When safe, push shape ops through elementwise operations
    - Example: `reshape(add(a, b), s) = add(reshape(a, s), reshape(b, s))` when shapes are compatible
  - **Expand Fusion**: `expand(expand(x, s1), s2) = expand(x, s2)`

**Step 3 - Implement Helper Functions**
- Add helper functions for:
  - Permutation composition (multiply permutation matrices conceptually)
  - Shape compatibility checking
  - Detecting identity permutations

**Step 4 - Update the Parser**
- Extend `parse_extracted_term` in `optimize/parse.rs` to handle new shape op AST nodes
- Map egglog's extracted shape operations back to the Graph representation

**Step 5 - Relax Safety Checks**
- Update `is_optimize_safe` to allow shape operations:
  - Continue allowing: Load, Const, elementwise ops (Add, Sub, Mul, Div, Exp, Ln, Sqrt, Neg)
  - Now also allow: Reshape, Permute, Transpose, Expand, Squeeze
  - Continue rejecting: Reduce ops (handle separately), Slice (dynamic), Pad (complex), Flip (uncommon)
  - Still require float dtype (egglog rules use floating-point constants)

**Expected Impact**:
- Transformers heavily use reshape/permute in attention patterns
- Common optimization: eliminate redundant reshape chains (`[B,S,H] -> [B,S,h,d] -> [B,S,H]` collapses)
- Compose permutations in attention: `permute(permute(x, [0,2,1,3]), [0,1,3,2])` becomes single permute
- Enable constant folding through shape operations
- Estimated 10-20% speedup on attention-heavy workloads from eliminated shape ops

**Risk**: 
- Egglog search space expansion may increase optimization time
- Some rewrites may be incorrect if shape constraints not properly encoded

**Mitigation**:
- Add iteration and node count limits to egglog's run schedule
- Implement timeout mechanism (e.g., 100ms max optimization time)
- Extensive testing with known shape operation patterns
- Start with conservative rules, expand incrementally

---

## Phase 3: Uncertain Optimizations (Potential Impact, Needs Validation)

These optimizations may help with scaling but require benchmarking to confirm benefits outweigh costs.

### 3.1 Replace Dense Vectors with Sparse HashMaps in FusedKernel

**Current Issue**: `input_trackers` and `shape_source_map` are `Vec<Option<T>>` sized to the entire graph's node count, but typically only a handful of entries are `Some`.

**Proposed Change**:
- Change `input_trackers` from `Vec<Option<ShapeTracker>>` to `HashMap<NodeId, ShapeTracker>`
- Change `shape_source_map` from `Vec<Option<NodeId>>` to `HashMap<NodeId, NodeId>`

**Implementation**:
- Update `FusedKernel` struct definition in `schedule/fused_kernel.rs`
- Modify `collect_kernel_inputs` in `schedule/topo.rs` to insert into HashMap instead of indexing Vec
- Update all JIT compilation code paths to use `.get(&node_id)` instead of `[node_id.0]`
- Update kernel signature hashing to iterate HashMap entries instead of Vec indices

**Potential Benefits**:
- Memory savings: For a graph with 10,000 nodes but only 10 tracked inputs, saves approximately 80KB per kernel
- Cache locality: Iteration only touches actual entries, not empty Option slots
- No need to track separate counts like `num_tracked_inputs`

**Potential Drawbacks**:
- HashMap lookup overhead vs. direct array indexing in hot JIT path
- Slightly more complex code

**Validation Required**:
- Benchmark on typical GPT-2 graph sizes (usually 100-500 nodes)
- Measure both memory usage and JIT compilation time
- Compare cache hit rates and overall throughput

**Decision Criteria**:
- If graph sizes typically < 1000 nodes: Vec is probably faster
- If graph sizes > 5000 nodes or memory is constrained: HashMap wins
- Consider hybrid approach: use Vec for small graphs, HashMap for large

**Alternative Considered**:
- Use a sparse vector data structure (Vec of (NodeId, T) pairs, binary searched)
- Provides middle ground but more complex

---

### 3.2 Cache Consumer Counts in Context

**Current Issue**: `compute_consumer_counts` runs on every `build_schedule` call, though the graph is immutable during inference.

**Proposed Change**:
- Add an `Option<HashMap<NodeId, usize>>` to Context for cached consumer counts
- On first schedule build, compute and cache
- Invalidate cache when new tensor operations append to the graph
- Reuse cached counts on subsequent schedules

**Implementation**:
- Add `consumer_counts_cache: Arc<Mutex<Option<HashMap<NodeId, usize>>>>` to Context
- Modify `build_schedule` to accept and use the cache
- In Tensor construction methods, set cache to None when graph mutates
- During inference (after graph construction), cache persists

**Potential Benefits**:
- Eliminates O(nodes) pass on every schedule build during inference
- Minor but measurable in scenarios with many small forward passes

**Potential Drawbacks**:
- Added complexity in cache invalidation logic
- Risk of stale cache if invalidation logic has bugs

**Validation Required**:
- Profile how much time is spent in `compute_consumer_counts` for typical workloads
- If < 1% of total time, optimization may not be worth the complexity

**Decision Criteria**:
- Implement only if profiling shows > 2% time spent here
- Otherwise, defer to later cleanup phase

**POSSIBLE_OPTIMIZATION**:
- Forward fusion currently may need O(nodes) scans to locate a single consumer node.
- If profiling shows scheduler overhead from these scans, add a reverse-edge map
  (producer -> consumers) during topo/schedule construction so single-consumer
  lookups become O(1)-ish instead of graph-wide scans.

---

## Phase 4: Deferred Optimizations (Future Work)

These optimizations are valuable but lower priority or require significant infrastructure changes.

### 4.1 Specialized Reduce Kernels (Deferred - Bitter Lesson)

**Why Deferred**: Following the "bitter lesson" philosophy, we should focus on general methods that scale with compute rather than hand-coding pattern-specific optimizations.

**Current Approach**: General nested-loop reduce kernel handles all cases

**Potential Specialized Patterns Identified**:
- Sum over last dimension (common in LayerNorm)
- Sum over middle dimensions (pooling)
- Global reduction (loss calculation)
- KeepDims variants

**Better Approach** (Future):
- Invest in auto-vectorization and loop optimization at the Cranelift level
- Explore MLIR-style progressive lowering with automatic tiling/vectorization
- Use polyhedral optimization techniques to automatically optimize loop nests
- Let the compiler/JIT discover optimal patterns rather than hand-coding them

**Action**: Document these patterns in benchmarks for future automated optimization efforts, but do not hand-code specializations now.

---

### 4.2 Two-Pass Scheduling to Eliminate Retroactive Removal

**Current Issue**: The scheduler emits schedule items, then retroactively removes some via `schedule.retain()` when it discovers they should be inlined.

**Why Deferred**: While cleaner conceptually, the current approach works correctly and the retroactive removal is fast (just a Vec filter). The benefit of refactoring is primarily code clarity, not performance.

**Future Approach** (if pursued):
- Implement two-pass scheduling:
  - Pass 1: Analyze graph and mark all nodes that will be inlined
  - Pass 2: Emit schedule items only for non-inlined nodes
- Requires careful design to handle reduce-fused kernels that discover inlineable nodes during fusion

**Complexity**: High, with risk of introducing subtle bugs in fusion logic

**Decision**: Defer until after Phase 1-3 complete and we have comprehensive benchmarks to validate correctness

---

### 4.3 Hierarchical Subgraph Optimization

**Current Issue**: `is_optimize_safe` is all-or-nothing: either the entire reachable subgraph is safe, or none of it gets optimized.

**Better Approach**:
- Identify maximal safe subgraphs within a larger graph
- Optimize each safe region independently
- Use unsafe operations (reduces, complex shape ops) as natural boundaries

**Implementation Sketch**:
- Walk graph from root, identifying "islands" of safe operations
- Create separate optimization problems for each island
- Reassemble optimized islands with original boundary nodes

**Benefits**:
- More optimization opportunities without expanding egglog to handle all operation types
- Natural decomposition of large graphs

**Complexity**: Moderate to high. Requires careful subgraph extraction and stitching logic.

**Decision**: Defer to Phase 5 or later, after egglog shape op support is stable. If egglog proves unstable with shape ops, this becomes a valuable fallback.

---

### 4.4 SIMD Vectorization for Reduces

**Why Deferred**: Requires significant Cranelift SIMD infrastructure work.

**Approach** (Future):
- Use Cranelift's SIMD instructions for horizontal reductions
- Emit vectorized loads, SIMD operations, and horizontal sum/max/min
- Particularly beneficial for reductions over small dimensions (e.g., 768 hidden dim)

**Estimated Impact**: 2-4x speedup on reduction operations

**Complexity**: High. Requires:
- Cranelift SIMD type and instruction support
- Target-specific code generation (AVX2, AVX-512, NEON, etc.)
- Fallback paths for unsupported targets
- Alignment handling

**Decision**: Document as future work. Potentially a separate effort after core optimizations stabilize.

---

### 4.5 Structured Error Types

**Current Issue**: Most errors are `anyhow::Result`, making it hard to distinguish error categories programmatically.

**Better Approach**:
- Define structured error types for different subsystems:
  - `GraphError` for invalid graph operations
  - `OptimizationError` for egglog failures
  - `JitError` for Cranelift compilation failures
  - `ExecutionError` for runtime errors
- Implement proper error context with source locations

**Benefits**:
- Better error messages for users
- Programmatic error handling (e.g., retry without optimization on egglog timeout)
- Easier debugging

**Complexity**: Moderate. Requires defining error types and updating all error sites.

**Decision**: Defer to final cleanup phase. Focus on correctness and performance first.

---

### 4.6 JIT Expression Builder API Cleanup

**Possible Optimization**: In `jit/expr.rs`, consider folding the per-call `byte_offset`
into a small expression-call context wrapper so recursive expression building can carry
fewer explicit parameters at call sites.

**Status**: Documented for future cleanup; do not implement yet.

---

## Implementation Timeline

### Sprint 1: Quick Wins (1 week)
- [ ] Remove BufferPool zeroing
- [ ] Implement LRU cache
- [ ] Precompute kernel metadata
- [ ] Remove dead code
- [ ] Add comprehensive benchmarks for transformer patterns

**Deliverable**: 10-15% performance improvement, bounded memory usage

---

### Sprint 2: Egglog Shape Operations (2 weeks)
- [ ] Define shape ops in egglog
- [ ] Implement basic rewrite rules (reshape, permute composition)
- [ ] Add advanced rules (commute with elementwise)
- [ ] Update parser for shape ops
- [ ] Relax `is_optimize_safe`
- [ ] Add timeout/limits to egglog
- [ ] Extensive testing of shape op rewrites

**Deliverable**: 10-20% additional improvement on attention-heavy workloads

---

### Sprint 3: Validation & Tuning (1 week)
- [ ] Profile HashMap vs Vec tradeoff
- [ ] Profile consumer count caching benefit
- [ ] Decide on Phase 3 optimizations based on data
- [ ] Implement selected Phase 3 items
- [ ] Comprehensive benchmark suite
- [ ] Memory profiling

**Deliverable**: Data-driven decisions, final optimized build

---

### Sprint 4: Documentation & Cleanup (3 days)
- [ ] Document all optimization decisions
- [ ] Clean up any remaining technical debt
- [ ] Add inline documentation for complex algorithms
- [ ] Create performance regression test suite
- [ ] Write migration guide if any APIs changed

**Deliverable**: Production-ready optimized engine

---

## Success Metrics

### Performance Targets
- **Overall**: 30-40% throughput improvement on GPT-2 inference
- **Memory**: Bounded cache sizes, stable memory usage over time
- **Latency**: No optimization-induced latency spikes (timeout mechanisms working)

### Code Quality Targets
- All optimizations covered by tests
- No regression in existing test suite
- Benchmark suite for ongoing performance tracking
- Clear documentation of all major design decisions

---

## Risk Management

### Risk: Egglog with Shape Ops Too Slow
**Mitigation**: Implement timeout (100ms), node limits. Fall back to non-optimized path on timeout.

### Risk: HashMap Slower Than Vec
**Mitigation**: Benchmark early in Sprint 3. If slower, keep Vec or implement hybrid approach.

### Risk: LRU Cache Thrashing
**Mitigation**: Tune cache sizes. Add metrics for hit rates. Implement size-aware eviction if needed.

### Risk: Optimization Introduces Correctness Bugs
**Mitigation**: Differential testing (run both optimized and unoptimized, compare results). Extensive fuzzing with random graphs.

---

## Open Questions

1. **Cache Sizing**: Default 1000 kernels / 500 plans appropriate for GPT-2 inference? Need to validate with real workloads.

2. **Optimization Timeout**: Is 100ms reasonable? Too generous? Too strict?

3. **Breaking Changes**: Are there external consumers of internal APIs (like `FusedKernel` structure) that need migration support?

4. **Vectorization**: Should SIMD reduces be prioritized higher given "bitter lesson" concerns about hand-coding?

5. **Graph Mutation**: Current design assumes graph is append-only during inference. If we add graph mutation/pruning later, how does caching change?

---

## Conclusion

This plan prioritizes high-confidence, high-impact optimizations first, followed by riskier extensions to egglog. Uncertain optimizations are explicitly flagged for validation before full implementation. Specialized hand-coded optimizations are deferred in favor of general methods that scale with compute.

The phased approach allows for early wins (Sprint 1) while building toward more ambitious goals (Sprint 2-3), with explicit decision points based on measured data rather than speculation.
