- search vs compile still disagree on cardinality.
ScheduleSearcher::search returns a ranked Vec<(ScheduleDecision, CostEstimate)>, but compile binds that result to a single decision and passes it into lower. That is an API mismatch unless search(...) is a wrapper that already selects one winner. The spec should say which one is intended.
- axis needs an explicit coordinate rule after transforms.
Opt is applied in sequence, and the lowering examples use axes after earlier tiling has already changed the loop nest. That implies axis indices are relative to the current transformed nest, not the original one. The spec should state that directly, otherwise Tile/Vectorize composition is ambiguous.
- TensorType.layout is more permissive than the polyhedral lowering path.
TensorType allows Strided(Vec<Dim>), but strides_to_access later rejects symbolic strides and only accepts concrete stride values. That is acceptable, but it should be labeled clearly as a normalization/lowering restriction, not as a general property of layouts. Otherwise the reader may assume all Strided layouts are directly polyhedrally representable.
