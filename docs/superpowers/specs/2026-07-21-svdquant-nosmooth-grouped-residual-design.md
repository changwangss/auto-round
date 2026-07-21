# SVDQuant No-Smooth Grouped Residual Design

## Goal

Make FLUX `no-smooth + RTN + SVDQuant` use the GPU efficiently and avoid redundant per-projection residual iterations while preserving data-free RTN semantics.

## Design

- Use the FLUX smooth-group adapter even when smooth calibration is disabled.
- Assign an identity smooth factor to every group; no activation calibration or smooth search is performed.
- Jointly decompose projections that share an input, including fused QKV groups.
- Score each residual iteration with weight reconstruction error. Keep the best candidate and stop only when a later candidate is worse, matching the existing strict early-stop behavior.
- Log progress every 10 iterations and log the selected iteration or early-stop event.
- Move only the current RTN block to the selected CUDA device, then use the existing block cleanup path.

## Compatibility

- The CLI remains unchanged.
- `--enable_svdquant_smooth` is still the only option that enables calibration and smooth search.
- Generic models without a registered grouping adapter retain the existing per-Linear no-smooth path.
- The Nunchaku export schema remains unchanged, but fused projections now reach export with a shared low-rank decomposition instead of being recomputed there.

## Validation

- Unit tests for identity-group decomposition, shared low-rank factors, strict early stopping, progress logging, and CUDA block placement.
- Existing SVDQuant, FLUX adapter, exporter, CLI, and lint suites.
- A card-0 FLUX smoke run confirming GPU utilization, first-block progress, successful export, Nunchaku loading, and image generation.
