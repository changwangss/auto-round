# SVDQuant Output-Error Residual Early Stop Design

## Goal

Align smooth SVDQuant low-rank residual iteration with DeepCompressor's calibration objective and early-stop rule without increasing persistent calibration memory.

## Scope

- Apply output-error scoring only to grouped residual iteration when smooth calibration data exists.
- Keep no-smooth residual iteration and terminal RTN behavior unchanged.
- Keep `residual_iters` as the hard maximum; the SVDQuant preset may use 100.

## Data Flow

Each `SmoothGroupCalibration` already owns bounded `CapturedEvaluation` records containing CPU inputs and reference outputs. For every residual candidate:

1. Build the candidate low-rank and MXFP4-QDQ residual weights.
2. Temporarily install candidate wrappers into the evaluation module.
3. Move one captured call at a time to the projection device.
4. Accumulate squared output error against the existing reference output.
5. Release call-local outputs before evaluating the next call.

No candidate output or per-iteration output history is retained.

## Selection And Early Stop

- A finite candidate whose output error is less than or equal to the best error becomes the new best candidate.
- When early stop is enabled, the first finite candidate with error greater than the best error stops iteration.
- Non-finite output error is a failed candidate and stops iteration.
- The final wrappers are built from the best candidate, not necessarily the last candidate.

This matches DeepCompressor's `error <= best_error` continuation rule.

## Memory

Persistent CPU calibration memory remains bounded by `smooth_max_calibration_calls`. GPU memory holds only the current candidate, one moved calibration call, and its output. Iteration count increases runtime but does not multiply retained outputs.

## Verification

- Unit-test online output SSE and ensure each call is evaluated independently.
- Unit-test equality continuation and first-worse early stop.
- Unit-test best-candidate restoration.
- Preserve existing smooth-search, dtype, residual, export, and no-smooth tests.
