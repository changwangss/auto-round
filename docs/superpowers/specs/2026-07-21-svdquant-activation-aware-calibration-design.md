# SVDQuant Activation-Aware Calibration Design

## Goal

Make AutoRound SVDQuant smooth search and residual outer iteration optimize the
same W4A4 computation deployed by Nunchaku. Preserve the existing Nunchaku
export format and record run provenance in logs only.

## Problem

The residual weights are fake-quantized during candidate evaluation, but the
candidate wrapper feeds BF16 activations to both the residual and low-rank
branches. Nunchaku dynamically quantizes the residual branch activation to
MXFP4 while keeping the low-rank branch in high precision. Smooth factors can
therefore improve the calibration approximation while making deployed W4A4
outputs worse.

## Data Flow

For each candidate and captured input `x`:

1. Apply the selected smooth coordinate transform to produce `x_hat`.
2. Feed `MXFP4_QDQ(x_hat)` to the quantized residual linear.
3. Feed the original BF16 `x_hat` to the BF16 low-rank down/up branch.
4. Add both outputs and score the existing projection, attention, or block
   output against its BF16 reference.

Smooth candidate search and every residual outer iteration use this same
wrapper. Final transformed modules remain ordinary `SVDQuantLinear` modules;
activation QDQ is calibration-only and is not serialized.

## Quantization Contract

Activation QDQ is built from each target linear's existing `act_data_type`,
`act_bits`, `act_group_size`, and `act_sym` attributes. Shared smooth groups
must use one identical activation scheme. Deployable MXFP4 requires 4 bits and
group size 32. Missing or inconsistent attributes fail with a module-specific
error instead of silently evaluating W4A16.

Both weight and activation QDQ use the Nunchaku UE8M0 scale rule
`ceil(log2(max_abs / 6))`. This differs from AutoRound's older
`floor(log2(max_abs)) - 2` MXFP4 approximation and prevents calibration from
optimizing a quantizer different from the exported Nunchaku kernel.

## Smooth Safeguards

`smooth_eps` clamps activation and weight span bases before exponentiation.
Candidates must remain finite and positive after conversion to the projection
deployment dtype, and their reciprocals must also remain finite and positive.
No arbitrary fixed min/max clamp is introduced because it would diverge from
the search objective.

For every selected smooth group, log the alpha, beta, output error, minimum,
maximum, dynamic-range ratio, and counts below `1e-3` or above `20`. Extreme
selected scales emit a warning.

## Provenance

At calibrated zero-shot quantization startup, log the model source, dataset,
`nsamples`, diffusion steps, retained calibration-call limit, smooth grid
count, rank, residual iteration limit, early-stop setting, residual method,
AutoRound version, and source checkout commit when available. Do not add these
fields to safetensors metadata.

## Testing

- Unit-test activation MXFP4 QDQ validation and shape/dtype/device preservation.
- Prove the candidate wrapper quantizes only the residual input.
- Prove smooth and residual scoring both request activation-aware wrappers.
- Test epsilon handling, deployability validation, selected-candidate logs, and
  extreme-scale warnings.
- Test provenance logging with and without a Git checkout.
- Run existing SVDQuant, smooth adapter, calibrated zero-shot, CLI, and export
  regressions.
