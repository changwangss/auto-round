# Diffusion Offload Calibration Forward Design

## Problem

AutoRound captures diffusion block inputs by replacing each target block's
`forward`. Diffusers model CPU offload uses Accelerate to install an outer
`forward` wrapper that moves a component to the accelerator before invoking the
module's saved `_old_forward`. Replacing `forward` after offload is enabled
bypasses that wrapper. FLUX calibration then executes the transformer on CPU
even though `low_gpu_mem_usage` names an accelerator.

The observed failure signal is sustained CPU use, near-zero accelerator use,
and no progress beyond the first denoising step.

## Decision

Preserve Accelerate's outer `forward` wrapper. When a calibration target has an
Accelerate hook and callable `_old_forward`, install AutoRound's capture wrapper
in `_old_forward`. Continue replacing `forward` for ordinary modules.

Record which attribute was replaced on each module. Recovery must restore that
same attribute in all normal and exceptional exits:

- Accelerate-managed module: restore `_old_forward`; leave `forward` intact.
- Ordinary module: restore `forward` using the existing behavior.

This keeps the runtime order:

1. Accelerate pre-forward hook moves the component to the accelerator.
2. AutoRound captures block inputs to CPU.
3. The original block forward executes when calibration must continue.
4. Accelerate post-forward handling performs its configured offload.

The change is generic to Accelerate-managed modules and does not add FLUX,
Nunchaku, or SVDQuant-specific imports.

## Compatibility

Modules are treated as Accelerate-managed only when both `_hf_hook` and a
callable `_old_forward` are present. Existing modules that coincidentally define
only one attribute retain the ordinary replacement path.

The existing `orig_forward`, `_true_orig_forward`, positional-wrapper, and
layer-forward-hook behavior remains unchanged for ordinary modules. Temporary
replacement markers are removed during recovery.

## Tests

Add CPU regression tests with a small fake Accelerate wrapper to verify:

- calibration installation preserves the outer `forward` object;
- the outer pre-forward action runs before AutoRound's capture function;
- recovery restores `_old_forward` and preserves the outer wrapper;
- recovery still runs after calibration raises;
- ordinary modules retain the current direct-`forward` replacement behavior.

Run the focused CPU calibration and diffusion tests, then the SVDQuant export
tests. Finally run a real FLUX `1 sample x 2 steps` GPU smoke test with
`low_gpu_mem_usage`; success requires sustained accelerator execution and
completion of both denoising steps. Only then restart `64 samples x 50 steps`.

