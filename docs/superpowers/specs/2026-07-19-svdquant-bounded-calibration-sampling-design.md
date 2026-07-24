# SVDQuant Bounded Calibration Sampling Design

## Goal

Bound the host-memory cost of SVDQuant smooth calibration independently of
`nsamples * num_inference_steps`, without adding another diffusion or block
replay pass. The configured calibration run still executes every diffusion
step, but only a deterministic, uniformly distributed subset of calls is
retained for block propagation and smooth candidate scoring.

## Motivation

The calibrated zero-shot path currently limits the number of block-group entry
points that are hooked, but each entry retains every diffusion call. The smooth
hooks then copy projection inputs and evaluation inputs and outputs for every
retained call in every smooth group. On a container with a 120 GiB cgroup limit,
both `16 * 50` and `8 * 20` calibration runs completed the initial diffusion
cache and were OOM-killed when smooth collection started for
`transformer_blocks.0`.

DeepCompressor avoids treating `128 samples * 50 steps` as 6400 in-memory
calibration states during quantization. Its offline dataset contains one cache
file per prompt-step state, and the quantization loader selects 128 files from
that larger pool. AutoRound needs the equivalent selection boundary in its
direct, cache-free calibration path.

## Public Configuration

Add one positive integer setting:

```python
SVDQuantConfig(smooth_max_calibration_calls=128)
```

Expose the same setting through the CLI:

```bash
--svdquant_smooth_max_calibration_calls 128
```

The default is 128. A memory-constrained run can override it, for example:

```bash
--svdquant_smooth_max_calibration_calls 16
```

The value must be a positive integer. There is no sentinel value or deprecated
alias. Setting a value greater than or equal to the number of available calls
retains every call.

The setting is meaningful only when SVDQuant smoothing is enabled. It controls
the complete SVDQuant smooth calibration path, not only the final smooth hook.

## Sampling Policy

Sampling is deterministic and uniform over the flattened calibration call
sequence. Let:

- `N` be the total number of diffusion transformer calls produced by the
  calibration run after accounting for batching;
- `K = min(N, smooth_max_calibration_calls)`.

When `K == N`, every call is selected. When `K == 1 < N`, select the midpoint
call. Otherwise, generate `K` monotonically increasing indices spanning the
first through last call. The implementation must not use random state, so
repeated runs with the same inputs and options select the same states.

The flattened call order is the order observed by the diffusion calibration
hook. Uniformly spanning this sequence distributes retained states across both
prompts and diffusion timesteps for the existing sample-major execution order.

## Data Flow

### Initial diffusion calibration

The pipeline still executes all requested `nsamples * num_inference_steps`
calls because later diffusion states depend on earlier states. The calibration
collector increments a global call index for each transformer invocation.

For an unselected index, it performs no CPU copy and stores no block input.
For a selected index, it stores the inputs required by each configured block
group entry. FLUX currently has two such entries:

- `transformer_blocks.0`;
- `single_transformer_blocks.0`.

Both entries use the same selected call indices.

### Blockwise propagation

Only the selected `K` states enter the blockwise compressor. Full-precision
outputs from the current block become the selected inputs for the next block.
The existing dual-stream FLUX contract remains intact: both
`encoder_hidden_states` and `hidden_states` propagate between double-stream
blocks.

For terminal RTN, quantized block outputs are not propagated as calibration
inputs. For terminal SignRound, the selected states form the shared
calibration pool used by smooth search, residual output-error scoring, and
SignRound optimization; SignRound continues to propagate quantized outputs.

### Smooth collection and scoring

Each smooth group receives at most the same `K` selected calls. Its calibration
state contains at most:

- `K` projection input captures;
- `K` evaluation captures containing arguments, keyword arguments, and the
  reference output.

Groups that share an evaluation module continue to share the same captured
evaluation object when they consume the same call. Non-selected calls never
invoke `_detach_to_cpu(..., copy=True)` in the smooth hooks.

The alpha/beta candidate set, low-rank-aware RTN QDQ scoring, tie-breaking, and
final decomposition remain unchanged. The MSE objective is evaluated over the
selected calibration subset.

## Memory Invariant

The number of retained activation states for this path must be bounded by the
configured maximum rather than by the full call count:

```text
retained block-group entry states <= groups * K
retained per-smooth-group projection calls <= K
retained per-smooth-group evaluation calls <= K
```

The implementation may still hold the current block inputs and next block
outputs concurrently during propagation, plus model weights and framework
state. The setting does not impose a byte-level memory limit.

For a 1 TB machine, the default `K=128` targets DeepCompressor-like calibration
coverage. It does not imply that only 128 tensors are resident: each block
group and smooth group may retain several tensors per selected call. For the
current 120 GiB cgroup, commands must explicitly use `K=16` until an end-to-end
memory measurement establishes a higher safe value.

## Integration Boundaries

The sampling policy belongs to SVDQuant smooth calibration. It is enabled for
both calibrated zero-shot RTN and data-driven SignRound when SVDQuant smoothing
is active. It must not change:

- plain zero-shot RTN without smooth calibration;
- no-smooth SignRound or other data-driven compressors;
- non-diffusion calibration behavior unless it explicitly opts into the same
  bounded-call contract;
- diffusion output normalization used by existing loss-based algorithms;
- Nunchaku export format or tensor metadata.

The diffusion calibrator should expose a narrow optional call-selection hook or
policy supplied by the selected compressor. It should not import or
special-case the concrete SVDQuant transform. Both compressor paths reuse the
same `smooth_max_calibration_calls` setting; no separate SignRound sampling
option is introduced.

## Failure Handling

- Reject non-integer, boolean, zero, and negative maximum values during config
  construction.
- Fail clearly if smoothing is enabled but no selected calibration calls reach
  a required smooth group.
- Preserve existing shape checks for projection inputs and evaluation outputs.
- Always remove hooks and release calibration state when candidate scoring or
  decomposition raises.

## Verification

Unit tests must prove:

1. Python API and CLI default to 128 and accept an explicit value such as 16.
2. Invalid maximum values are rejected.
3. `N <= K` retains all calls.
4. `N > K > 1` retains exactly `K` deterministic indices including sequence
   endpoints, while `K == 1` selects the sequence midpoint.
5. Initial block-group caches and downstream propagated block inputs contain
   only selected calls.
6. Smooth groups retain no more than `K` projection and evaluation calls.
7. Shared FLUX evaluation modules still share captured evaluation objects.
8. FLUX dual-stream propagation remains correct.
9. Smooth SignRound uses the same selected calls for smooth search and
   SignRound optimization.
10. Plain RTN and no-smooth SignRound routing remain unchanged.

After focused CPU tests and Ruff pass, run a FLUX smoke calibration on physical
GPU 1 with:

```bash
--nsamples 8 \
--num_inference_steps 20 \
--svdquant_smooth_max_calibration_calls 16
```

The acceptance boundary is stronger than completing the initial cache: the run
must pass `transformer_blocks.0` without increasing the cgroup `oom_kill`
counter. A full successful export and Nunchaku-load smoke remain the final
end-to-end validation.
