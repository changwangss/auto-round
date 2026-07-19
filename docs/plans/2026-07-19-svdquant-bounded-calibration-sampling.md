# SVDQuant Bounded Calibration Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bound cache-free diffusion SVDQuant smooth calibration to a deterministic maximum number of retained transformer calls while still executing the complete diffusion trajectory.

**Architecture:** Add a small deterministic index-selection helper and expose an optional call-selection extension point in the generic block capture hook. `CalibratedZeroShotCompressor` owns the configured limit and per-block counters, while `DiffusionMixin` prepares the plan immediately before cache collection. Once the initial cache contains only K states, existing FP block propagation and smooth collection remain bounded by K without changing RTN or SignRound behavior.

**Tech Stack:** Python 3.12, PyTorch, pytest, argparse, Ruff, AutoRound calibration/compressor pipeline.

---

### Task 1: Configuration And CLI Contract

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/config.py`
- Modify: `auto_round/cli/parser.py`
- Modify: `auto_round/cli/algorithms.py`
- Modify: `auto_round/compressors/entry.py`
- Modify: `auto_round/autoround.py`
- Test: `test/test_cpu/algorithms/test_svdquant.py`
- Test: `test/test_cpu/utils/test_cli_usage.py`
- Test: `test/test_cpu/core/test_pipeline_fail_fast.py`

**Step 1:** Add failing tests for the Python default of 128, explicit value 16, compatibility-constructor forwarding, CLI forwarding, and rejection of booleans/non-integers/non-positive values.

**Step 2:** Run the focused tests and confirm they fail because `smooth_max_calibration_calls` is not defined.

**Step 3:** Add `smooth_max_calibration_calls: int = 128` to `SVDQuantConfig`, validate it, include it in `__repr__`, and thread `svdquant_smooth_max_calibration_calls` through every existing SVDQuant CLI/API compatibility path.

**Step 4:** Re-run the focused tests and confirm they pass.

### Task 2: Deterministic Uniform Call Selection

**Files:**
- Create: `auto_round/calibration/sampling.py`
- Test: `test/test_cpu/calibration/test_sampling.py`

**Step 1:** Add failing tests proving that selection retains all calls when `N <= K`, selects the midpoint for `K == 1`, and otherwise returns exactly K monotonic indices including 0 and `N - 1` deterministically.

**Step 2:** Implement `uniform_call_indices(total_calls, max_calls)` without random state or tensor allocation.

**Step 3:** Run the sampling tests and Ruff.

### Task 3: Bound Initial Diffusion Block Cache

**Files:**
- Modify: `auto_round/calibration/hooks.py`
- Modify: `auto_round/compressors/calibrated_zero_shot.py`
- Modify: `auto_round/compressors/diffusion_mixin.py`
- Test: `test/test_cpu/core/test_pipeline_fail_fast.py`

**Step 1:** Add failing tests for an optional capture selector: selected calls are copied, unselected calls still invoke the original block forward, two block-group entries use the same indices, and counters reset for a new collection.

**Step 2:** Refactor the block capture hook to call the original block through one helper and consult an optional `should_cache_calibration_call(name)` method before any CPU copy.

**Step 3:** Add preparation and selection methods to `CalibratedZeroShotCompressor`. Compute the expected number of diffusion transformer calls from prompt batches and inference steps, create one shared deterministic index set, and maintain an independent call counter per block-group entry.

**Step 4:** Have `DiffusionMixin.quantize()` invoke the optional preparation method immediately before `try_cache_inter_data_gpucpu()`. Do not import SVDQuant from the mixin.

**Step 5:** Assert through tests that ordinary compressors without the optional selector preserve existing capture behavior.

### Task 4: Smooth Cache Bound And Regression Coverage

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py` only if an explicit defensive cap is still needed after the bounded block cache is connected
- Modify: `test/test_cpu/algorithms/test_svdquant_smooth_adapters.py`
- Modify: `test/test_cpu/core/test_pipeline_fail_fast.py`

**Step 1:** Add a test that replays more source calls than K through the full selector plus block propagation boundary and proves the smooth hook receives and retains no more than K calls.

**Step 2:** Preserve shared `CapturedEvaluation` objects for groups using the same evaluation module.

**Step 3:** Re-run SVDQuant, smooth-adapter, calibrated-zero-shot, diffusion BlockIO, and export tests.

### Task 5: Documentation And Verification

**Files:**
- Modify: `docs/svdquant_nunchaku_mxfp4_review.md`
- Modify: `/home/user2/data/xixi/run_autoround_flux_mxfp4_rtn_calibration.sh`

**Step 1:** Document the default 128-call limit, explain that calls are selected from the full prompt-step pool, and show the 120 GiB override `--svdquant_smooth_max_calibration_calls 16`.

**Step 2:** Add an optional fourth script argument for the maximum call count, defaulting to 16 for this machine's operational script, without changing the API default of 128.

**Step 3:** Run focused pytest suites, Ruff on all touched Python files, and `git diff --check`.

**Step 4:** Launch physical GPU 1 with `nsamples=8`, `num_inference_steps=20`, and K=16. Record the cgroup `oom_kill` counter before launch, verify initial cache retention is 16, and require the run to pass `transformer_blocks.0` without incrementing `oom_kill`.

**Step 5:** If block 0 passes, allow the full quantization/export to continue. Validate the exported directory and Nunchaku onefile metadata, then run the existing Nunchaku load/generation smoke if disk and GPU resources remain available.
