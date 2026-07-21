# SVDQuant Activation-Aware Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make SVDQuant smooth and residual selection reproduce Nunchaku MXFP4 W4A4 activation behavior while adding smooth diagnostics and log-only provenance.

**Architecture:** Add a validated stateless activation QDQ helper beside residual weight QDQ. Candidate wrappers optionally apply it only to the residual branch, and all calibrated smooth/residual scoring enables that option. Smooth scale creation gains epsilon and deployment checks; the calibrated compressor logs run provenance once before quantization.

**Tech Stack:** Python 3.12, PyTorch, pytest, AutoRound quantization registry, Nunchaku MXFP4 export.

---

### Task 1: Activation QDQ Contract

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/residual.py`
- Test: `test/test_cpu/algorithms/test_svdquant.py`

1. Add failing tests for valid MXFP4 activation QDQ, missing attributes, invalid group size, and shape/dtype/device preservation.
2. Run the focused tests and confirm failure.
3. Add `ActivationQuantScheme` and `rtn_qdq_activation` using the registered AutoRound quantization function.
4. Run the focused tests and confirm success.

### Task 2: Deployment-Faithful Candidate Wrapper

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/wrapper.py`
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`
- Test: `test/test_cpu/algorithms/test_svdquant.py`
- Test: `test/test_cpu/algorithms/test_svdquant_smooth_adapters.py`

1. Add a failing test proving activation QDQ changes only the residual branch input.
2. Add failing tests proving smooth candidates and residual iterations enable activation QDQ.
3. Extend `SVDQuantLinear` with an optional, non-serialized activation processor.
4. Resolve and validate a shared activation scheme for each smooth group.
5. Enable activation QDQ in `_candidate_group_wrappers` and residual-iteration scoring, but not in final wrappers.
6. Run focused SVDQuant and smooth-adapter tests.

### Task 3: Smooth Scale Diagnostics And Protection

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/smooth.py`
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`
- Test: `test/test_cpu/algorithms/test_svdquant_smooth.py`

1. Add failing tests for epsilon-clamped spans and BF16 deployment validation.
2. Preserve alpha/beta/error in `SmoothCandidate` selection.
3. Log selected scale statistics and warn for channels below `1e-3` or above `20`.
4. Run smooth unit tests and lint touched files.

### Task 4: Log-Only Provenance

**Files:**
- Modify: `auto_round/compressors/calibrated_zero_shot.py`
- Test: `test/test_cpu/core/test_pipeline_fail_fast.py`

1. Add failing tests for a single structured provenance log and graceful unknown Git commit handling.
2. Collect existing compressor and SVDQuant config values without modifying model metadata.
3. Emit the provenance before calibration/quantization starts.
4. Run calibrated zero-shot and CLI forwarding tests.

### Task 5: Regression Verification

**Files:**
- Test: `test/test_cpu/algorithms/test_svdquant.py`
- Test: `test/test_cpu/algorithms/test_svdquant_smooth.py`
- Test: `test/test_cpu/algorithms/test_svdquant_smooth_adapters.py`
- Test: `test/test_cpu/core/test_pipeline_fail_fast.py`
- Test: `test/test_cpu/export/test_svdquant_nunchaku_export.py`

1. Run all focused CPU tests.
2. Run Ruff on every touched Python file.
3. Run `git diff --check`.
4. Confirm safetensors metadata generation is unchanged.
5. Record the recommended FLUX ablation commands without launching a full quantization run.

