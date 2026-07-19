# Diffusion Offload Calibration Forward Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve Accelerate's model-offload `forward` wrapper while AutoRound captures diffusion block inputs.

**Architecture:** Calibration will select the replaceable forward slot per module. Ordinary modules continue to replace `forward`; Accelerate-managed modules replace `_old_forward` while retaining the outer offload wrapper. Recovery restores the recorded slot on normal and exceptional exits.

**Tech Stack:** Python, PyTorch modules, Accelerate forward hooks, pytest, Ruff, Black.

---

### Task 1: Lock Down Accelerate Forward Ordering

**Files:**
- Modify: `test/test_cpu/utils/test_calibration_hooks.py`
- Test: `test/test_cpu/utils/test_calibration_hooks.py`

**Step 1: Write the failing test**

Create a fake module whose outer `forward` records an `offload_pre_forward` event and delegates to `_old_forward`. Install calibration capture through `replace_forward_with_hooks`, then assert the outer wrapper remains installed and runs before capture.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest -q test/test_cpu/utils/test_calibration_hooks.py -k accelerate
```

Expected: the current implementation replaces the outer `forward`, so the ordering assertion fails.

### Task 2: Preserve and Recover the Accelerate Wrapper

**Files:**
- Modify: `auto_round/calibration/hooks.py`
- Modify: `auto_round/context/model.py`
- Test: `test/test_cpu/utils/test_calibration_hooks.py`

**Step 1: Implement slot-aware installation**

When `_hf_hook` exists and `_old_forward` is callable, save the original callable in `orig_forward`, replace `_old_forward`, and record `_old_forward` as the calibration replacement slot. Otherwise retain direct `forward` replacement.

**Step 2: Implement slot-aware recovery**

When the replacement marker names `_old_forward`, restore `_old_forward` from the true original and leave the outer `forward` untouched. Remove temporary calibration attributes. Preserve existing ordinary and positional-wrapper recovery behavior.

**Step 3: Run focused tests**

```bash
python -m pytest -q test/test_cpu/utils/test_calibration_hooks.py test/test_cpu/models/test_diffusion.py
```

Expected: all tests pass.

### Task 3: Regression and Formatting Verification

**Files:**
- Verify: `auto_round/calibration/hooks.py`
- Verify: `auto_round/context/model.py`
- Verify: `test/test_cpu/utils/test_calibration_hooks.py`

**Step 1: Run related CPU suites**

```bash
python -m pytest -q \
  test/test_cpu/utils/test_calibration_hooks.py \
  test/test_cpu/models/test_diffusion.py \
  test/test_cpu/export/test_svdquant_flux_adapter.py \
  test/test_cpu/export/test_svdquant_nunchaku_export.py
```

**Step 2: Run Ruff and Black checks**

Run Ruff on changed Python files and Black 26.5.1 in check mode. Both must pass without rewriting unrelated files.

### Task 4: Real GPU Smoke and Full Calibration Restart

**Files:**
- Use: `/home/user2/data/xixi/run_autoround_flux_mxfp4_rtn_calibration.sh`
- Log: `/home/user2/data/xixi/autoround-flux-mxfp4-r32-smooth-r100-early-rtn-n1-s2.log`

**Step 1: Run a `1 sample x 2 steps` smoke**

Use GPU 0 with pure RTN and `low_gpu_mem_usage`. Verify both denoising steps complete, GPU kernels execute, and the process advances beyond input capture.

**Step 2: Restart `64 samples x 50 steps`**

Only after the smoke passes, remove its incomplete output if any and launch the requested calibration in a persistent tmux window. Report PID, log path, GPU memory, and host available memory.

