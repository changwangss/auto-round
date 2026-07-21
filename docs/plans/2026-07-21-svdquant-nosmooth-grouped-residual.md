# SVDQuant No-Smooth Grouped Residual Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run FLUX no-smooth SVDQuant residual iteration on shared projection groups with identity scales and weight-error early stopping.

**Architecture:** Reuse the existing FLUX smooth-group discovery without enabling calibration. Add a data-free grouped decomposition path that scores reconstructed grouped weights, while retaining the per-Linear fallback for models without an adapter.

**Tech Stack:** Python, PyTorch, pytest, Ruff

---

### Task 1: Define data-free grouped behavior

**Files:**
- Modify: `test/test_cpu/algorithms/test_svdquant_smooth_adapters.py`
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`

1. Add failing tests showing that no-smooth FLUX groups use identity scales and shared low-rank factors without calibration objects.
2. Run the focused tests and confirm failure.
3. Route supported blocks through group discovery when smooth is disabled.
4. Implement grouped residual scoring with weight reconstruction error.
5. Run the focused tests and confirm success.

### Task 2: Add residual iteration observability

**Files:**
- Modify: `test/test_cpu/algorithms/test_svdquant.py`
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`

1. Add failing tests for 10-iteration progress, early-stop, and final-selection logs.
2. Add concise INFO logs without changing strict error comparison semantics.
3. Run focused tests and Ruff.

### Task 3: Validate integration

**Files:**
- Modify: `auto_round/compressors/zero_shot.py`
- Modify: `test/test_cpu/core/test_pipeline_fail_fast.py`

1. Retain and verify the current-block CUDA placement regression test.
2. Run all SVDQuant, FLUX adapter, export, CLI, and compressor tests.
3. Run Ruff and `git diff --check`.
4. Start a card-0 no-smooth RTN rank-32 residual-100 smoke run and verify GPU execution plus first-block progress.
