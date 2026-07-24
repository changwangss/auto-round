# SVDQuant SignRound Bounded Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the existing SVDQuant smooth calibration-call limit bound the shared smooth and SignRound diffusion calibration pool.

**Architecture:** DataDrivenCompressor detects an enabled SVDQuant smooth preprocessor and installs the same deterministic per-block call selector already used by calibrated zero-shot RTN. Diffusion still executes the complete trajectory, but only selected calls are copied into the shared block pool. No-smooth SignRound remains unchanged.

**Tech Stack:** Python 3.12, PyTorch, pytest, AutoRound calibration hooks.

---

### Task 1: Lock Down The Routing Behavior

**Files:**
- Modify: `test/test_cpu/core/test_pipeline_fail_fast.py`

1. Add a failing test proving smooth SVDQuant config enables bounded selection in `DataDrivenCompressor`.
2. Add coverage proving block groups reuse the same indices.
3. Add coverage proving no-smooth SignRound retains all calls.

### Task 2: Connect The Existing Selector

**Files:**
- Modify: `auto_round/compressors/data_driven.py`

1. Resolve the call limit only from an enabled smooth preprocessor.
2. Prepare deterministic uniform indices before diffusion capture.
3. Maintain an independent call counter per block-group entry.
4. Leave the selector inactive for no-smooth configurations.

### Task 3: Verify The Combined Path

**Files:**
- Test: `test/test_cpu/core/test_pipeline_fail_fast.py`
- Test: `test/test_cpu/algorithms/test_svdquant*.py`

1. Run the new regression tests.
2. Run focused SVDQuant, pipeline, calibration, and export tests.
3. Run Ruff and `git diff --check`.
