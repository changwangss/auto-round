# SVDQuant Output-Error Residual Early Stop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Score smooth SVDQuant residual candidates with bounded calibration output SSE and stop at the first candidate worse than the best.

**Architecture:** Extract a reusable online group-output scoring helper from smooth candidate search. Pass `SmoothGroupCalibration`, block, and local module names into grouped residual decomposition so every residual candidate can be temporarily installed, scored call-by-call, and restored without retaining outputs.

**Tech Stack:** Python 3.12, PyTorch, pytest, Ruff.

---

### Task 1: Output Scoring Contract

**Files:**
- Modify: `test/test_cpu/algorithms/test_svdquant_smooth_adapters.py`

1. Add a failing test that supplies two captured calls and verifies output SSE is accumulated online.
2. Add a failing test with controlled candidate errors proving equal error continues and the first worse error stops.
3. Run the two tests and confirm failure against weight-reconstruction scoring.

### Task 2: Reusable Online Group Scoring

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`

1. Extract wrapper installation, one-call-at-a-time forward, normalized output comparison, and restoration into a helper.
2. Keep smooth alpha/beta candidate scoring on that helper.
3. Ensure only a CPU float64 scalar is accumulated and no candidate outputs are retained.
4. Run existing smooth-search tests.

### Task 3: DeepCompressor Residual Selection Semantics

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`
- Modify: `test/test_cpu/algorithms/test_svdquant_smooth_adapters.py`

1. Thread `SmoothGroupCalibration`, block, and local names into `_decompose_group` and `_iterate_group_residual`.
2. Build candidate wrappers from each iteration's deployed low-rank and QDQ residual.
3. Score output SSE through the reusable helper.
4. Update best on `error <= best_error`; with early stop, break on the first finite worse candidate.
5. Restore and materialize the best candidate after iteration.
6. Keep no-smooth `_iterate_residual` unchanged.

### Task 4: Documentation And Verification

**Files:**
- Modify: `docs/svdquant_nunchaku_mxfp4_review.md`

1. Document output-error residual selection, bounded memory, and the 100-iteration maximum.
2. Run SVDQuant smooth, residual, pipeline, and export tests.
3. Run Ruff and `git diff --check`.
4. Run a reduced GPU smoke configuration and inspect iteration logs, RAM, and OOM counters.
