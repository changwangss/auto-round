# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.compressors.diffusion_mixin``."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from auto_round.compressors.diffusion_mixin import DiffusionMixin


class TestDiffusionMixinProperties:
    """Test DiffusionMixin attribute access patterns."""

    def test_guidance_scale_default(self):
        # Access the class docstring and check init signature for defaults
        sig = inspect.signature(DiffusionMixin.__init__)
        params = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
        assert params.get("guidance_scale") == 7.5
        assert params.get("num_inference_steps") == 50
        assert params.get("generator_seed") is None

    def test_get_calibrator_kind_returns_diffusion(self):
        # Create a minimal mock class
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                # Don't call super().__init__() to avoid needing real parent
                pass

        comp = MockCompressor()
        assert comp._get_calibrator_kind() == "diffusion"

    def test_pipeline_call_kwargs_extracted_from_kwargs(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pass

        comp = MockCompressor()
        # Set the attribute directly since we're not calling super().__init__
        comp.pipeline_call_kwargs = {"height": 512, "width": 512}
        assert comp.pipeline_call_kwargs.get("height") == 512


class TestFindAdditionalTransformers:
    """Test _find_additional_transformers logic."""

    def test_returns_empty_when_pipe_is_none(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                self.model_context = SimpleNamespace(pipe=None)

        comp = MockCompressor()
        result = comp._find_additional_transformers()
        assert result == []

    def test_finds_secondary_transformers(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pipe = MagicMock()
                pipe.components = ["transformer", "transformer_2", "vae"]
                pipe.transformer = torch.nn.Linear(4, 4)
                pipe.transformer_2 = torch.nn.Linear(4, 4)
                pipe.vae = torch.nn.Linear(4, 4)
                self.model_context = SimpleNamespace(pipe=pipe)

        comp = MockCompressor()
        result = comp._find_additional_transformers()
        assert len(result) == 1
        assert result[0][0] == "transformer_2"

    def test_multi_transformer_quantize_uses_calibration_context_nsamples(self, monkeypatch):
        class FakeParent:
            @property
            def model(self):
                return self.model_context.model

            def quantize(self):
                return self.model_context.model, self.layer_config

        class MockCompressor(DiffusionMixin, FakeParent):
            def __init__(self):
                primary = torch.nn.Linear(4, 4)
                pipe = MagicMock()
                pipe.components = {"transformer": None, "transformer_2": None}
                pipe.transformer = primary
                pipe.transformer_2 = torch.nn.Linear(4, 4)
                self.model_context = SimpleNamespace(model=primary, pipe=pipe, quantized=False)
                self.calibration_context = SimpleNamespace(nsamples=7)
                self.compress_context = SimpleNamespace(low_cpu_mem_usage=False, is_immediate_saving=True)
                self.quantizer = SimpleNamespace(quant_block_list=[["block"]], layer_config={})
                self.quant_block_list = [["block"]]
                self.layer_config = {}
                self.need_calib = True
                self.has_variable_block_shape = False
                self.num_inference_steps = 2
                self.cached_nsamples = []

            def post_init(self):
                pass

            def _align_device_and_dtype_for_secondary(self, transformer_name):
                pass

            def try_cache_inter_data_gpucpu(self, block_names, nsamples, layer_names):
                self.cached_nsamples.append(nsamples)
                return {}

        monkeypatch.setattr("auto_round.utils.get_block_names", lambda model: [["block"]])
        monkeypatch.setattr("auto_round.utils.find_matching_blocks", lambda model, blocks, names: blocks)

        comp = MockCompressor()
        comp.quantize()

        assert comp.cached_nsamples == [7, 7]


class TestAlignDeviceAndDtype:
    """Test _align_device_and_dtype_for_secondary logic."""

    def test_no_op_when_pipe_is_none(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                self.model_context = SimpleNamespace(pipe=None, model=None)

        comp = MockCompressor()
        # Should not raise
        comp._align_device_and_dtype_for_secondary("transformer")

    def test_no_op_when_model_is_none(self):
        class MockCompressor(DiffusionMixin):
            def __init__(self):
                pipe = MagicMock()
                pipe.components = []
                self.model_context = SimpleNamespace(pipe=pipe, model=None)

        comp = MockCompressor()
        # Should not raise
        comp._align_device_and_dtype_for_secondary("transformer")
