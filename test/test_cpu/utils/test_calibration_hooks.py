# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from auto_round.calibration.hooks import replace_forward_with_hooks
from auto_round.context.model import ModelContext


class _AccelerateManagedBlock(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self._hf_hook = object()
        self._old_forward = self._inner_forward

        def accelerate_forward(value):
            self.events.append("accelerate_pre_forward")
            return self._old_forward(value)

        self.forward = accelerate_forward

    def _inner_forward(self, value):
        self.events.append("original_forward")
        return value + 1


class _Model(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block


class _OrdinaryBlock(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

        def original_forward(value):
            self.events.append("original_forward")
            return value + 1

        self.forward = original_forward


class _CalibrationState:
    def __init__(self, model_context, events):
        self.model_context = model_context
        self.events = events
        self.to_cached_layers = ["block"]

    def _get_block_forward_func(self, _name):
        def capture(module, value):
            self.events.append("autoround_capture")
            return module.orig_forward(value)

        return capture

    def _get_cache_data_hook_for_layer(self, _name):
        raise AssertionError("the test target is a block, not a layer")


def _new_model_context(model):
    context = ModelContext.__new__(ModelContext)
    context.model = model
    context._init_model = True
    context._has_true_orig_forward_set = False
    context.hook_handles = []
    return context


def test_accelerate_forward_wrapper_runs_before_calibration_capture_and_survives_recovery():
    events = []
    block = _AccelerateManagedBlock(events)
    outer_forward = block.forward
    inner_forward = block._old_forward
    context = _new_model_context(_Model(block))
    state = _CalibrationState(context, events)

    replace_forward_with_hooks(state)

    assert block.forward is outer_forward
    assert block(torch.tensor(1)).item() == 2
    assert events == ["accelerate_pre_forward", "autoround_capture", "original_forward"]

    context.recover_forward(restore_positional_wrapper=False)

    assert block.forward is outer_forward
    assert block._old_forward == inner_forward
    assert not hasattr(block, "orig_forward")


def test_accelerate_forward_wrapper_is_recovered_after_calibration_error():
    events = []
    block = _AccelerateManagedBlock(events)
    outer_forward = block.forward
    inner_forward = block._old_forward
    context = _new_model_context(_Model(block))

    replace_forward_with_hooks(_CalibrationState(context, events))
    try:
        raise RuntimeError("calibration failed")
    except RuntimeError:
        context.recover_forward(restore_positional_wrapper=False)

    assert block.forward is outer_forward
    assert block._old_forward == inner_forward
    assert not hasattr(block, "_autoround_replaced_forward_attr")


def test_ordinary_block_keeps_direct_forward_replacement_behavior():
    events = []
    block = _OrdinaryBlock(events)
    original_forward = block.forward
    context = _new_model_context(_Model(block))

    replace_forward_with_hooks(_CalibrationState(context, events))

    assert block.forward is not original_forward
    assert block(torch.tensor(1)).item() == 2
    assert events == ["autoround_capture", "original_forward"]

    context.recover_forward(restore_positional_wrapper=False)

    assert block.forward is original_forward
    assert not hasattr(block, "_autoround_replaced_forward_attr")
