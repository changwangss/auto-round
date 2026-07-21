# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
from contextlib import ExitStack
from pathlib import Path
import subprocess
from typing import Any, Callable, Optional, Union

import torch

from auto_round.algorithms.pipeline import BlockContext
from auto_round.calibration.sampling import uniform_call_indices
from auto_round.compressors.zero_shot import ZeroShotCompressor
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_
from auto_round.utils import convert_module_to_hp_if_necessary, get_block_names
from auto_round.utils.device_manager import device_manager
from auto_round.version import __version__


def _source_checkout_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _describe_provenance_value(value: Any) -> str:
    if isinstance(value, (str, Path)):
        return str(value)
    return type(value).__name__


class CalibratedZeroShotCompressor(ZeroShotCompressor):
    """Calibrate weight preprocessors, then run terminal zero-shot quantization.

    Calibration data is used only by preprocessors such as SVDQuant smoothing.
    The terminal RTN pass remains block-local and never propagates quantized
    outputs between blocks.
    """

    need_calib: bool = True

    def __init__(
        self,
        config: Union[object, list[object]],
        model: Union[torch.nn.Module, str],
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        **kwargs,
    ) -> None:
        configs = config if isinstance(config, list) else [config]
        self._provenance_model_source = _describe_provenance_value(model)
        self._provenance_configs = configs
        self._smooth_max_calibration_calls = next(
            (
                candidate.smooth_max_calibration_calls
                for candidate in configs
                if hasattr(candidate, "smooth_max_calibration_calls")
            ),
            128,
        )
        kwargs["iters"] = 0
        super().__init__(config=config, model=model, **kwargs)
        self.dataset = dataset

    def _log_svdquant_provenance(self) -> None:
        if getattr(self, "_svdquant_provenance_logged", False):
            return
        config = next((candidate for candidate in self._provenance_configs if hasattr(candidate, "rank")), None)
        if config is None:
            return
        logger.info(
            "SVDQuant run provenance: model=%s dataset=%s nsamples=%s diffusion_steps=%s "
            "smooth_max_calibration_calls=%s smooth_num_grids=%s rank=%s residual_iters=%s "
            "residual_early_stop=%s residual_quant_method=%s autoround_version=%s source_commit=%s",
            self._provenance_model_source,
            _describe_provenance_value(self.dataset),
            getattr(self, "nsamples", "unknown"),
            getattr(self, "num_inference_steps", "n/a"),
            getattr(config, "smooth_max_calibration_calls", "n/a"),
            getattr(config, "smooth_num_grids", "n/a"),
            getattr(config, "rank", "n/a"),
            getattr(config, "residual_iters", "n/a"),
            getattr(config, "residual_early_stop", "n/a"),
            getattr(config, "residual_quant_method", "n/a"),
            __version__,
            _source_checkout_commit(),
        )
        self._svdquant_provenance_logged = True

    def prepare_calibration_call_selection(self, total_calls: int) -> None:
        """Prepare one deterministic call sample shared by every block group."""
        selected = uniform_call_indices(total_calls, self._smooth_max_calibration_calls)
        self._selected_calibration_call_indices = frozenset(selected)
        self._calibration_call_counts: dict[str, int] = {}
        logger.info(
            "Retaining %d of %d diffusion transformer calls for SVDQuant calibration",
            len(selected),
            total_calls,
        )

    def should_cache_calibration_call(self, name: str) -> bool:
        """Return whether this block invocation belongs to the shared call sample."""
        selected = getattr(self, "_selected_calibration_call_indices", None)
        if selected is None:
            return True
        call_index = self._calibration_call_counts.get(name, 0)
        self._calibration_call_counts[name] = call_index + 1
        return call_index in selected

    def post_init(self) -> None:
        if self._post_init_done:
            return
        super().post_init()
        if self.calibration is None:
            from auto_round.calibration import get_calibrator

            self.calibration = get_calibrator(self._get_calibrator_kind())(self)

    def _get_calibrator_kind(self) -> str:
        return "llm"

    @torch.no_grad()
    def _get_block_forward_func(self, name: str) -> Callable:
        from auto_round.calibration.hooks import make_block_forward_func

        forward = make_block_forward_func(self, name)
        if self.calibration is not None:
            forward = self.calibration.wrap_block_forward(forward)
        return forward

    @torch.no_grad()
    def _get_cache_data_hook_for_layer(self, name: str) -> Callable:
        from auto_round.calibration.hooks import make_layer_cache_hook

        return make_layer_cache_hook(self, name)

    def _replace_forward(self) -> None:
        from auto_round.calibration.hooks import replace_forward_with_hooks

        replace_forward_with_hooks(self)

    def _should_stop_cache_forward(self, name: str) -> bool:
        if self.calibration is not None:
            return self.calibration.should_stop(name)
        from auto_round.calibration.hooks import should_stop_cache_forward

        return should_stop_cache_forward(self, name)

    @torch.no_grad()
    def try_cache_inter_data_gpucpu(
        self,
        block_names: list,
        nsamples: int,
        layer_names: Optional[list] = None,
        last_cache_name: Optional[str] = None,
    ) -> Any:
        if self.calibration is None:
            self.post_init()
        return self.calibration.collect(block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name)

    @torch.no_grad()
    def cache_inter_data(
        self,
        block_names: list,
        nsamples: int,
        layer_names: Optional[list] = None,
        last_cache_name: Optional[str] = None,
    ) -> Any:
        if self.calibration is None:
            self.post_init()
        return self.calibration.cache_inter_data(
            block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
        )

    @torch.no_grad()
    def calib(self, nsamples: int, bs: int) -> Any:
        if self.calibration is None:
            self.post_init()
        return self.calibration.calib(nsamples, bs)

    def _ensure_calibration_inputs(self) -> None:
        all_blocks = self.quant_block_list or get_block_names(self.model_context.model)
        if not all_blocks:
            return
        block_names = self.get_calibration_block_names(all_blocks)
        if getattr(self, "_inputs_cached", False) and all(name in self.inputs for name in block_names):
            return
        self.inputs = self.try_cache_inter_data_gpucpu(block_names, self.nsamples, layer_names=[])
        self._inputs_cached = True

    @staticmethod
    def get_calibration_block_names(all_blocks: list[list[str]]) -> list[str]:
        """Cache one input per block group; FP outputs feed the remaining blocks."""
        return [group[0] for group in all_blocks if group]

    def _create_block_context(self, block: torch.nn.Module, block_name: str) -> BlockContext:
        from auto_round.calibration.inputs import preprocess_block_inputs

        if block_name in self.inputs:
            cached_inputs = self.inputs.pop(block_name)
            input_ids, input_others = preprocess_block_inputs(
                cached_inputs,
                model_context=self.model_context,
                compress_context=self.compress_context,
                first_input_name="input_ids",
            )
        else:
            if not hasattr(self, "_next_calibration_inputs"):
                raise ValueError(f"No calibration input is available for block {block_name!r}.")
            input_ids, input_others = self._next_calibration_inputs
            del self._next_calibration_inputs
        self._current_input_others = input_others
        materialize_model_(block)
        convert_module_to_hp_if_necessary(block, self.model_context.amp_dtype, device_manager.device)
        block.to(device_manager.device)
        return BlockContext(
            model=self.model_context.model,
            block=block,
            block_names=[block_name],
            block_name=block_name,
            block_index=0,
            io=self.quantizer.create_block_io(input_ids, input_others, None, block),
            bs=self.quantizer.batch_size * self.quantizer.infer_bs_coeff,
            loss_device=device_manager.device,
            device=device_manager.device,
            is_mllm=self.model_context.is_mllm,
            is_diffusion=self.model_context.is_diffusion,
        )

    def _run_block_pipeline(self, ctx: BlockContext) -> None:
        with ExitStack() as forward_stack:
            self.pipeline.enter_preprocessor_hooks(ctx, forward_stack)
            reference_inputs = ctx.collect_reference_inputs(forward_stack)
        self._next_calibration_inputs = (reference_inputs, self._current_input_others)
        super()._run_block_pipeline(ctx)

    @torch.no_grad()
    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        self._log_svdquant_provenance()
        self.post_init()
        self._ensure_calibration_inputs()
        for algorithm in self.pipeline.all():
            algorithm.prepare_run(self)
        try:
            return super().quantize()
        finally:
            for algorithm in self.pipeline.all():
                algorithm.finalize_run(self)
