"""Fast unit tests for algorithm registry and pipeline construction."""

from types import SimpleNamespace

import pytest
import torch

from auto_round import AWQConfig, OptimizedRTNConfig, RotationConfig, RTNConfig, SignRoundConfig, SpinQuantConfig
from auto_round.algorithms.config_resolver import (
    get_algorithm_class,
    resolve_shared_config_values,
    split_quantization_configs,
    sync_shared_config_from,
)
from auto_round.algorithms.pipeline import DiffusionBlockIO, QuantizationPipeline
from auto_round.algorithms.quantization import registry as _r
from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer
from auto_round.compressors.base import collect_user_scheme_overrides
import auto_round.compressors.calibrated_zero_shot as calibrated_zero_shot_module
from auto_round.compressors.calibrated_zero_shot import CalibratedZeroShotCompressor
from auto_round.compressors.data_driven import DataDrivenCompressor
from auto_round.compressors.diffusion_mixin import DiffusionMixin
from auto_round.compressors.entry import AutoRound as NewAutoRound
from auto_round.compressors.entry import _select_rtn_compressor_base_cls
import auto_round.compressors.zero_shot as zero_shot_module
from auto_round.compressors.zero_shot import ZeroShotCompressor
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme


class PartialSharedConfig(RTNConfig):
    def __init__(self, *, weight_clip_ratio=None, **kwargs):
        super().__init__(**kwargs)
        self.weight_clip_ratio = weight_clip_ratio


class NoWeightClipConfig(RTNConfig):
    pass


def test_split_awq_plus_rtn():
    pre, block = split_quantization_configs([AWQConfig(), RTNConfig()])
    assert len(pre) == 1 and type(pre[0]).__name__ == "AWQConfig"
    assert len(block) == 1 and type(block[0]).__name__ == "RTNConfig"


def test_pipeline_preprocessor_only_auto_appends_rtn():
    pipeline = QuantizationPipeline.from_configs([AWQConfig()])
    assert type(pipeline.preprocessors[0]).__name__ == "AWQTransform"
    assert isinstance(pipeline.block_quantizer, RTNQuantizer)


def test_svdquant_plus_rtn_pipeline_uses_svdquant_as_preprocessor():
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    pipeline = QuantizationPipeline.from_configs([SVDQuantConfig(rank=8), RTNConfig()])

    assert type(pipeline.preprocessors[0]).__name__ == "SVDQuantTransform"
    assert isinstance(pipeline.block_quantizer, RTNQuantizer)


def test_diffusion_block_io_preserves_single_sample_tensor_batch_dimension():
    io = DiffusionBlockIO(
        _fp_inputs={"hidden_states": [torch.zeros(1, 4, 8)]},
        _input_others={"positional_inputs": [], "temb": torch.zeros(1, 32)},
    )

    _, input_others = io._select_inputs(io._fp_inputs, io._input_others, torch.tensor([0]))

    assert input_others["temb"].shape == (1, 32)


def test_diffusion_reference_inputs_preserve_flux_dual_streams():
    class FluxBlock(torch.nn.Module):
        def forward(self, hidden_states, encoder_hidden_states):
            return encoder_hidden_states + 1, hidden_states + 2

    class Quantizer:
        batch_size = 1
        enable_quanted_input = True
        model_context = SimpleNamespace(amp=False, amp_dtype=torch.float32)
        compress_context = SimpleNamespace(cache_device="cpu", clear_memory=lambda: None)

        @staticmethod
        def _resolve_block_forward():
            def run(block, hidden_states, input_others, *_args):
                input_others = dict(input_others)
                input_others.pop("positional_inputs", None)
                return block(hidden_states, **input_others)

            return run

    io = DiffusionBlockIO(
        _fp_inputs={
            "encoder_hidden_states": [torch.tensor([[10.0]])],
            "hidden_states": [torch.tensor([[20.0]])],
        },
        _input_others={"positional_inputs": []},
        _quantizer=Quantizer(),
        _block=FluxBlock(),
        output_config=["encoder_hidden_states", "hidden_states"],
    )

    outputs = io.collect_reference_inputs()

    torch.testing.assert_close(outputs["encoder_hidden_states"][0], torch.tensor([[11.0]]))
    torch.testing.assert_close(outputs["hidden_states"][0], torch.tensor([[22.0]]))
    torch.testing.assert_close(io.get_reference_outputs(torch.tensor([0])), torch.tensor([[11.0, 22.0]]))

    predicted = io.forward_block_batch(torch.tensor([0]), device="cpu")

    torch.testing.assert_close(predicted, torch.tensor([[11.0, 22.0]]))

    next_inputs = io.collect_next_inputs()

    assert set(next_inputs) == {"encoder_hidden_states", "hidden_states"}
    torch.testing.assert_close(next_inputs["encoder_hidden_states"][0], torch.tensor([[11.0]]))
    torch.testing.assert_close(next_inputs["hidden_states"][0], torch.tensor([[22.0]]))


def test_pipeline_duplicate_preprocessor_rejected():
    with pytest.raises(ValueError, match="Duplicate preprocessor"):
        QuantizationPipeline.from_configs([AWQConfig(), AWQConfig()])


def test_pipeline_multiple_block_quantizers_rejected():
    with pytest.raises(ValueError, match="exactly one block-quantization config"):
        QuantizationPipeline.from_configs([RTNConfig(), SignRoundConfig()])


def test_registry_builtin_aliases_and_unknown():
    assert isinstance(_r.resolve_alg_config("RTN"), RTNConfig)
    assert isinstance(_r.resolve_alg_config("awq"), AWQConfig)
    assert isinstance(_r.resolve_alg_config("autoround"), SignRoundConfig)
    with pytest.raises(ValueError, match="Unknown algorithm alias"):
        _r.resolve_alg_config("definitely_not_registered_abc123")


def test_registry_resolves_variant_configs_to_registered_members():
    assert get_algorithm_class(OptimizedRTNConfig()) is not None
    assert get_algorithm_class(SignRoundConfig(enable_adam=True)).__name__ == "AdamRoundQuantizer"


def test_top_level_config_exports():
    from auto_round import AWQConfig as TopAWQConfig
    from auto_round import OptimizedRTNConfig as TopOptimizedRTNConfig
    from auto_round import RotationConfig as TopRotationConfig
    from auto_round import RTNConfig as TopRTNConfig
    from auto_round import SignRoundConfig as TopSignRoundConfig
    from auto_round import SpinQuantConfig as TopSpinQuantConfig

    assert TopAWQConfig is AWQConfig
    assert TopOptimizedRTNConfig is OptimizedRTNConfig
    assert TopRTNConfig is RTNConfig
    assert TopSignRoundConfig is SignRoundConfig
    assert TopRotationConfig is RotationConfig
    assert TopSpinQuantConfig is SpinQuantConfig


def test_new_entry_defaults_to_autoround_config(monkeypatch):
    captured = {}

    def _fake_init(self, config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(DataDrivenCompressor, "__init__", _fake_init)
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *args, **kwargs: "llm")

    NewAutoRound("dummy-model", "W4A16", iters=1, seqlen=8, nsamples=1)

    assert isinstance(captured["config"], SignRoundConfig)


def test_entry_rejects_configs_without_quantization_members():
    with pytest.raises(ValueError, match="At least one quantization algorithm config"):
        NewAutoRound("dummy-model", "W4A16", [RotationConfig()])


def test_compat_entry_preserves_spinquant_dict_config(monkeypatch):
    captured = {}
    rotation_config = {
        "algorithm": "spinquant",
        "r1": True,
        "r2": True,
        "r3": False,
        "r4": False,
        "rotation_size": 128,
        "trainable_rotation": False,
        "trainable_smooth": False,
    }

    def _fake_init(self, config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(DataDrivenCompressor, "__init__", _fake_init)
    monkeypatch.setattr("auto_round.utils.is_mllm_model", lambda *args, **kwargs: False)
    monkeypatch.setattr("auto_round.utils.is_diffusion_model", lambda *args, **kwargs: False)
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *args, **kwargs: "llm")

    from auto_round.autoround import AutoRound as CompatAutoRound

    CompatAutoRound(
        "dummy-model",
        scheme="W4A16",
        iters=1,
        seqlen=8,
        nsamples=1,
        rotation_config=rotation_config,
    )

    configs = captured["config"] if isinstance(captured["config"], list) else [captured["config"]]
    spinquant_cfg = next(cfg for cfg in configs if isinstance(cfg, SpinQuantConfig))
    assert spinquant_cfg.rotation_size == rotation_config["rotation_size"]
    assert spinquant_cfg.r1 is rotation_config["r1"]
    assert spinquant_cfg.r2 is rotation_config["r2"]
    assert spinquant_cfg.r3 is rotation_config["r3"]
    assert spinquant_cfg.r4 is rotation_config["r4"]
    assert spinquant_cfg.trainable_rotation is rotation_config["trainable_rotation"]
    assert spinquant_cfg.trainable_smooth is rotation_config["trainable_smooth"]


def test_compat_entry_forwards_disabled_svdquant_smoothing(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
    from auto_round.autoround import AutoRound as CompatAutoRound

    captured = {}

    def _fake_init(self, config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(DataDrivenCompressor, "__init__", _fake_init)
    monkeypatch.setattr("auto_round.utils.is_mllm_model", lambda *args, **kwargs: False)
    monkeypatch.setattr("auto_round.utils.is_diffusion_model", lambda *args, **kwargs: False)
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *args, **kwargs: "llm")

    CompatAutoRound(
        "dummy-model",
        scheme="W4A16",
        iters=1,
        seqlen=8,
        nsamples=1,
        enable_svdquant=True,
        svdquant_smooth_enabled=False,
        svdquant_smooth_max_calibration_calls=16,
    )

    configs = captured["config"] if isinstance(captured["config"], list) else [captured["config"]]
    svdquant_config = next(config for config in configs if isinstance(config, SVDQuantConfig))
    assert svdquant_config.smooth_enabled is False
    assert svdquant_config.smooth_max_calibration_calls == 16


def test_data_free_svdquant_rtn_routes_to_zero_shot(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    captured = {}

    def _fake_init(self, config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(ZeroShotCompressor, "__init__", _fake_init)
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *args, **kwargs: "llm")

    result = NewAutoRound(
        "dummy-model",
        "MXFP4",
        [SVDQuantConfig(smooth_enabled=False), RTNConfig(disable_opt_rtn=True)],
        format="svdquant_nunchaku",
    )

    assert isinstance(result, ZeroShotCompressor)
    assert captured["config"][0].requires_calibration is False


def test_zero_shot_block_context_moves_current_block_to_quantization_device(monkeypatch):
    calls = []

    class Block:
        def to(self, device):
            calls.append(("move", device))
            return self

    block = Block()
    compressor = ZeroShotCompressor.__new__(ZeroShotCompressor)
    compressor.model_context = SimpleNamespace(
        model=object(),
        amp_dtype=torch.bfloat16,
        is_mllm=False,
        is_diffusion=True,
    )
    compressor.quantizer = SimpleNamespace(create_block_io=lambda *args: object())
    monkeypatch.setattr(zero_shot_module.device_manager, "device", torch.device("cuda:0"))
    monkeypatch.setattr(
        zero_shot_module,
        "convert_module_to_hp_if_necessary",
        lambda current, dtype, device: calls.append(("convert", current, dtype, device)),
    )

    context = compressor._create_block_context(block, "transformer_blocks.0")

    assert calls == [
        ("convert", block, torch.bfloat16, "cuda:0"),
        ("move", "cuda:0"),
    ]
    assert context.block is block
    assert context.device == "cuda:0"


def test_calibrated_svdquant_rtn_routes_to_calibrated_zero_shot(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    captured = {}

    def _fake_init(self, config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(CalibratedZeroShotCompressor, "__init__", _fake_init)
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *args, **kwargs: "llm")

    result = NewAutoRound(
        "dummy-model",
        "MXFP4",
        [SVDQuantConfig(smooth_enabled=True), RTNConfig(disable_opt_rtn=True)],
        format="svdquant_nunchaku",
    )

    assert isinstance(result, CalibratedZeroShotCompressor)
    assert isinstance(result, ZeroShotCompressor)
    assert not isinstance(result, DataDrivenCompressor)
    assert captured["config"][0].requires_calibration is True


def test_calibrated_zero_shot_collects_reference_before_rtn_without_propagation():
    events = []

    class Preprocessor:
        def pre_quantize_block(self, ctx):
            events.append("transform")

        def post_quantize_block(self, ctx):
            events.append("cleanup")

    class Quantizer:
        def quantize_block(self, ctx):
            events.append("rtn")

    class Pipeline:
        preprocessors = [Preprocessor()]

        def enter_preprocessor_hooks(self, ctx, stack):
            events.append("hooks")

    class Context:
        def collect_reference_inputs(self, stack):
            events.append("reference")
            return [torch.zeros(1)]

        def collect_next_inputs(self):
            raise AssertionError("calibrated zero-shot must not propagate quantized block outputs")

    compressor = CalibratedZeroShotCompressor.__new__(CalibratedZeroShotCompressor)
    compressor._pipeline = Pipeline()
    compressor.quantizer = Quantizer()
    compressor._current_input_others = {}

    compressor._run_block_pipeline(Context())

    assert events == ["hooks", "reference", "transform", "rtn", "cleanup"]


def test_calibrated_zero_shot_caches_only_each_block_group_entry():
    blocks = [["blocks.0", "blocks.1"], ["single_blocks.0", "single_blocks.1"]]

    assert CalibratedZeroShotCompressor.get_calibration_block_names(blocks) == [
        "blocks.0",
        "single_blocks.0",
    ]


def test_calibrated_zero_shot_reuses_deterministic_selection_for_each_block_group():
    compressor = CalibratedZeroShotCompressor.__new__(CalibratedZeroShotCompressor)
    compressor._smooth_max_calibration_calls = 3
    compressor.prepare_calibration_call_selection(total_calls=10)

    first_group = [index for index in range(10) if compressor.should_cache_calibration_call("blocks.0")]
    second_group = [index for index in range(10) if compressor.should_cache_calibration_call("single_blocks.0")]

    assert first_group == [0, 4, 9]
    assert second_group == first_group


def test_calibrated_zero_shot_resets_call_counters_for_new_collection():
    compressor = CalibratedZeroShotCompressor.__new__(CalibratedZeroShotCompressor)
    compressor._smooth_max_calibration_calls = 2
    compressor.prepare_calibration_call_selection(total_calls=4)
    first = [index for index in range(4) if compressor.should_cache_calibration_call("blocks.0")]

    compressor.prepare_calibration_call_selection(total_calls=4)
    second = [index for index in range(4) if compressor.should_cache_calibration_call("blocks.0")]

    assert first == [0, 3]
    assert second == first


def test_calibrated_zero_shot_logs_svdquant_provenance_once(monkeypatch):
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    messages = []
    monkeypatch.setattr(logger, "info", lambda message, *args: messages.append(message % args))
    monkeypatch.setattr(calibrated_zero_shot_module, "_source_checkout_commit", lambda: "abc123")
    compressor = CalibratedZeroShotCompressor.__new__(CalibratedZeroShotCompressor)
    compressor._provenance_configs = [
        SVDQuantConfig(
            rank=32,
            smooth_enabled=True,
            smooth_num_grids=20,
            smooth_max_calibration_calls=16,
            residual_iters=100,
            residual_early_stop=True,
        )
    ]
    compressor._provenance_model_source = "/models/flux"
    compressor._calibration_state = SimpleNamespace(dataset="/data/coco2017.tsv", nsamples=128)
    compressor.num_inference_steps = 50

    compressor._log_svdquant_provenance()
    compressor._log_svdquant_provenance()

    assert len(messages) == 1
    assert "model=/models/flux" in messages[0]
    assert "dataset=/data/coco2017.tsv" in messages[0]
    assert "nsamples=128" in messages[0]
    assert "diffusion_steps=50" in messages[0]
    assert "smooth_max_calibration_calls=16" in messages[0]
    assert "smooth_num_grids=20" in messages[0]
    assert "rank=32" in messages[0]
    assert "residual_iters=100" in messages[0]
    assert "residual_early_stop=True" in messages[0]
    assert "source_commit=abc123" in messages[0]


def test_source_checkout_commit_falls_back_to_unknown(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(calibrated_zero_shot_module.subprocess, "run", _raise)

    assert calibrated_zero_shot_module._source_checkout_commit() == "unknown"


def test_diffusion_prepares_call_selection_from_prompt_batches_and_steps():
    compressor = DiffusionMixin.__new__(DiffusionMixin)
    compressor.nsamples = 10
    compressor.batch_size = 4
    compressor.num_inference_steps = 20
    prepared = []
    compressor.prepare_calibration_call_selection = prepared.append

    compressor._prepare_calibration_call_selection()

    assert prepared == [60]


def test_entry_warns_and_drops_unsupported_kwargs(monkeypatch, tiny_opt_model_path):
    calls = []

    def _record_warning(message, *args):
        calls.append(message % args)

    monkeypatch.setattr(logger, "warning_once", _record_warning)

    NewAutoRound(
        tiny_opt_model_path,
        "W4A16",
        RTNConfig(disable_opt_rtn=True),
        nsamples=1,
        seqlen=8,
        low_cpu_mem_usage=False,
        nonsense_kwarg=123,
    )

    assert any("unsupported kwargs nonsense_kwarg" in msg for msg in calls)


def test_shared_config_values_inherit_across_matching_attrs_only():
    awq = PartialSharedConfig(weight_clip_ratio=0.9)
    smoothquant_like = NoWeightClipConfig()
    signround = PartialSharedConfig(weight_clip_ratio=None)

    resolve_shared_config_values([awq, smoothquant_like, signround])

    assert signround.weight_clip_ratio == 0.9
    assert not hasattr(smoothquant_like, "weight_clip_ratio")


def test_shared_config_values_reject_conflicts():
    with pytest.raises(ValueError, match="Conflicting shared config field 'weight_clip_ratio'"):
        resolve_shared_config_values(
            [PartialSharedConfig(weight_clip_ratio=0.8), PartialSharedConfig(weight_clip_ratio=0.9)]
        )


def test_shared_config_sync_from_source_skips_missing_attrs():
    source = PartialSharedConfig(weight_clip_ratio=0.75)
    target = PartialSharedConfig()
    no_clip_target = NoWeightClipConfig()

    sync_shared_config_from(source, [target, no_clip_target, RotationConfig()])

    assert target.weight_clip_ratio == 0.75
    assert not hasattr(no_clip_target, "weight_clip_ratio")


def test_user_scheme_overrides_merge_across_all_configs():
    awq = AWQConfig(bits=8)
    rtn = RTNConfig()
    assert collect_user_scheme_overrides([awq, rtn])["bits"] == 8

    resolve_shared_config_values([awq, rtn])

    assert rtn.bits == 8


def test_user_scheme_overrides_reject_explicit_conflicts():
    with pytest.raises(ValueError, match="Conflicting shared scheme field 'bits'"):
        collect_user_scheme_overrides([AWQConfig(bits=8), RTNConfig(bits=4)])
    with pytest.raises(ValueError, match="Conflicting shared scheme field 'bits'"):
        resolve_shared_config_values([AWQConfig(bits=8), RTNConfig(bits=4)])


# ===========================================================================
#  Scheme-dependent config heuristics must see resolved values, not just
#  whatever (often None) bits/lr the config was constructed with directly.
# ===========================================================================


@pytest.mark.parametrize(
    "scheme, expect_disable_opt_rtn",
    [
        ("W8A16", True),
        # "INT8" (bits=8, act_bits=8, data_type=int) is W8A8-equivalent but was
        # previously missed because routing only matched the literal strings
        # "W8A16"/"W8A8", not schemes reaching the same resolved values.
        ("INT8", True),
        ("W4A16", False),
        ({"bits": 8, "act_bits": 8, "data_type": "int", "sym": True}, True),
    ],
)
def test_rtn_routing_disable_opt_rtn_from_resolved_scheme(scheme, expect_disable_opt_rtn):
    config = RTNConfig()
    _select_rtn_compressor_base_cls(config, scheme, "auto_round", {})
    assert config.disable_opt_rtn is expect_disable_opt_rtn


def test_rtn_routing_respects_explicit_enable_opt_rtn():
    """An explicit user choice must not be clobbered by the W8A16/W8A8 heuristic."""
    config = RTNConfig(enable_opt_rtn=True)
    _select_rtn_compressor_base_cls(config, "W8A16", "auto_round", {})
    assert config.disable_opt_rtn is False


@pytest.mark.parametrize("bits, expected_lr", [(3, 2.0 / 1000), (4, 1.0 / 1000)])
def test_sign_round_finalize_scheme_lr_heuristic(bits, expected_lr):
    """The low-bit lr bump must apply once `bits` is resolved via the scheme,
    even though it was unset (None) at construction time (e.g. `scheme=` alone,
    no explicit `bits=`)."""
    config = SignRoundConfig(iters=1000)
    config.scheme = QuantizationScheme(bits=bits, act_bits=16, data_type="int")
    config.finalize_scheme()
    assert config.lr == expected_lr


def test_sign_round_finalize_scheme_respects_explicit_lr():
    config = SignRoundConfig(iters=1000, lr=0.01, minmax_lr=0.05)
    config.scheme = QuantizationScheme(bits=2, act_bits=16, data_type="int")
    config.finalize_scheme()
    assert config.lr == 0.01
    assert config.minmax_lr == 0.05
