import json

import pytest
import torch
from safetensors.torch import load_file

from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.export.svdquant_adapters.wan import WAN_SVDQUANT_TARGET_MODULES, WanSVDQuantNunchakuAdapter
from auto_round.export.svdquant_mxfp4 import NunchakuMXFP4Packer, unpack_nibbles
from auto_round.export.svdquant_nunchaku import SVDQuantExportConfig, save_svdquant_nunchaku_safetensors
from auto_round.export.svdquant_omni import (
    collect_svdquant_omni_tensors,
    convert_wan_nunchaku_to_omni,
    save_svdquant_omni,
)


def tiny_wan(num_layers=1):
    model = torch.nn.Module()
    model.config = dict(
        _class_name="WanTransformer3DModel",
        num_layers=num_layers,
        num_attention_heads=2,
        attention_head_dim=32,
        ffn_dim=96,
    )
    model._keep_in_fp32_modules = ["scale_shift_table"]
    model.register_buffer("scale_shift_table", torch.tensor([1.001]))
    for i in range(num_layers):
        for path in WAN_SVDQUANT_TARGET_MODULES:
            n, k = (96, 64) if path == "ffn.net.0.proj" else (64, 96) if path == "ffn.net.2" else (64, 64)
            linear = torch.nn.Linear(k, n)
            for name, value in dict(
                data_type="mx_fp",
                bits=4,
                group_size=32,
                sym=True,
                act_data_type="mx_fp",
                act_bits=4,
                act_group_size=32,
                act_sym=True,
                act_dynamic=True,
            ).items():
                setattr(linear, name, value)
            wrapper = SVDQuantLinear(
                linear,
                torch.nn.Linear(k, 16, bias=False),
                torch.nn.Linear(16, n, bias=False),
                torch.linspace(0.7, 1.9, k),
            )
            parent = model
            parts = f"blocks.{i}.{path}".split(".")
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            parent.add_module(parts[-1], wrapper)
    return model


def test_logical_codes_and_smooth_contract():
    model = tiny_wan()
    tensors, config = collect_svdquant_omni_tensors(model)
    assert config["fuse_qkv"] is False
    assert config["rank"] == 16
    assert config["precision"] == "mxfp4"
    assert sum(key.endswith(".qweight") for key in tensors) == 10
    assert tensors["scale_shift_table"].dtype == torch.float32
    torch.testing.assert_close(tensors["scale_shift_table"], model.scale_shift_table, rtol=0, atol=0)
    packer = NunchakuMXFP4Packer()
    for path in WAN_SVDQUANT_TARGET_MODULES:
        prefix = "blocks.0." + path
        module = model.get_submodule(prefix)
        n, k = module.residual_linear.weight.shape
        packed = packer.pack_residual(module.residual_linear.weight.detach())
        assert torch.equal(
            unpack_nibbles(tensors[prefix + ".qweight"].view(torch.uint8)),
            packer._unpack_weight_codes(packed.qweight)[:n, :k],
        )
        assert torch.equal(tensors[prefix + ".wscales"], packer._unpack_scale_codes(packed.wscales)[:n, : k // 32].T)
        expected = (module.lora_down.weight.double() * module.smooth.double()).T.bfloat16()
        assert torch.equal(tensors[prefix + ".proj_down"], expected)
        assert torch.equal(tensors[prefix + ".smooth_factor"], module.smooth.double().reciprocal().bfloat16())
        x = torch.randn(3, k, dtype=torch.float64)
        exact = (x * module.smooth.double()) @ module.lora_down.weight.double().T @ module.lora_up.weight.double().T
        actual = x @ tensors[prefix + ".proj_down"].double() @ tensors[prefix + ".proj_up"].double().T
        assert torch.linalg.vector_norm(actual - exact) / torch.linalg.vector_norm(exact) < 0.012


def test_complete_wan_rejects_missing_projection():
    model = tiny_wan()
    del model.blocks._modules["0"].attn1.to_q
    with pytest.raises(ValueError, match="complete Wan projection mismatch"):
        collect_svdquant_omni_tensors(model)


def test_invalid_bf16_smoothing_rejected():
    model = tiny_wan()
    model.get_submodule("blocks.0.attn1.to_q").smooth.fill_(1e-40)
    with pytest.raises(ValueError, match="positive finite"):
        collect_svdquant_omni_tensors(model)


def test_converter_matches_direct_export_and_keeps_source(tmp_path):
    model = tiny_wan()
    checkpoint = tmp_path / "source.safetensors"
    save_svdquant_nunchaku_safetensors(
        model, str(checkpoint), config=SVDQuantExportConfig(runtime_loadable=True), adapter=WanSVDQuantNunchakuAdapter()
    )
    original = checkpoint.read_bytes()
    save_svdquant_omni(model, tmp_path / "direct")
    convert_wan_nunchaku_to_omni(checkpoint, tmp_path / "converted")
    direct = load_file(tmp_path / "direct/diffusion_pytorch_model.safetensors")
    converted = load_file(tmp_path / "converted/diffusion_pytorch_model.safetensors")
    assert direct.keys() == converted.keys()
    for key in direct:
        assert torch.equal(direct[key], converted[key]), key
    assert checkpoint.read_bytes() == original
    config = json.loads((tmp_path / "converted/config.json").read_text())
    assert config["quantization_config"]["fuse_qkv"] is False
    with pytest.raises(FileExistsError):
        convert_wan_nunchaku_to_omni(checkpoint, tmp_path / "converted")


def test_converter_two_experts_and_configs_do_not_share_source_storage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    index = dict(
        _class_name="WanPipeline",
        boundary_ratio=0.875,
        transformer=["nunchaku", "NunchakuWanTransformer3DModel"],
        transformer_2=["nunchaku", "NunchakuWanTransformer3DModel"],
    )
    (source / "model_index.json").write_text(json.dumps(index))
    (source / "scheduler").mkdir()
    (source / "scheduler/scheduler_config.json").write_text('{"test": 1}')
    for expert in ("transformer", "transformer_2"):
        model = tiny_wan()
        save_svdquant_nunchaku_safetensors(
            model,
            str(source / expert / "diffusion_pytorch_model.safetensors"),
            config=SVDQuantExportConfig(runtime_loadable=True),
            adapter=WanSVDQuantNunchakuAdapter(),
        )
    output = tmp_path / "output"
    convert_wan_nunchaku_to_omni(source, output)
    saved_index = json.loads((output / "model_index.json").read_text())
    for expert in ("transformer", "transformer_2"):
        assert saved_index[expert] == ["diffusers", "WanTransformer3DModel"]
        config = json.loads((output / expert / "config.json").read_text())
        quantization = json.loads((output / expert / "quantization_config.json").read_text())
        assert config["quantization_config"] == quantization
        tensors = load_file(output / expert / "diffusion_pytorch_model.safetensors")
        assert sum(key.endswith(".qweight") for key in tensors) == 10
    copied = output / "scheduler/scheduler_config.json"
    assert copied.stat().st_ino != (source / "scheduler/scheduler_config.json").stat().st_ino
    copied.write_text("{}")
    assert json.loads((source / "scheduler/scheduler_config.json").read_text()) == {"test": 1}
    assert json.loads((source / "model_index.json").read_text()) == index


def test_all_400_a14b_projection_paths():
    tensors, _ = collect_svdquant_omni_tensors(tiny_wan(num_layers=40))
    assert {key[:-8] for key in tensors if key.endswith(".qweight")} == {
        f"blocks.{i}.{path}" for i in range(40) for path in WAN_SVDQUANT_TARGET_MODULES
    }


@pytest.mark.parametrize(
    "requested,expected",
    [(None, "svdquant_nunchaku"), ("svdquant_nunchaku", "svdquant_nunchaku"), ("svdquant_omni", "svdquant_omni")],
)
def test_factory_respects_explicit_omni_format(monkeypatch, requested, expected):
    import importlib

    from auto_round import AutoRound
    from auto_round.algorithms.transforms.svdquant import SVDQuantConfig

    entry = importlib.import_module("auto_round.autoround")
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda model: "diffusion")
    monkeypatch.setattr(entry, "_get_compressor_class", lambda *args: lambda configs, **kwargs: kwargs)
    kwargs = {} if requested is None else {"format": requested}
    result = AutoRound(tiny_wan(), scheme="MXFP4", alg_configs=[SVDQuantConfig(rank=16), "rtn"], **kwargs)
    assert result["format"] == expected


def test_format_registration_and_residual_scheme_validation():
    from types import SimpleNamespace

    from auto_round.formats import get_formats
    from auto_round.schemes import PRESET_SCHEMES

    scheme = PRESET_SCHEMES["MXFP4"].copy()
    context = SimpleNamespace(**scheme.to_dict(), scheme="MXFP4")
    output_format = get_formats("svdquant_omni", context)[0]
    assert output_format.format_name == "svdquant_omni"
    assert not output_format.is_supported_immediate_packing()
    model = tiny_wan()
    with pytest.raises(ValueError, match="incompatible residual scheme"):
        output_format._validate_svd_layer_overrides(model, {"blocks.0.attn1.to_q": {"group_size": 64}})


def test_converter_rejects_pipeline_missing_declared_expert(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model_index.json").write_text(
        json.dumps(
            dict(
                transformer=["nunchaku", "NunchakuWanTransformer3DModel"],
                transformer_2=["nunchaku", "NunchakuWanTransformer3DModel"],
            )
        )
    )
    (source / "transformer").mkdir()
    with pytest.raises(FileNotFoundError, match="missing component"):
        convert_wan_nunchaku_to_omni(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("explicit_list", [False, True])
def test_save_explicit_format_replaces_initialized_nunchaku_format(tmp_path, monkeypatch, pipeline, explicit_list):
    from types import SimpleNamespace

    from safetensors import safe_open

    from auto_round.compressors.base import BaseOrchestrator
    from auto_round.compressors.diffusion_mixin import DiffusionMixin
    from auto_round.formats import get_formats

    class SaveHarness:
        save_quantized = BaseOrchestrator.save_quantized
        scheme = "MXFP4"
        act_bits = 4
        layer_config = {}
        quant_block_list = None

        @property
        def model(self):
            return self.model_context.model

        def _resolve_format_string(self, value):
            return get_formats(value, self)

    class PipelineHarness(DiffusionMixin, SaveHarness):
        pass

    compressor = PipelineHarness.__new__(PipelineHarness) if pipeline else SaveHarness()
    model = tiny_wan()
    second = tiny_wan()
    pipe = SimpleNamespace(
        transformer=model,
        transformer_2=second,
        components={"transformer": model, "transformer_2": second},
        config={
            "_class_name": "WanPipeline",
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "transformer_2": ["diffusers", "WanTransformer3DModel"],
        },
    )
    compressor.model_context = SimpleNamespace(model=model, quantized=True, tokenizer=None, pipe=pipe)
    compressor.formats = compressor._resolve_format_string("svdquant_nunchaku")
    compressor.compress_context = SimpleNamespace(is_immediate_saving=False, formats=compressor.formats)
    if pipeline:
        compressor._quantized_transformers = {"transformer_2": (second, {})}
    # Folder selection normally reads process-wide contexts; keep this fixture isolated.
    monkeypatch.setattr("auto_round.compressors.base._get_save_folder_name", lambda fmt: compressor.output_dir)
    selected = compressor._resolve_format_string("svdquant_omni") if explicit_list else "svdquant_omni"
    compressor.save_quantized(str(tmp_path), format=selected)
    assert compressor.formats[0].format_name == "svdquant_omni"
    assert compressor.compress_context.formats[0].format_name == "svdquant_omni"
    components = [tmp_path / name for name in ("transformer", "transformer_2")] if pipeline else [tmp_path]
    for component in components:
        with safe_open(str(component / "diffusion_pytorch_model.safetensors"), framework="pt") as checkpoint:
            assert "blocks.0.attn1.to_q.proj_down" in checkpoint.keys()
            assert "blocks.0.attn1.to_q.lora_down" not in checkpoint.keys()
            metadata = json.loads(checkpoint.metadata()["quantization_config"])
            assert metadata["quant_method"] == "svdquant"
        assert json.loads((component / "config.json").read_text())["quantization_config"] == metadata
    if pipeline:
        index = json.loads((tmp_path / "model_index.json").read_text())
        assert index["transformer"] == index["transformer_2"] == ["diffusers", "WanTransformer3DModel"]
