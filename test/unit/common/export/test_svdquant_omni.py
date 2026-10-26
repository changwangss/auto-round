# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coverage for the Omni-specific tensor layout, independent of model architecture."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.data_type.nvfp import nv_fp4
from auto_round.export.svdquant_nunchaku import IdentitySVDQuantModelAdapter
from auto_round.export.svdquant_omni import save_svdquant_omni
from auto_round.formats import get_formats
from auto_round.schemes import PRESET_SCHEMES


def test_omni_tensor_layout_and_adapter_mapping(tmp_path):
    model = torch.nn.Module()
    model.config = {"_name_or_path": tmp_path}
    residual = torch.nn.Linear(32, 16, dtype=torch.bfloat16)
    for name, value in PRESET_SCHEMES["NVFP4"].to_dict().items():
        setattr(residual, name, value)
    down = torch.nn.Linear(32, 8, bias=False, dtype=torch.bfloat16)
    up = torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
    model.projection = SVDQuantLinear(residual, down, up, torch.full((32,), 2, dtype=torch.bfloat16))
    model.unquantized = torch.nn.Linear(32, 16)

    class RenameAdapter(IdentitySVDQuantModelAdapter):
        def map_modules(self, model, records):
            return [replace(record, prefix="runtime.projection") for record in super().map_modules(model, records)]

    for adapter, prefix in ((None, "projection"), (RenameAdapter(), "runtime.projection")):
        if adapter is not None:
            # Also exercise export of tuned residuals through the same ABI.
            residual.data_type, residual.bits, residual.group_size = "nv_fp4", 4, 16
            residual.weight_global_scale = torch.tensor([4.0])
            quantized, residual.scale, _ = nv_fp4(residual.weight.float(), global_scale=residual.weight_global_scale)
            residual.weight.data = quantized.bfloat16()
        output = tmp_path / prefix
        output_format = get_formats("svdquant_omni", SimpleNamespace(scheme="NVFP4"))[0]
        assert output_format.save_quantized(output, model=model, adapter=adapter) is model
        tensors = load_file(output / "diffusion_pytorch_model.safetensors")
        assert set(tensors) == {
            *(
                f"{prefix}.{suffix}"
                for suffix in (
                    "qweight",
                    "wscales",
                    "wtscale",
                    "wcscales",
                    "proj_down",
                    "proj_up",
                    "smooth_factor",
                    "bias",
                )
            ),
            "unquantized.weight",
            "unquantized.bias",
        }
        assert tensors[f"{prefix}.qweight"].shape == (16, 16)
        assert tensors[f"{prefix}.qweight"].dtype == torch.int8
        assert tensors[f"{prefix}.wscales"].shape == (2, 16)
        assert tensors[f"{prefix}.wscales"].dtype == torch.float8_e4m3fn
        assert tensors[f"{prefix}.wtscale"].shape == (1,)
        assert tensors[f"{prefix}.wtscale"].dtype == torch.bfloat16
        if adapter is not None:
            torch.testing.assert_close(tensors[f"{prefix}.wtscale"], torch.tensor([0.25], dtype=torch.bfloat16))
            torch.testing.assert_close(tensors[f"{prefix}.wscales"].float(), residual.scale.reshape(16, 2).T)
        torch.testing.assert_close(tensors[f"{prefix}.wcscales"], torch.ones(16, dtype=torch.bfloat16))
        torch.testing.assert_close(tensors[f"{prefix}.proj_down"], (down.weight.float() * 2).T.bfloat16())
        torch.testing.assert_close(tensors[f"{prefix}.proj_up"], up.weight)
        torch.testing.assert_close(tensors[f"{prefix}.smooth_factor"], torch.full((32,), 0.5, dtype=torch.bfloat16))
        torch.testing.assert_close(tensors[f"{prefix}.bias"], residual.bias)
        torch.testing.assert_close(tensors["unquantized.weight"], model.unquantized.weight)
        config = json.loads((output / "config.json").read_text())
        quantization = json.loads((output / "quantization_config.json").read_text())
        assert config["quantization_config"] == quantization
        assert quantization == {
            "quant_method": "svdquant",
            "precision": "nvfp4",
            "rank": 8,
            "act_unsigned": False,
            "modules_to_not_convert": ["unquantized"],
        }


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize_mxfp4(qweight: torch.Tensor, wscales: torch.Tensor) -> torch.Tensor:
    """Decode Omni's MXFP4 layout back to float for verification."""
    codes = torch.stack((qweight & 0x0F, qweight >> 4), dim=-1).reshape(qweight.shape[0], -1)
    magnitudes = torch.tensor(_E2M1_VALUES)[(codes & 0x07).long()]
    values = torch.where((codes & 0x08) != 0, -magnitudes, magnitudes)
    scales = torch.pow(2.0, wscales.T.float() - 127.0).repeat_interleave(32, dim=-1)
    return values * scales


def test_omni_mxfp4_block_scales_follow_rceil_convention():
    from auto_round.export.svdquant_omni import _mxfp4_block_scales

    weight = torch.zeros(3, 32)
    weight[0, 0] = 6.0
    weight[1, 0] = 7.5
    weight[2, 0] = 3.0
    assert _mxfp4_block_scales(weight).tolist() == [[0.0], [1.0], [-1.0]]


def test_omni_mxfp4_tensor_layout_and_dequantization(tmp_path):
    model = torch.nn.Module()
    model.config = {"_name_or_path": tmp_path}
    residual = torch.nn.Linear(32, 16, bias=False, dtype=torch.bfloat16)
    for name, value in PRESET_SCHEMES["MXFP4"].to_dict().items():
        setattr(residual, name, value)
    grid = torch.tensor(_E2M1_VALUES, dtype=torch.bfloat16)
    residual.weight.data = torch.stack([torch.cat([grid, -grid, grid, -grid]) for _ in range(16)])
    down = torch.nn.Linear(32, 8, bias=False, dtype=torch.bfloat16)
    up = torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
    model.projection = SVDQuantLinear(residual, down, up, torch.ones(32, dtype=torch.bfloat16))
    model.unquantized = torch.nn.Linear(32, 16)

    output = tmp_path / "mxfp4"
    output_format = get_formats("svdquant_omni", SimpleNamespace(scheme="MXFP4"))[0]
    assert output_format.save_quantized(output, model=model) is model
    tensors = load_file(output / "diffusion_pytorch_model.safetensors")

    assert set(tensors) == {
        *(f"projection.{suffix}" for suffix in ("qweight", "wscales", "proj_down", "proj_up", "smooth_factor")),
        "unquantized.weight",
        "unquantized.bias",
    }
    qweight = tensors["projection.qweight"]
    wscales = tensors["projection.wscales"]
    assert qweight.shape == (16, 16)
    assert qweight.dtype == torch.uint8
    assert wscales.shape == (1, 16)
    assert wscales.dtype == torch.uint8
    # The test weight is exactly representable, so RTN is lossless here.
    torch.testing.assert_close(_dequantize_mxfp4(qweight, wscales), residual.weight.float())
    torch.testing.assert_close(tensors["projection.proj_down"], down.weight.T.bfloat16())
    torch.testing.assert_close(tensors["projection.proj_up"], up.weight)
    torch.testing.assert_close(tensors["projection.smooth_factor"], torch.ones(32, dtype=torch.bfloat16))

    quantization = json.loads((output / "quantization_config.json").read_text())
    assert quantization["quant_method"] == "svdquant"
    assert quantization["precision"] == "mxfp4"
    assert quantization["rank"] == 8


def test_omni_format_rejects_incompatible_schemes(tmp_path, monkeypatch):
    output_format = get_formats("svdquant_omni", SimpleNamespace(scheme="NVFP4"))[0]
    assert output_format.output_format == "svdquant_omni"
    assert not output_format.is_supported_immediate_packing()
    assert not output_format.is_supported_immediate_saving()
    for preset in ("W4A16", "NVFP4_E5M3"):
        with pytest.raises(ValueError, match="NVFP4 W4A4 group16"):
            get_formats("svdquant_omni", SimpleNamespace(scheme=preset))
    for field, value in (("bits", 8), ("group_size", 32), ("act_bits", 16), ("act_dynamic", False)):
        scheme = PRESET_SCHEMES["NVFP4"].copy()
        setattr(scheme, field, value)
        with pytest.raises(ValueError, match=field):
            output_format.check_scheme_args(scheme)
    model = torch.nn.Module()
    residual = torch.nn.Linear(32, 16)
    for name, value in PRESET_SCHEMES["W4A16"].to_dict().items():
        setattr(residual, name, value)
    model.projection = SVDQuantLinear(
        residual, torch.nn.Linear(32, 8, bias=False), torch.nn.Linear(8, 16, bias=False), torch.ones(32)
    )
    with pytest.raises(ValueError, match="NVFP4 W4A4 group16"):
        save_svdquant_omni(model, tmp_path / "invalid")
    with pytest.raises(ValueError, match="incompatible residual scheme"):
        output_format._validate_svd_layer_overrides(model, {"projection": "W4A16"})

    from auto_round import AutoRound
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.algorithms.transforms.svdquant import SVDQuantConfig

    created = {}

    class FakeCompressor:
        def __init__(self, config, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr("auto_round.autoround._build_model_type_ctor_kwargs", lambda *args, **kwargs: ("llm", {}))
    monkeypatch.setattr("auto_round.autoround._get_compressor_class", lambda model_type, base_cls: FakeCompressor)
    for format in ("svdquant_omni", "svdquant_omni,fake"):
        AutoRound(
            "dummy-model",
            scheme="NVFP4",
            format=format,
            alg_configs=[SVDQuantConfig(smooth_enabled=False), RTNConfig(disable_opt_rtn=True)],
        )
        assert created["format"] == format
