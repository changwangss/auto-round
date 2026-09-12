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

"""Canonical Wan SVDQuant MXFP4 export, independent of inference runtimes.

Disk weights contain ordinary low-first E2M1 nibbles and raw UE8M0 scale bytes.
Self-attention Q/K/V remain independent, retaining their individual smoothing
and low-rank decompositions. Both branches include the source smoothing exactly
once; only the residual branch divides its input by ``smooth_factor``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from auto_round.export.svdquant_adapters.wan import WAN_SVDQUANT_TARGET_MODULES, WanSVDQuantNunchakuAdapter
from auto_round.export.svdquant_mxfp4 import NunchakuMXFP4Packer, pack_nibbles, unpack_lowrank_weight
from auto_round.export.svdquant_nunchaku import (
    NUNCHAKU_WEIGHT_FILENAME,
    MXFP4ResidualTensorProvider,
    SVDQuantExportConfig,
    _prepare_export_records,
    _validate_packed_residual,
    unpack_nunchaku_16bit_vector,
)

# These non-block branches must retain their original floating-point linears.
WAN_OMNI_MODULES_TO_NOT_CONVERT = ["condition_embedder", "patch_embedding", "proj_out"]


def omni_quantization_config(rank: int) -> dict:
    """Return the explicit canonical layout and independent-QKV contract."""
    return dict(
        quant_method="svdquant",
        precision="mxfp4",
        rank=rank,
        fuse_qkv=False,
        act_unsigned=False,
        modules_to_not_convert=list(WAN_OMNI_MODULES_TO_NOT_CONVERT),
    )


def _canonical_residual(qweight, wscales, n, k):
    """Invert only integer physical layout; never reconstruct or requantize weights."""
    if k % 32:
        raise ValueError("Omni MXFP4 input width must be divisible by 32")
    packer = NunchakuMXFP4Packer()
    padded_n, padded_k = packer._ceil_to(n, 128), packer._ceil_to(k, 128)
    if qweight.dtype != torch.int8 or tuple(qweight.shape) != (padded_n, padded_k // 2):
        raise ValueError("invalid physical qweight shape or dtype")
    if wscales.dtype != torch.uint8 or tuple(wscales.shape) != (padded_k // 32, padded_n):
        raise ValueError("invalid physical wscales shape or dtype")
    codes = packer._unpack_weight_codes(qweight)[:n, :k]
    scales = packer._unpack_scale_codes(wscales)[:n, : k // 32]
    if bool((scales == 255).any()):
        raise ValueError("UE8M0 scale bytes must represent finite values")
    return pack_nibbles(codes).view(torch.int8), scales.T.contiguous()


def _cpu_tensor(value, name, *, positive=False):
    value = value.detach().cpu().contiguous()
    if value.is_floating_point() and (
        not bool(torch.isfinite(value).all()) or (positive and not bool((value > 0).all()))
    ):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must contain only {qualifier} values after BF16 conversion")
    return value


def collect_svdquant_omni_tensors(model, *, residual_provider=None):
    """Return ``(tensors, quantization_config)`` for a complete decomposed Wan expert."""
    adapter = WanSVDQuantNunchakuAdapter()
    _, records, rank = _prepare_export_records(model, SVDQuantExportConfig(), adapter)
    provider = residual_provider or MXFP4ResidualTensorProvider()
    tensors = {}
    for record in records:
        prefix = record.prefix
        if not bool((record.smooth > 0).all()):
            raise ValueError(f"{prefix} smooth must contain only positive finite values")
        payload = provider.tensors_for(record)
        _validate_packed_residual(payload, record)
        n, k = record.residual_weight.shape
        qweight, wscales = _canonical_residual(payload["qweight"], payload["wscales"], n, k)
        values = dict(
            qweight=qweight,
            wscales=wscales,
            proj_down=(record.lora_down.double() * record.smooth.double()).T.bfloat16(),
            proj_up=record.lora_up.bfloat16(),
            smooth_factor=record.smooth.double().reciprocal().bfloat16(),
            bias=torch.zeros(n, dtype=torch.bfloat16) if record.bias is None else record.bias.bfloat16(),
        )
        for suffix, value in values.items():
            tensors[f"{prefix}.{suffix}"] = _cpu_tensor(value, f"{prefix}.{suffix}", positive=suffix == "smooth_factor")
    for key, value in adapter.extra_tensors(model).items():
        if key in tensors:
            raise ValueError(f"duplicate export tensor {key}")
        tensors[key] = _cpu_tensor(value, key)
    return tensors, omni_quantization_config(rank)


def _save_component(tensors, config, quantization, output_dir):
    from safetensors.torch import save_file

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["_class_name"] = "WanTransformer3DModel"
    config["quantization_config"] = quantization
    metadata = dict(
        format="pt",
        model_class="WanTransformer3DModel",
        config=json.dumps(config, sort_keys=True),
        quantization_config=json.dumps(quantization, sort_keys=True),
    )
    temporary = output / ".diffusion_pytorch_model.omni.tmp.safetensors"
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, output / NUNCHAKU_WEIGHT_FILENAME)
    finally:
        temporary.unlink(missing_ok=True)
    for filename, value in [("config.json", config), ("quantization_config.json", quantization)]:
        (output / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output / NUNCHAKU_WEIGHT_FILENAME)


def save_svdquant_omni(model, output_dir, *, residual_provider=None):
    """Save one complete Wan expert using canonical Omni tensor names and layouts."""
    tensors, quantization = collect_svdquant_omni_tensors(model, residual_provider=residual_provider)
    config = WanSVDQuantNunchakuAdapter()._resolved_config(model)
    return _save_component(tensors, config, quantization, output_dir)


def _convert_component(source, output):
    from safetensors import safe_open

    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("model_class") != "NunchakuWanTransformer3DModel":
            raise ValueError("converter requires a Nunchaku Wan checkpoint with model_class metadata")
        config = json.loads(metadata.get("config", "{}"))
        quantization = json.loads(metadata.get("quantization_config", "{}"))
        expected_weight = dict(dtype="fp4_e2m1_all", scale_dtype="ue8m0", group_size=32)
        if (
            quantization.get("method") != "svdquant"
            or quantization.get("weight") != expected_weight
            or quantization.get("activation") != dict(dtype="fp4_e2m1_all", scale_dtype="ue8m0", group_size=32)
        ):
            raise ValueError("converter requires SVDQuant MXFP4 E2M1 group32 metadata")
        rank = quantization.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("checkpoint rank must be a positive integer")
        try:
            layers = int(config["num_layers"])
            hidden = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
            ffn = int(config["ffn_dim"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint config must supply Wan logical dimensions") from exc
        if min(layers, hidden, ffn) <= 0:
            raise ValueError("Wan logical dimensions must be positive")
        prefixes = {f"blocks.{i}.{path}" for i in range(layers) for path in WAN_SVDQUANT_TARGET_MODULES}
        keys = set(handle.keys())
        if {key[:-8] for key in keys if key.endswith(".qweight")} != prefixes:
            raise ValueError("complete Wan projection mismatch in checkpoint")
        tensors = {}
        consumed = set()
        for prefix in sorted(prefixes):
            n, k = (
                (ffn, hidden)
                if prefix.endswith("ffn.net.0.proj")
                else ((hidden, ffn) if prefix.endswith("ffn.net.2") else (hidden, hidden))
            )

            def get(suffix):
                key = f"{prefix}.{suffix}"
                consumed.add(key)
                return handle.get_tensor(key)

            qweight, wscales = _canonical_residual(get("qweight"), get("wscales"), n, k)
            down = unpack_lowrank_weight(get("lora_down"), down=True)
            up = unpack_lowrank_weight(get("lora_up"), down=False)
            if rank > down.shape[0] or rank > up.shape[1] or k > down.shape[1] or n > up.shape[0]:
                raise ValueError(f"{prefix} low-rank shape is inconsistent with metadata")
            smooth = unpack_nunchaku_16bit_vector(get("smooth"))
            bias = unpack_nunchaku_16bit_vector(get("bias"))
            if smooth.numel() < k or bias.numel() < n:
                raise ValueError(f"{prefix} vector shape is inconsistent with metadata")
            values = dict(
                qweight=qweight,
                wscales=wscales,
                proj_down=down[:rank, :k].T.bfloat16(),
                proj_up=up[:n, :rank].bfloat16(),
                smooth_factor=smooth[:k].bfloat16(),
                bias=bias[:n].bfloat16(),
            )
            for suffix, value in values.items():
                tensors[f"{prefix}.{suffix}"] = _cpu_tensor(
                    value, f"{prefix}.{suffix}", positive=suffix == "smooth_factor"
                )
            # Original smoothing is calibration metadata, not another runtime operation.
            consumed.add(f"{prefix}.smooth_orig")
        for key in sorted(keys - consumed):
            if any(key.startswith(prefix + ".") for prefix in prefixes):
                raise ValueError(f"unexpected quantized tensor {key}")
            tensors[key] = _cpu_tensor(handle.get_tensor(key), key)
    return _save_component(tensors, config, omni_quantization_config(rank), output)


def _copy_asset(source, destination):
    # Weight blobs can share storage; configuration and other editable assets cannot.
    source, destination = Path(source), Path(destination)
    if source.suffix == ".safetensors":
        try:
            os.link(source, destination)
            return str(destination)
        except OSError:
            pass
    return shutil.copy2(source, destination)


def convert_wan_nunchaku_to_omni(source, output_dir):
    """Convert a Wan onefile or complete pipeline into a new directory without requantization.

    Immutable auxiliary safetensors may be hardlinked on the same filesystem.
    Configurations are copied, and source components are never modified.
    """
    source, output = Path(source).resolve(), Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if source.is_file():
        return _convert_component(source, output)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.resolve().is_relative_to(source):
        raise ValueError("output must not be inside source pipeline")
    index_path = source / "model_index.json"
    if not index_path.is_file():
        return _convert_component(source / NUNCHAKU_WEIGHT_FILENAME, output)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    components = [name for name in ("transformer", "transformer_2") if index.get(name) not in (None, [None, None])]
    for name in components:
        if not (source / name).is_dir():
            raise FileNotFoundError(f"pipeline declares missing component: {source / name}")
    if not components:
        raise ValueError("pipeline has no Wan transformer components")
    for name in components:
        if not (source / name / NUNCHAKU_WEIGHT_FILENAME).is_file():
            raise FileNotFoundError(source / name / NUNCHAKU_WEIGHT_FILENAME)
    output.mkdir(parents=True, exist_ok=False)
    for name in components:
        _convert_component(source / name / NUNCHAKU_WEIGHT_FILENAME, output / name)
        index[name] = ["diffusers", "WanTransformer3DModel"]
    for item in source.iterdir():
        if item.name in {*components, "model_index.json"}:
            continue
        if item.is_dir():
            shutil.copytree(item, output / item.name, copy_function=_copy_asset)
        else:
            _copy_asset(item, output / item.name)
    (output / "model_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output)
