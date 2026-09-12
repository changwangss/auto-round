# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Quantize Wan2.2 A14B experts concurrently on two GPUs and merge the pipeline.

The parent process starts one isolated worker per expert. Each worker loads a
complete BF16 pipeline and runs the real scheduler trajectory, but installs
calibration hooks and quantizers only on its assigned expert. The workers export
one canonical ``svdquant_omni`` component each. The parent then copies immutable
pipeline assets once and atomically installs both components.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from quantize_wan_a14b_svdquant import (
    MODEL_ID,
    MODEL_REVISION,
    PROFILES as BASE_PROFILES,
    audit_export,
    calibration_data,
)

EXPERTS = ("transformer", "transformer_2")
PROFILES = {name: dict(values) for name, values in BASE_PROFILES.items()}
PROFILES["quality"]["iters"] = 50
DEFAULT_PROMPTS_FILE = (
    Path(__file__).resolve().parent / "calibration_prompts" / "wan22_boxing_cats_32.txt"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["svdquant_omni"], default="svdquant_omni")
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--prompts-file", type=Path)
    data.add_argument("--dataset")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=16,
        help="Scheduler steps used by calibration",
    )
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Two physical CUDA device IDs, one for transformer and transformer_2",
    )
    parser.add_argument("--low-gpu-mem-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--gradient-accumulate-steps", type=int, default=8)
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--work-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-expert", choices=EXPERTS, help=argparse.SUPPRESS)
    parser.add_argument("--device", type=int, default=0, help=argparse.SUPPRESS)
    for key in PROFILES["smoke"]:
        if key in {"calib_steps", "nsamples"}:
            continue
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    args = parser.parse_args(argv)
    if args.prompts_file is None and args.dataset is None:
        args.prompts_file = DEFAULT_PROMPTS_FILE
    for key, value in PROFILES[args.profile].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.height < 16 or args.width < 16 or args.height % 16 or args.width % 16:
        parser.error("height and width must be positive multiples of 16")
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        parser.error("num-frames must be 4k+1")
    if not 0.0 < args.boundary_ratio < 1.0:
        parser.error("boundary-ratio must be between 0 and 1")
    if args.flow_shift <= 0.0:
        parser.error("flow-shift must be positive")
    if args.rank <= 0 or args.rank % 16:
        parser.error("rank must be a positive multiple of 16")
    if args.gradient_accumulate_steps < 1:
        parser.error("gradient-accumulate-steps must be positive")
    if min(getattr(args, key) for key in PROFILES["smoke"]) < 1 or args.calib_steps < 2:
        parser.error("profile parameters must be positive; calib-steps must be at least 2")
    if args.profile == "quality" and not (args.prompts_file or args.dataset):
        parser.error("quality requires representative --prompts-file or --dataset")
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not args.worker_expert and (len(gpu_ids) != 2 or len(set(gpu_ids)) != 2):
        parser.error("--gpus must contain two distinct CUDA device IDs")
    args.gpu_ids = gpu_ids
    return args


def _source_kwargs(args):
    return dict(revision=args.revision) if not Path(args.model).is_dir() else dict(local_files_only=True)


def _validate_expert(model, name):
    config = getattr(model, "config", None)
    expected = dict(
        num_layers=40,
        num_attention_heads=40,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
    )
    if config is None or any(getattr(config, key, None) != value for key, value in expected.items()):
        raise ValueError(f"{name} is not a Wan2.2 T2V A14B expert")


def run_worker(args):
    import diffusers
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    from auto_round import AutoRound
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.transforms.svdquant import SVDQuantConfig
    from auto_round.export.svdquant_omni import save_svdquant_omni

    if not torch.cuda.is_available():
        raise RuntimeError("worker requires a visible CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"worker must see exactly one GPU, found {torch.cuda.device_count()}; "
            "launch through the parent instead"
        )
    expert = args.worker_expert
    component_dir = args.work_root / expert
    component_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    source = _source_kwargs(args)
    start = time.monotonic()

    vae = AutoencoderKLWan.from_pretrained(
        args.model,
        subfolder="vae",
        torch_dtype=torch.float32,
        **source,
    )
    vae._keep_in_fp32_modules = [""]
    pipe = WanPipeline.from_pretrained(
        args.model,
        vae=vae,
        torch_dtype=torch.bfloat16,
        **source,
    )
    for name in EXPERTS:
        _validate_expert(getattr(pipe, name), name)
    source_boundary_ratio = float(pipe.config.boundary_ratio)
    if abs(source_boundary_ratio - args.boundary_ratio) > 1e-9:
        raise ValueError(
            f"source boundary_ratio={source_boundary_ratio} does not match "
            f"requested {args.boundary_ratio}"
        )
    pipe.scheduler.register_to_config(flow_shift=args.flow_shift)

    svd = SVDQuantConfig(
        rank=args.rank,
        smooth_enabled=True,
        smooth_num_grids=args.smooth_grids,
        smooth_max_calibration_calls=args.smooth_calls,
        residual_iters=args.residual_iters,
        residual_early_stop=True,
        low_rank_dtype="bf16",
        model_adapter="wan",
    )
    terminal = SignRoundConfig(
        iters=args.iters,
        nblocks=1,
        enable_quanted_input=True,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
    )
    compressor = AutoRound(
        pipe,
        scheme="MXFP4",
        alg_configs=[svd, terminal],
        dataset=calibration_data(args),
        nsamples=args.nsamples,
        batch_size=1,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        low_cpu_mem_usage=False,
        device_map=args.device,
        model_dtype="bf16",
        num_inference_steps=args.calib_steps,
        calib_num_inference_steps=args.calib_steps,
        pipeline_call_kwargs=dict(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            output_type="latent",
            guidance_scale_2=args.guidance_scale_2,
        ),
        guidance_scale=args.guidance_scale,
        generator_seed=args.seed,
        seed=args.seed,
        format="svdquant_omni",
    )
    # AutoRound normally selects pipe.transformer and discovers transformer_2,
    # then quantizes both sequentially. Select exactly one model for this worker
    # while leaving both BF16 experts in the pipeline so the latent trajectory
    # follows the real boundary_ratio and guidance_scale_2 behavior.
    compressor.model_context.model = getattr(pipe, expert)
    compressor._find_additional_transformers = lambda: []
    quantized_model, _ = compressor.quantize()
    save_svdquant_omni(quantized_model, component_dir)
    report = dict(
        expert=expert,
        boundary_ratio=source_boundary_ratio,
        flow_shift=float(pipe.scheduler.config.flow_shift),
        calib_num_inference_steps=args.calib_steps,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        seconds=time.monotonic() - start,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 2**30,
        gpu=torch.cuda.get_device_name(args.device),
        torch=torch.__version__,
        diffusers=diffusers.__version__,
    )
    (args.work_root / f"{expert}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def _resolve_source_dir(args):
    source = Path(args.model)
    if source.is_dir():
        return source.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(args.model, revision=args.revision)).resolve()


def _copy_asset(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink():
        source = source.resolve()
    if source.suffix == ".safetensors":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _copy_pipeline_assets(source, destination):
    excluded = {"transformer", "transformer_2"}
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.name in excluded:
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, copy_function=_copy_asset)
        else:
            _copy_asset(item, target)


def merge_outputs(args, work_root, elapsed):
    source = _resolve_source_dir(args)
    staging = args.output.parent / f".{args.output.name}.merge-{uuid.uuid4().hex}"
    try:
        _copy_pipeline_assets(source, staging)
        for expert in EXPERTS:
            shutil.copytree(work_root / expert, staging / expert, copy_function=_copy_asset)
        index_path = staging / "model_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for expert in EXPERTS:
            index[expert] = ["diffusers", "WanTransformer3DModel"]
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        worker_reports = {
            expert: json.loads((work_root / f"{expert}.json").read_text()) for expert in EXPERTS
        }
        manifest = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "gpu_ids"
        }
        manifest.update(
            mode="dual-expert-parallel",
            elapsed_seconds=elapsed,
            workers=worker_reports,
        )
        (staging / "quantization-run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report = audit_export(staging, "svdquant_omni")
        report["seconds"] = elapsed
        report["workers"] = worker_reports
        (staging / "export-audit.json").write_text(json.dumps(report, indent=2) + "\n")
        os.replace(staging, args.output)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _worker_command(args, expert, work_root):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-expert",
        expert,
        "--work-root",
        str(work_root),
        "--model",
        str(args.model),
        "--revision",
        args.revision,
        "--output",
        str(args.output),
        "--format",
        args.format,
        "--profile",
        args.profile,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-frames",
        str(args.num_frames),
        "--boundary-ratio",
        str(args.boundary_ratio),
        "--flow-shift",
        str(args.flow_shift),
        "--rank",
        str(args.rank),
        "--seed",
        str(args.seed),
        "--device",
        "0",
        "--guidance-scale",
        str(args.guidance_scale),
        "--guidance-scale-2",
        str(args.guidance_scale_2),
        "--gradient-accumulate-steps",
        str(args.gradient_accumulate_steps),
    ]
    for key in PROFILES["smoke"]:
        command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    if args.prompts_file:
        command.extend(["--prompts-file", str(args.prompts_file.resolve())])
    elif args.dataset:
        command.extend(["--dataset", args.dataset])
    command.append("--low-gpu-mem-usage" if args.low_gpu_mem_usage else "--no-low-gpu-mem-usage")
    return command


def run_parent(args):
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_root = args.output.parent / f".{args.output.name}.workers-{uuid.uuid4().hex}"
    commands = {
        expert: _worker_command(args, expert, work_root) for expert in EXPERTS
    }
    if args.dry_run:
        for expert, gpu in zip(EXPERTS, args.gpu_ids):
            print(f"CUDA_VISIBLE_DEVICES={gpu} " + " ".join(commands[expert]))
        return

    work_root.mkdir(parents=True, exist_ok=False)
    processes = {}
    logs = {}
    start = time.monotonic()
    try:
        for expert, gpu in zip(EXPERTS, args.gpu_ids):
            log_path = work_root / f"{expert}.log"
            log = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"Launching {expert} on physical GPU {gpu}; log={log_path}", flush=True)
            process = subprocess.Popen(
                commands[expert],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes[expert] = process
            logs[expert] = log

        failures = []
        while processes:
            for expert, process in list(processes.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                logs[expert].close()
                del processes[expert]
                if returncode:
                    failures.append((expert, returncode))
                else:
                    print(f"{expert} completed", flush=True)
            if failures:
                for process in processes.values():
                    os.killpg(process.pid, signal.SIGTERM)
                for process in processes.values():
                    process.wait()
                details = []
                for expert, returncode in failures:
                    log_path = work_root / f"{expert}.log"
                    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
                    details.append(f"{expert} exited {returncode}:\n{tail}")
                raise RuntimeError("\n".join(details))
            if processes:
                time.sleep(2)
        elapsed = time.monotonic() - start
        report = merge_outputs(args, work_root, elapsed)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for log in logs.values():
            if not log.closed:
                log.close()
        if not args.keep_workdirs and not processes:
            shutil.rmtree(work_root, ignore_errors=True)
        elif work_root.exists():
            print(f"Worker artifacts retained at {work_root}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.worker_expert:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
