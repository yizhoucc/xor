"""Profile compute cost of the deploy-time CNN and Transformer operators.

This is a synthetic-batch systems benchmark, not a training experiment.  Every
operator is measured sequentially in one process on the same GPU.  PyTorch's
profiler reports FLOPs for supported major operators; empirical latency and
throughput remain the primary runtime measurements.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deploy_distilled import build_model
from scripts.deploy_distilled_cnn import CNN


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loss(logits):
    classes = logits.shape[-1]
    target = torch.arange(logits.numel() // classes, device=logits.device) % classes
    return F.cross_entropy(logits.reshape(-1, classes), target)


def _time_forward(model, inputs, device, warmup, iterations):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)
        _sync(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
        _sync(device)
    return 1000.0 * (time.perf_counter() - start) / iterations


def _time_train(model, inputs, device, warmup, iterations):
    model.train()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _loss(model(inputs)).backward()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iterations):
        model.zero_grad(set_to_none=True)
        _loss(model(inputs)).backward()
    _sync(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - start) / iterations
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    return elapsed_ms, peak_mb


def _profile_forward_flops(model, inputs, device):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    model.eval()
    with torch.no_grad(), profile(activities=activities, with_flops=True) as prof:
        model(inputs)
        _sync(device)
    return int(sum(event.flops or 0 for event in prof.key_averages()))


def _measure(name, model, inputs, item_count, device, args):
    model = model.to(device)
    inputs = inputs.to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    forward_ms = _time_forward(model, inputs, device, args.warmup, args.forward_iters)
    train_ms, peak_mb = _time_train(
        model, inputs, device, max(2, args.warmup // 2), args.train_iters
    )
    flops = _profile_forward_flops(model, inputs, device)
    result = {
        "parameters": parameters,
        "profiled_forward_flops_per_batch": flops,
        "profiled_forward_flops_per_item": flops / item_count,
        "forward_ms_per_batch": forward_ms,
        "inference_items_per_second": item_count * 1000.0 / forward_ms,
        "train_step_ms_per_batch": train_ms,
        "training_items_per_second": item_count * 1000.0 / train_ms,
        "peak_training_memory_mb": peak_mb,
    }
    print(name, json.dumps(result, sort_keys=True), flush=True)
    del model, inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-results", default="exp/deploy_cnn_cifar10/results.json")
    parser.add_argument("--ffn-results", default="exp/deploy_ffn_d128/results.json")
    parser.add_argument("--output", default="results/audit/compute_cost_profile.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--forward-iters", type=int, default=50)
    parser.add_argument("--train-iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark must run in a Slurm GPU job")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)

    cnn_data = _load_json(args.cnn_results)
    ffn_data = _load_json(args.ffn_results)
    ffn_args = SimpleNamespace(**ffn_data["args"])
    batch_size = int(ffn_args.batch_size)
    context_size = int(ffn_args.context_size)
    vocab_size = 10000

    output = {
        "environment": {
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "precision": "float32",
            "notes": "Profiler FLOPs include only operations supported by PyTorch profiler.",
        },
        "cnn": {},
        "transformer_ffn_d128": {},
    }

    cnn_input = torch.randn(100, 3, 32, 32)
    for op in ("relu", "swiglu", "innernet", "distilled"):
        output["cnn"][op] = _measure(
            f"cnn/{op}", CNN(op, cnn_data.get("coeffs", {})), cnn_input,
            item_count=cnn_input.shape[0], device=device, args=args,
        )

    token_input = torch.randint(0, vocab_size, (batch_size, context_size))
    for op in ("gelu", "swiglu", "innernet", "distilled"):
        output["transformer_ffn_d128"][op] = _measure(
            f"ffn/{op}",
            build_model(op, vocab_size, ffn_args, ffn_data.get("coeffs", {})),
            token_input, item_count=token_input.numel(), device=device, args=args,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
