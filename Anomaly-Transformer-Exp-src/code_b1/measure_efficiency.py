import os
import time
import copy
import random
import numpy as np
import torch
from thop import profile

from solver import Solver


# =========================
# Basic utils
# =========================
def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def measure_flops_g(model, input_shape, device):
    """
    FLOPs are measured for a single forward pass with batch size = 1.
    input_shape: (1, win_size, input_c)
    """
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return flops / 1e9


@torch.no_grad()
def measure_inference_time_ms(model, batch_shape, device, warmup=20, test_iters=100):
    """
    Inference time per batch (ms), measured on synthetic input with fixed batch size.
    batch_shape: (batch_size, win_size, input_c)
    """
    model.eval()
    x = torch.randn(*batch_shape).to(device)

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(test_iters):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms

        return float(np.mean(times))

    else:
        # CPU fallback
        for _ in range(warmup):
            _ = model(x)

        start = time.time()
        for _ in range(test_iters):
            _ = model(x)
        end = time.time()

        return (end - start) * 1000.0 / test_iters


def build_base_config():
    """
    Put your common config here.
    """
    return {
        "lr": 1e-4,
        "num_epochs": 10,
        "k": 3,
        "win_size": 100,
        "batch_size": 128,
        "pretrained_model": None,
        "mode": "train",
        "model_save_path": "./checkpoints_tmp",
        "anormly_ratio": 0.5,
    }


def build_dataset_configs():
    """
    Edit data_path if needed.
    input_c/output_c below are common settings in many public implementations.
    Please verify they match your processed data.
    """
    return {
        "SMD": {
            "dataset": "SMD",
            "data_path": "./dataset/SMD",
            "input_c": 38,
            "output_c": 38,
        },
        "SMAP": {
            "dataset": "SMAP",
            "data_path": "./dataset/SMAP",
            "input_c": 25,
            "output_c": 25,
        },
        "PSM": {
            "dataset": "PSM",
            "data_path": "./dataset/PSM",
            "input_c": 25,
            "output_c": 25,
        },
    }


# =========================
# Training time measurement
# =========================
def measure_training_time_per_epoch_with_solver_train(cfg, repeats=3):
    """
    Measure training time per epoch by calling solver.train() with num_epochs=1.
    This assumes solver.train() performs exactly the training pipeline for the given epochs.
    If your solver.train() also includes heavy evaluation/checkpointing, the time measured
    will include that overhead.

    repeats: run several times and average for stability.
    """
    times = []

    for r in range(repeats):
        cfg_run = copy.deepcopy(cfg)
        cfg_run["num_epochs"] = 1

        # Rebuild solver each repeat to avoid contamination from previous run
        solver = Solver(cfg_run)
        device = solver.device

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        start = time.time()
        solver.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        times.append(end - start)

    return float(np.mean(times))


# =========================
# Full measurement for one dataset
# =========================
def measure_one_dataset(dataset_name, cfg, infer_warmup=20, infer_test_iters=100, train_repeats=3):
    print(f"\n{'=' * 20} {dataset_name} {'=' * 20}")
    print("Building solver for static measurements ...")

    solver = Solver(copy.deepcopy(cfg))
    model = solver.model
    device = solver.device

    win_size = cfg["win_size"]
    input_c = cfg["input_c"]
    batch_size = cfg["batch_size"]

    # 1) Params
    params_m = count_trainable_params_m(model)

    # 2) FLOPs for a single forward pass, batch=1
    flops_g = measure_flops_g(model, (1, win_size, input_c), device)

    # 3) Inference time per batch
    infer_ms = measure_inference_time_ms(
        model=model,
        batch_shape=(batch_size, win_size, input_c),
        device=device,
        warmup=infer_warmup,
        test_iters=infer_test_iters,
    )

    # 4) Training time per epoch
    print("Measuring training time per epoch ...")
    train_sec = measure_training_time_per_epoch_with_solver_train(
        cfg=cfg,
        repeats=train_repeats,
    )

    result = {
        "dataset": dataset_name,
        "input_c": input_c,
        "params_m": params_m,
        "flops_g": flops_g,
        "train_sec": train_sec,
        "infer_ms": infer_ms,
    }

    print(f"Dataset: {dataset_name} (C = {input_c})")
    print(f"Params (M): {params_m:.6f}")
    print(f"FLOPs (G): {flops_g:.6f}")
    print(f"Training Time / Epoch (s): {train_sec:.6f}")
    print(f"Inference Time / Batch (ms): {infer_ms:.6f}")

    return result


def print_latex_friendly_summary(results):
    print("\n" + "=" * 60)
    print("LaTeX-friendly summary")
    print("=" * 60)
    for r in results:
        print(
            f'{r["dataset"]} (C={r["input_c"]}) '
            f'& AT / Ours '
            f'& {r["params_m"]:.3f} '
            f'& {r["flops_g"]:.3f} '
            f'& {r["train_sec"]:.2f} '
            f'& {r["infer_ms"]:.2f}'
        )


def main():
    set_seed(2024)
    torch.backends.cudnn.benchmark = True

    base_cfg = build_base_config()
    dataset_cfgs = build_dataset_configs()

    all_results = []

    for name in ["SMD", "SMAP", "PSM"]:
        cfg = copy.deepcopy(base_cfg)
        cfg.update(dataset_cfgs[name])

        # Create checkpoint folder per dataset
        cfg["model_save_path"] = os.path.join("./checkpoints_tmp", name)
        os.makedirs(cfg["model_save_path"], exist_ok=True)

        result = measure_one_dataset(
            dataset_name=name,
            cfg=cfg,
            infer_warmup=20,
            infer_test_iters=100,
            train_repeats=3,
        )
        all_results.append(result)

    print_latex_friendly_summary(all_results)


if __name__ == "__main__":
    main()