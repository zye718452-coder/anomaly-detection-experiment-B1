import time
import torch
from thop import profile

from solver import Solver


def count_trainable_params_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


@torch.no_grad()
def measure_inference_time_ms(model, data_loader, device, warmup=10, test_batches=100):
    model.eval()
    batches = []

    for batch in data_loader:
        x = batch[0].float().to(device)
        batches.append(x)
        if len(batches) >= warmup + test_batches:
            break

    for i in range(min(warmup, len(batches))):
        _ = model(batches[i])

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    actual = min(test_batches, len(batches) - warmup)
    for i in range(warmup, warmup + actual):
        _ = model(batches[i])

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    return (end - start) * 1000 / actual


def measure_flops_g(model, input_shape, device):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops / 1e9


def build_config():
    return {
        "lr": 1e-4,
        "num_epochs": 10,
        "k": 3,
        "win_size": 100,
        "input_c": 38,
        "output_c": 38,
        "batch_size": 128,
        "pretrained_model": None,
        "dataset": "SMD",
        "mode": "train",
        "data_path": "./dataset/SMD",   # 如果路径不对，改这里
        "model_save_path": "./checkpoints_tmp",
        "anormly_ratio": 0.5,
    }


def main():
    print("Building AT ...")
    solver_at = Solver(build_config())
    device = solver_at.device

    at_params = count_trainable_params_m(solver_at.model)
    at_flops = measure_flops_g(solver_at.model, (1, 100, 38), device)
    at_infer_time = measure_inference_time_ms(solver_at.model, solver_at.test_loader, device)

    print("\n===== AT =====")
    print(f"Params (M): {at_params:.6f}")
    print(f"FLOPs (G): {at_flops:.6f}")
    print(f"Inference Time / Batch (ms): {at_infer_time:.6f}")


if __name__ == "__main__":
    main()