"""Reproduce Figure 11 settings from https://huggingface.co/blog/torch-profiler.

Run: .venv/bin/python matmul_add.py
Open the exported JSON at https://ui.perfetto.dev.
"""
import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule


def matmul_add(x, w, b):
    return torch.add(torch.matmul(x, w), b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--warmup", action="store_true", help="Run 3 extra warmup calls before profiling")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU and CUDA-enabled PyTorch are required.")
    if args.size < 1:
        parser.error("--size must be positive")

    torch.manual_seed(0)
    x, w, b = [torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

    def step():
        with record_function("matmul_add"):
            return matmul_add(x, w, b)

    if args.warmup:
        for _ in range(3):
            step()
        torch.cuda.synchronize()

    # Figure 11: record all 20 steps, starting at step 0.
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=20, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(20):
            step()
            prof.step()
    torch.cuda.synchronize()

    output = Path(__file__).resolve().parent / "traces" / "01_matmul_add"
    output.mkdir(parents=True, exist_ok=True)
    tag = f"{args.size}_bf16_{'warm' if args.warmup else 'cold'}_eager"
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    (output / f"{tag}.txt").write_text(table + "\n")
    prof.export_chrome_trace(str(output / f"{tag}.json"))
    metadata = f"PyTorch: {torch.__version__}\nCUDA runtime: {torch.version.cuda}\nGPU: {torch.cuda.get_device_name(0)}\nShape: ({args.size}, {args.size}) for x, w, b\nDtype: bfloat16\nSchedule: wait=0, warmup=0, active=20, repeat=1\nExtra warmup: {args.warmup}\n"
    (output / f"{tag}.meta.txt").write_text(metadata)
    print(metadata)
    print(table)
    print(f"Saved table and Chrome trace to {output / tag}.*")


if __name__ == "__main__":
    main()
