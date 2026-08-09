from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sharded_dataset import ShardBatchSampler, ShardedMILDataset
from train_mil_vnn_sharded import VNNMILAttention


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare old per-.pt loading with new sharded MIL loading.")
    p.add_argument("--sharded-manifest", required=True)
    p.add_argument("--old-manifest", help="Manifest from 1_precompute_features.py; optional.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--out-json")
    return p.parse_args()


def load_old_class():
    path = Path(__file__).with_name("2_dataset_cached.py")
    spec = importlib.util.spec_from_file_location("old_cached_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CachedMammoPatchDataset


class GPUMonitor:
    def __init__(self):
        self.samples: list[tuple[float, float]] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        def sample() -> None:
            while not self.stop.is_set():
                try:
                    raw = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                        text=True,
                    ).strip().splitlines()[0]
                    util, memory = raw.split(",")
                    self.samples.append((float(util), float(memory)))
                except Exception:
                    pass
                self.stop.wait(0.2)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def finish(self) -> dict[str, float | None]:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        return {
            "mean_gpu_utilization_percent": float(np.mean([v[0] for v in self.samples])) if self.samples else None,
            "max_gpu_memory_mib": float(max([v[1] for v in self.samples])) if self.samples else None,
        }


def benchmark(loader: DataLoader, model: VNNMILAttention, device: torch.device, max_batches: int, old: bool) -> dict:
    monitor = GPUMonitor()
    monitor.start()
    processed = 0
    count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            if old:
                x = batch[0].unsqueeze(1)
                valid = torch.ones((x.shape[0], 1), dtype=torch.bool)
            else:
                x, _y, valid, _metadata = batch
            x = x.to(device, non_blocking=True)
            if device.type != "cuda":
                x = x.float()
            valid = valid.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                model(x, valid)
            processed += int(x.shape[0] * x.shape[1])
            count += 1
            if count >= max_batches:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    metrics = {
        "batches": count,
        "patches": processed,
        "seconds": seconds,
        "seconds_per_batch": seconds / max(1, count),
        "patches_per_second": processed / max(seconds, 1e-9),
        **monitor.finish(),
    }
    if device.type == "cuda":
        metrics["torch_peak_vram_gb"] = torch.cuda.max_memory_reserved() / 1024**3
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    model = VNNMILAttention().to(device).eval()
    new_ds = ShardedMILDataset(args.sharded_manifest, split="train")
    new_sampler = ShardBatchSampler(new_ds, args.batch_size, shuffle=False)
    loader_kwargs = {
        "batch_sampler": new_sampler,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    new_metrics = benchmark(DataLoader(new_ds, **loader_kwargs), model, device, args.batches, old=False)
    result = {"sharded": new_metrics}
    print("B) sharded:", json.dumps(new_metrics, indent=2))

    if args.old_manifest:
        old_ds = load_old_class()(args.old_manifest)
        old_loader = DataLoader(
            old_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        old_metrics = benchmark(old_loader, model, device, args.batches, old=True)
        result["old_small_pt_files"] = old_metrics
        print("A) old small .pt files:", json.dumps(old_metrics, indent=2))
    else:
        result["old_small_pt_files"] = "not run; supply --old-manifest"
        print("A) old small .pt files skipped: provide --old-manifest to compare.")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
