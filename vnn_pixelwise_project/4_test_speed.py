from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vnn_pixelwise_mammo import VNNPixelMammo


def dataset_class():
    path = Path(__file__).with_name("2_dataset_cached.py")
    spec = importlib.util.spec_from_file_location("dataset_cached", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.CachedMammoPatchDataset


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark cached patch DataLoader and VNN forward")
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batches", type=int, default=20)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    ds = dataset_class()(args.manifest, return_metadata=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device == "cuda"), persistent_workers=(args.num_workers > 0))
    model = VNNPixelMammo(in_ch=8).to(device).eval()
    times = []
    with torch.no_grad():
        for i, (x, _y) in enumerate(loader):
            if i >= args.batches:
                break
            start = time.perf_counter()
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if i == 0:
                print("input shape:", tuple(x.shape))
                print("output shape:", tuple(out.shape))
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        print("memory allocated GB:", torch.cuda.memory_allocated() / 1024**3)
        print("memory reserved GB:", torch.cuda.memory_reserved() / 1024**3)
    print("batches measured:", len(times))
    print("mean seconds per batch:", sum(times) / max(len(times), 1))


if __name__ == "__main__":
    main()
