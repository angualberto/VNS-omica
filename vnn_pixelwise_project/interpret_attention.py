from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hybrid_cnn_vnn_mil import HybridCNNVNNMIL
from sharded_dataset import ShardedMILDataset
from train_hybrid_mil import load_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render MIL attention over cached mammography patches only.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--metadata-csv")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--predictions", help="predictions.csv for prioritizing representative examples.")
    p.add_argument("--n-positive", type=int, default=5)
    p.add_argument("--n-negative", type=int, default=5)
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def choose_examples(dataset: ShardedMILDataset, predictions: str | None, n_positive: int, n_negative: int) -> pd.DataFrame:
    frame = dataset.df.copy()
    if predictions:
        scores = pd.read_csv(predictions, dtype={"patient_id": str, "image_id": str})
        frame = frame.merge(scores[["patient_id", "image_id", "score"]], on=["patient_id", "image_id"], how="left")
        positives = frame[frame.y == 1].sort_values("score", ascending=False).head(n_positive)
        negatives = frame[frame.y == 0].sort_values("score", ascending=False).head(n_negative)
    else:
        positives = frame[frame.y == 1].head(n_positive)
        negatives = frame[frame.y == 0].head(n_negative)
    return pd.concat([positives, negatives]).drop_duplicates(["patient_id", "image_id"])


def main() -> None:
    args = parse_args()
    manifest_args = SimpleNamespace(manifest=args.manifest, metadata_csv=args.metadata_csv, max_images=0)
    manifest = load_manifest(manifest_args)
    dataset = ShardedMILDataset(manifest, split="test")
    examples = choose_examples(dataset, args.predictions, args.n_positive, args.n_negative)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mode = checkpoint.get("mode", "hybrid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCNNVNNMIL(mode=mode).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    lookup = {(str(row.patient_id), str(row.image_id)): i for i, row in dataset.df.iterrows()}
    with torch.inference_mode():
        for row in examples.itertuples(index=False):
            index = lookup[(str(row.patient_id), str(row.image_id))]
            x, y, valid, metadata = dataset[index]
            batch = x.unsqueeze(0).to(device)
            if device.type != "cuda": batch = batch.float()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, details = model(batch, valid.unsqueeze(0).to(device), return_details=True)
            attention = details["attention"][0].float().cpu().numpy()
            score = float(torch.sigmoid(logits)[0].cpu())
            score_cnn = float(torch.sigmoid(details["cnn_logits"])[0].cpu()) if details["cnn_logits"] is not None else None
            score_vnn = float(torch.sigmoid(details["vnn_logits"])[0].cpu()) if details["vnn_logits"] is not None else None
            valid_ids = np.flatnonzero(valid.numpy())
            ranked = valid_ids[np.argsort(attention[valid_ids])[::-1]]
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            for patch_id, axis in enumerate(axes.flat):
                patch = x[patch_id, 0].float().numpy()
                axis.imshow(patch, cmap="gray", vmin=0, vmax=1)
                axis.imshow(np.ones_like(patch), cmap="magma", alpha=float(attention[patch_id]) * 0.45, vmin=0, vmax=1)
                axis.set_title(f"p{patch_id} att={attention[patch_id]:.3f}")
                axis.axis("off")
            fig.suptitle(f"{mode} label={int(y.item())} score={score:.4f} patient={row.patient_id} image={row.image_id}")
            fig.tight_layout()
            stem = f"label{int(y.item())}_patient_{row.patient_id}_image_{row.image_id}"
            fig.savefig(out / f"{stem}_attention_overlay.png", dpi=150); plt.close(fig)
            for rank, patch_id in enumerate(ranked[:args.top_k], start=1):
                plt.imsave(out / f"{stem}_top{rank}_patch{patch_id}.png", x[patch_id, 0].float().numpy(), cmap="gray", vmin=0, vmax=1)
                coords = metadata["coords"][patch_id]
                rows.append({"patient_id": str(row.patient_id), "image_id": str(row.image_id), "y": int(y.item()),
                             "mode": mode, "score": score, "score_cnn": score_cnn, "score_vnn": score_vnn,
                             "rank": rank, "patch_index": int(patch_id), "attention": float(attention[patch_id]),
                             "x0": int(coords[0]), "y0": int(coords[1]), "view": metadata["view"],
                             "density": metadata["density"], "laterality": metadata["laterality"],
                             "difficult_negative_case": metadata["difficult_negative_case"]})
    pd.DataFrame(rows).to_csv(out / "attention_examples.csv", index=False)
    print("saved:", out)


if __name__ == "__main__":
    main()
