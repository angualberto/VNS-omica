from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from features_pixelwise import load_mammo_image, make_8ch_features
from vnn_pixelwise_mammo import VNNPixelMammo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict VNN pixel-wise risk map for one mammography image")
    p.add_argument("--image", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--patch-size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_mammo_image(args.image)
    img_resized = cv2.resize(img, (args.patch_size, args.patch_size), interpolation=cv2.INTER_AREA)
    x = make_8ch_features(img_resized, args.patch_size).unsqueeze(0).to(device)
    model = VNNPixelMammo(in_ch=8).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
    prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "prob_map.png"), prob_u8)
    heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    gray = np.clip(img_resized * 255.0, 0, 255).astype(np.uint8)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.65, heat, 0.35, 0)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.imshow(gray, cmap="gray"); plt.title("mamografia"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(prob, cmap="magma"); plt.title("probabilidade"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.title("overlay"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "painel_predicao.png", dpi=160)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
