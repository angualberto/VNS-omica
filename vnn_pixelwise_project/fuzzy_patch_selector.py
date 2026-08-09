#!/usr/bin/env python3
"""Offline fuzzy chromatic/spectral re-selection of already cached MIL patches.

The selector reads precomputed patch shards, never DICOM files, and never uses
the cancer label when ranking patches. The output remains experimental and is
not a lesion localization or medical diagnosis.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from analisar_alizams_cores_dicom_csv import apply_lut, load_lut, lut_profile
from fuzzy_cancer_rules import OUTPUT_MEMBERSHIP, UNIVERSE, membership


KEYS = ["patient_id", "image_id"]
COLOR_NAMES = ["vermelho", "laranja", "amarelo", "verde", "ciano", "azul", "roxo", "rosa"]


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Select informative cached patches using interpretable fuzzy color rules.")
    p.add_argument("--manifest", required=True, help="Existing image-level sharded MIL manifest.")
    p.add_argument("--out-dir", required=True, help="Output cache_fuzzy_selected directory.")
    p.add_argument("--metadata-csv", help="RSNA train.csv used only to restore view/density/laterality metadata.")
    p.add_argument("--lut-file", default=str(project.parent / "sistema_integrado_rsna_csv_treino/alizams_rainbowb_cores_todas_imagens/alizams_luts.h"))
    p.add_argument("--top-k", type=int, default=4, help="Selected patches per image from the cached candidate bag.")
    p.add_argument("--shard-size", type=int, default=1024)
    p.add_argument("--ranking-mode", choices=["color", "spectral"], default="spectral",
                   help="spectral adds FFT, multilevel Haar wavelet and entropy rules to color rules.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--metric-batch-size", type=int, default=1024)
    p.add_argument("--overlay-images", type=int, default=12)
    p.add_argument("--selection-strategy", choices=["score", "greedy-avl"], default="greedy-avl",
                   help="score keeps the old top-k ordering; greedy-avl selects patches iteratively using an AVL tree.")
    p.add_argument("--greedy-distance-penalty", type=float, default=0.25,
                   help="How strongly the greedy selector penalizes patches near already selected ones.")
    p.add_argument("--greedy-distance-scale", type=float, default=512.0,
                   help="Spatial scale in pixels used by the greedy diversity penalty.")
    p.add_argument("--no-mmap", action="store_true", help="Load input shards sequentially instead of memory mapping.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda x: str(x).strip().lower() in {"1", "true", "t", "yes", "y"})


def restore_metadata(manifest: pd.DataFrame, source: str | None) -> pd.DataFrame:
    out = manifest.copy()
    for key in KEYS:
        out[key] = out[key].astype(str)
    if source:
        meta = pd.read_csv(source, dtype={"patient_id": str, "image_id": str})
        cols = KEYS + [c for c in ["laterality", "view", "density", "difficult_negative_case"] if c in meta]
        out = out.merge(meta[cols].drop_duplicates(KEYS), on=KEYS, how="left", suffixes=("", "_rsna"), validate="one_to_one")
        for name in ["laterality", "view", "density", "difficult_negative_case"]:
            restored = f"{name}_rsna"
            if restored in out:
                out[name] = out[restored].combine_first(out[name]) if name in out else out[restored]
                out = out.drop(columns=restored)
    for name, default in {"laterality": "", "view": "UNKNOWN", "density": "UNKNOWN", "difficult_negative_case": False}.items():
        if name not in out:
            out[name] = default
        out[name] = out[name].fillna(default)
    out["difficult_negative_case"] = normalize_bool(out["difficult_negative_case"])
    train = set(out.loc[out.split.astype(str) == "train", "patient_id"])
    test = set(out.loc[out.split.astype(str) == "test", "patient_id"])
    if train & test:
        raise ValueError("Patient leakage detected in input manifest.")
    return out


class ChromaticMetrics:
    def __init__(self, lut: np.ndarray, device: torch.device):
        bins, hue, _sat, _val, weight = lut_profile(lut)
        self.device = device
        self.nlut = len(lut)
        self.bin = torch.as_tensor(bins - 1, dtype=torch.long, device=device)
        self.hue = torch.as_tensor(hue, dtype=torch.float32, device=device)
        self.weight = torch.as_tensor(weight, dtype=torch.float32, device=device)

    def compute(self, gray_cpu: torch.Tensor, batch_size: int) -> np.ndarray:
        outputs = []
        for start in range(0, len(gray_cpu), batch_size):
            gray = gray_cpu[start:start + batch_size].to(self.device, dtype=torch.float32, non_blocking=True)
            mask = gray > 0.02
            empty = ~mask.flatten(1).any(dim=1)
            if empty.any():
                mask[empty] = True
            index = torch.clamp(torch.floor(gray * self.nlut).long(), 0, self.nlut - 1)
            color_bin = self.bin[index].flatten(1)
            pixel_weight = (self.weight[index] * mask).flatten(1)
            total = pixel_weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
            frequencies = torch.zeros((len(gray), 8), dtype=torch.float32, device=self.device)
            frequencies.scatter_add_(1, color_bin, pixel_weight)
            frequencies = frequencies / total
            hue_mean = (self.hue[index] * self.weight[index] * mask).flatten(1).sum(1, keepdim=True) / total
            outputs.append(torch.cat([hue_mean, frequencies], dim=1).cpu().numpy())
            del gray, mask, index, color_bin, pixel_weight, frequencies, hue_mean
        return np.concatenate(outputs, axis=0)


def fuzzy_color_score(values: dict[str, float], density: str, view: str) -> tuple[float, list[dict[str, object]]]:
    m = lambda feature, level: membership(feature, level, values.get(feature, np.nan))
    rules = [
        ("C1", min(m("freq_roxo", "alto"), m("freq_rosa", "alto"), m("freq_azul", "baixo")), "alta", "roxo e rosa altos com azul baixo"),
        ("C2", min(m("magenta_ratio", "alto"), m("hue_medio", "alto")), "alta", "razao magenta e hue altos"),
        ("C3", min(max(m("freq_verde", "alto"), m("freq_azul", "alto")), m("freq_roxo", "baixo")), "baixa", "verde/azul altos com roxo baixo"),
        ("C4", 0.35 * m("magenta_ratio", "alto") if str(density).upper() in {"B", "C"} else 0.0, "alta", "density B/C reforca razao magenta"),
        ("C5", 0.30 * min(max(m("freq_roxo", "alto"), m("freq_rosa", "alto")), 1.0) if str(view).upper() == "MLO" else 0.0, "alta", "view MLO reforca roxo/rosa"),
    ]
    aggregate = np.zeros_like(UNIVERSE)
    active = []
    for code, strength, consequence, text in rules:
        if strength <= 0:
            continue
        aggregate = np.maximum(aggregate, np.minimum(strength, OUTPUT_MEMBERSHIP[consequence]))
        active.append({"regra": code, "intensidade": float(strength), "consequencia": consequence, "descricao": text})
    score = 0.5 if aggregate.sum() <= 1e-12 else float(np.sum(UNIVERSE * aggregate) / np.sum(aggregate))
    return score, sorted(active, key=lambda row: row["intensidade"], reverse=True)


def unit_low(value: float) -> float:
    return float(np.clip((0.45 - value) / 0.25, 0.0, 1.0))


def unit_high(value: float) -> float:
    return float(np.clip((value - 0.45) / 0.25, 0.0, 1.0))


def fuzzy_spectral_score(values: dict[str, float]) -> tuple[float, list[dict[str, object]]]:
    """Mamdani score over per-image normalized spectral patch evidence."""
    rules = [
        ("S1", min(unit_high(values["fft_high_score"]), unit_high(values["wavelet_high_score"]), unit_high(values["entropia_score"])), "alta", "FFT, wavelet e entropia altos"),
        ("S2", min(unit_high(values["high_low_ratio_score"]), unit_high(values["gradiente_score"])), "alta", "razao espectral high/low e gradiente altos"),
        ("S3", min(unit_low(values["fft_high_score"]), unit_low(values["wavelet_high_score"]), unit_low(values["entropia_score"])), "baixa", "FFT, wavelet e entropia baixos"),
    ]
    aggregate = np.zeros_like(UNIVERSE); active = []
    for code, strength, consequence, description in rules:
        if strength <= 0:
            continue
        aggregate = np.maximum(aggregate, np.minimum(strength, OUTPUT_MEMBERSHIP[consequence]))
        active.append({"regra": code, "intensidade": float(strength), "consequencia": consequence, "descricao": description})
    score = 0.5 if aggregate.sum() <= 1e-12 else float(np.sum(UNIVERSE * aggregate) / np.sum(aggregate))
    return score, sorted(active, key=lambda row: row["intensidade"], reverse=True)


def normalized(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    low, high = float(values[finite].min()), float(values[finite].max())
    if high <= low + 1e-12:
        return np.full_like(values, 0.5, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


class AVLNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left: AVLNode | None = None
        self.right: AVLNode | None = None
        self.height = 1


class AVLTree:
    def __init__(self):
        self.root: AVLNode | None = None

    @staticmethod
    def _height(node: AVLNode | None) -> int:
        return node.height if node is not None else 0

    @staticmethod
    def _balance(node: AVLNode | None) -> int:
        return AVLTree._height(node.left) - AVLTree._height(node.right) if node is not None else 0

    @staticmethod
    def _update(node: AVLNode) -> AVLNode:
        node.height = 1 + max(AVLTree._height(node.left), AVLTree._height(node.right))
        return node

    @staticmethod
    def _rotate_right(node: AVLNode) -> AVLNode:
        pivot = node.left
        assert pivot is not None
        node.left = pivot.right
        pivot.right = node
        AVLTree._update(node)
        return AVLTree._update(pivot)

    @staticmethod
    def _rotate_left(node: AVLNode) -> AVLNode:
        pivot = node.right
        assert pivot is not None
        node.right = pivot.left
        pivot.left = node
        AVLTree._update(node)
        return AVLTree._update(pivot)

    def insert(self, key, value) -> None:
        def _insert(node: AVLNode | None, key, value) -> AVLNode:
            if node is None:
                return AVLNode(key, value)
            if key < node.key:
                node.left = _insert(node.left, key, value)
            else:
                node.right = _insert(node.right, key, value)
            node = self._update(node)
            balance = self._balance(node)
            if balance > 1:
                if key > node.left.key:
                    node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
            if balance < -1:
                if key < node.right.key:
                    node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
            return node

        self.root = _insert(self.root, key, value)

    def pop_max(self):
        if self.root is None:
            return None, None

        def _pop_max(node: AVLNode) -> tuple[AVLNode | None, AVLNode]:
            if node.right is None:
                return node.left, node
            node.right, best = _pop_max(node.right)
            node = self._update(node)
            balance = self._balance(node)
            if balance > 1:
                if self._balance(node.left) < 0:
                    node.left = self._rotate_left(node.left)
                node = self._rotate_right(node)
            elif balance < -1:
                if self._balance(node.right) > 0:
                    node.right = self._rotate_right(node.right)
                node = self._rotate_left(node)
            return node, best

        self.root, best = _pop_max(self.root)
        return best.key, best.value


def greedy_select_patches(candidate_rows: list[dict[str, object]], top_k: int, distance_penalty: float, distance_scale: float) -> list[dict[str, object]]:
    remaining = [dict(item) for item in candidate_rows]
    selected: list[dict[str, object]] = []
    while remaining and len(selected) < top_k:
        tree = AVLTree()
        for item in remaining:
            adjusted = float(item["final_score"])
            if selected:
                closest = min(abs(float(item["x0"]) - float(sel["x0"])) + abs(float(item["y0"]) - float(sel["y0"])) for sel in selected)
                diversity = 1.0 - distance_penalty * np.exp(-closest / max(distance_scale, 1.0))
                adjusted *= float(np.clip(diversity, 0.25, 1.0))
            item["greedy_score"] = adjusted
            tree.insert((adjusted, float(item["final_score"]), -int(item["source_patch_id"])), item)
        _key, best = tree.pop_max()
        if best is None:
            break
        selected.append(best)
        best_source_patch = int(best["source_patch_id"])
        remaining = [item for item in remaining if int(item["source_patch_id"]) != best_source_patch]
    return selected


def haar_high_score(gray: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    """Three-level Haar-like high-frequency energy from cached normalized pixels."""
    outputs = []
    for start in range(0, len(gray), batch_size):
        current = gray[start:start + batch_size].to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
        energy = torch.zeros(len(current), dtype=torch.float32, device=device)
        for level in range(3):
            low = F.avg_pool2d(current, kernel_size=2, stride=2)
            reconstructed = F.interpolate(low, size=current.shape[-2:], mode="nearest")
            energy += (current - reconstructed).abs().mean(dim=(1, 2, 3)) / float(level + 1)
            current = low
        outputs.append(energy.cpu().numpy())
    return np.concatenate(outputs)


class OutputWriter:
    def __init__(self, out: Path, shard_size: int, overwrite: bool, selection_strategy: str):
        self.out, self.shards, self.shard_size, self.overwrite = out, out / "shards", shard_size, overwrite
        self.selection_strategy = selection_strategy
        self.shards.mkdir(parents=True, exist_ok=True)
        self.shard_id, self.x, self.fields, self.images, self.patches = 0, [], {k: [] for k in ["y", "patient_id", "image_id", "view", "density", "difficult_negative_case", "coords", "split", "valid_patch"]}, [], []

    def add(self, row: pd.Series, selected: list[dict[str, object]], source_x: torch.Tensor) -> None:
        if self.x and len(self.x) + len(selected) > self.shard_size:
            self.flush()
        start = len(self.x)
        selected_scores = [float(item.get("greedy_score", item["final_score"])) for item in selected]
        mean_selection_score = float(np.mean([float(item["final_score"]) for item in selected])) if selected else 0.5
        mean_selection_weight = float(np.mean(selected_scores)) if selected_scores else 1.0
        for rank, item in enumerate(selected, start=1):
            self.x.append(source_x[int(item["source_patch_id"])] .clone())
            self.fields["y"].append(float(row.y)); self.fields["patient_id"].append(str(row.patient_id)); self.fields["image_id"].append(str(row.image_id))
            self.fields["view"].append(str(row.view)); self.fields["density"].append(str(row.density)); self.fields["difficult_negative_case"].append(bool(row.difficult_negative_case))
            self.fields["coords"].append((int(item["x0"]), int(item["y0"]))); self.fields["split"].append(str(row.split)); self.fields["valid_patch"].append(bool(item["valid_patch"]))
            self.patches.append({"patient_id": str(row.patient_id), "image_id": str(row.image_id), "patch_id": rank - 1, "source_patch_id": int(item["source_patch_id"]), "x0": int(item["x0"]), "y0": int(item["y0"]), "coords": f"({int(item['x0'])},{int(item['y0'])})", "hue_medio": float(item["hue_medio"]), "freq_roxo": float(item["freq_roxo"]), "freq_rosa": float(item["freq_rosa"]), "freq_azul": float(item["freq_azul"]), "freq_verde": float(item["freq_verde"]), "magenta_ratio": float(item["magenta_ratio"]), "magenta_minus_cold": float(item["magenta_minus_cold"]), "fuzzy_color_score": float(item["fuzzy_color_score"]), "fuzzy_spectral_score": float(item.get("fuzzy_spectral_score", 0.5)), "gradiente_score": float(item["gradiente_score"]), "entropia_score": float(item["entropia_score"]), "fft_high_score": float(item.get("fft_high_score", 0.0)), "fft_low_raw": float(item.get("fft_low_raw", 0.0)), "fft_high_raw": float(item.get("fft_high_raw", 0.0)), "high_low_ratio_score": float(item.get("high_low_ratio_score", 0.0)), "high_low_ratio_raw": float(item.get("high_low_ratio_raw", 0.0)), "wavelet_high_score": float(item["wavelet_high_score"]), "wavelet_high_raw": float(item.get("wavelet_high_raw", 0.0)), "final_score": float(item["final_score"]), "greedy_score": float(item.get("greedy_score", item["final_score"])), "regras_ativas": json.dumps(item["regras_ativas"], ensure_ascii=True), "selected_rank": rank, "valid_patch": bool(item["valid_patch"]), "split": str(row.split), "view": str(row.view), "density": str(row.density), "cancer": int(row.y), "selection_used_cancer": False})
        self.images.append({"shard_id": self.shard_id, "shard_path": str(self.shards / f"shard_{self.shard_id:04d}.pt"), "start": start, "count": len(selected), "valid_patches": int(sum(bool(x["valid_patch"]) for x in selected)), "patient_id": str(row.patient_id), "image_id": str(row.image_id), "image_path": str(row.image_path), "y": int(row.y), "view": str(row.view), "density": str(row.density), "laterality": str(row.laterality), "difficult_negative_case": bool(row.difficult_negative_case), "split": str(row.split), "selection_strategy": self.selection_strategy, "selection_score": mean_selection_score, "selection_weight": mean_selection_weight})

    def flush(self) -> None:
        if not self.x:
            return
        path = self.shards / f"shard_{self.shard_id:04d}.pt"
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite or use a different output directory.")
        torch.save({"x": torch.stack(self.x), "y": torch.tensor(self.fields["y"], dtype=torch.float32), "patient_id": self.fields["patient_id"], "image_id": self.fields["image_id"], "view": self.fields["view"], "density": self.fields["density"], "difficult_negative_case": torch.tensor(self.fields["difficult_negative_case"], dtype=torch.bool), "coords": torch.tensor(self.fields["coords"], dtype=torch.int32), "split": self.fields["split"], "valid_patch": torch.tensor(self.fields["valid_patch"], dtype=torch.bool)}, path)
        self.shard_id += 1; self.x.clear()
        for field in self.fields.values():
            field.clear()
        gc.collect()


def render_comparison(source_x: torch.Tensor, candidates: list[dict[str, object]], selected: list[dict[str, object]], lut: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, max(len(candidates), len(selected)), figsize=(2.2 * max(len(candidates), len(selected)), 4.8))
    for col, item in enumerate(candidates):
        axes[0, col].imshow(apply_lut(source_x[int(item["source_patch_id"]), 0].float().numpy(), lut))
        axes[0, col].set_title(f"orig p{item['source_patch_id']}\n{item['final_score']:.3f}")
    for col, item in enumerate(selected):
        axes[1, col].imshow(apply_lut(source_x[int(item["source_patch_id"]), 0].float().numpy(), lut))
        axes[1, col].set_title(f"sel p{item['source_patch_id']}\n{item['final_score']:.3f}")
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].set_ylabel("antes"); axes[1, 0].set_ylabel("depois")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.shard_size < args.top_k:
        raise ValueError("--top-k must be >= 1 and --shard-size must hold one selected bag.")
    device = torch.device("cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    overlay_dir = out / "patch_comparisons"; overlay_dir.mkdir(parents=True, exist_ok=True)
    lut = load_lut(Path(args.lut_file)); metrics = ChromaticMetrics(lut, device)
    manifest = restore_metadata(pd.read_csv(args.manifest), args.metadata_csv)
    writer = OutputWriter(out, args.shard_size, args.overwrite, args.selection_strategy)
    overlays_done = 0
    print(f"device={device} mode={args.ranking_mode} images={len(manifest)} input_candidates={int(manifest['count'].sum())} top_k={args.top_k}")
    for shard_path, frame in tqdm(manifest.groupby("shard_path", sort=False), desc="cached patches -> fuzzy shards"):
        load_options = {"map_location": "cpu", "weights_only": False}
        if not args.no_mmap:
            load_options["mmap"] = True
        payload = torch.load(shard_path, **load_options)
        color = metrics.compute(payload["x"][:, 0], args.metric_batch_size)
        gradient = payload["x"][:, 1].float().mean(dim=(1, 2)).numpy()
        entropy = payload["x"][:, 4].float().mean(dim=(1, 2)).numpy()
        wavelet = haar_high_score(payload["x"][:, 0], device, args.metric_batch_size)
        fft_low = payload["x"][:, 5].float().mean(dim=(1, 2)).numpy()
        fft_high = payload["x"][:, 6].float().mean(dim=(1, 2)).numpy()
        high_low = fft_high / np.maximum(fft_low, 1e-8)
        for row in frame.itertuples(index=False):
            start, count = int(row.start), int(row.count); valid = payload["valid_patch"][start:start + count].bool().numpy()
            valid_ids = np.flatnonzero(valid)
            if len(valid_ids) == 0:
                valid_ids = np.array([0])
            g, e, w = normalized(gradient[start:start + count][valid_ids]), normalized(entropy[start:start + count][valid_ids]), normalized(wavelet[start:start + count][valid_ids])
            fh = normalized(fft_high[start:start + count][valid_ids]); hl = normalized(high_low[start:start + count][valid_ids])
            candidate_rows = []
            for local_i, source_id in enumerate(valid_ids):
                values = {"hue_medio": float(color[start + source_id, 0]), **{f"freq_{name}": float(color[start + source_id, index + 1]) for index, name in enumerate(COLOR_NAMES)}}
                values["magenta_ratio"] = (values["freq_roxo"] + values["freq_rosa"]) / (values["freq_azul"] + values["freq_verde"] + 1e-8)
                values["magenta_minus_cold"] = (values["freq_roxo"] + values["freq_rosa"]) - (values["freq_azul"] + values["freq_verde"] + values["freq_ciano"])
                fuzzy, active = fuzzy_color_score(values, row.density, row.view)
                spectral_values = {"gradiente_score": float(g[local_i]), "entropia_score": float(e[local_i]), "wavelet_high_score": float(w[local_i]), "fft_high_score": float(fh[local_i]), "high_low_ratio_score": float(hl[local_i])}
                spectral, spectral_rules = fuzzy_spectral_score(spectral_values)
                if args.ranking_mode == "spectral":
                    final = 0.25 * fuzzy + 0.25 * spectral + 0.15 * g[local_i] + 0.15 * e[local_i] + 0.10 * w[local_i] + 0.10 * fh[local_i]
                    active = active + spectral_rules
                else:
                    final = 0.40 * fuzzy + 0.25 * g[local_i] + 0.20 * e[local_i] + 0.15 * w[local_i]
                coords = payload["coords"][start + int(source_id)]
                candidate_rows.append({**values, **spectral_values, "source_patch_id": int(source_id), "x0": int(coords[0]), "y0": int(coords[1]), "fuzzy_color_score": fuzzy, "fuzzy_spectral_score": spectral, "fft_low_raw": float(fft_low[start + source_id]), "fft_high_raw": float(fft_high[start + source_id]), "high_low_ratio_raw": float(high_low[start + source_id]), "wavelet_high_raw": float(wavelet[start + source_id]), "final_score": float(final), "valid_patch": True, "regras_ativas": active})
            if args.selection_strategy == "greedy-avl":
                selected = greedy_select_patches(candidate_rows, args.top_k, args.greedy_distance_penalty, args.greedy_distance_scale)
            else:
                selected = sorted(candidate_rows, key=lambda item: item["final_score"], reverse=True)[:args.top_k]
            while len(selected) < args.top_k:
                duplicate = dict(selected[-1]); duplicate["valid_patch"] = False; selected.append(duplicate)
            writer.add(pd.Series(row._asdict()), selected, payload["x"][start:start + count])
            if overlays_done < args.overlay_images:
                render_comparison(payload["x"][start:start + count], candidate_rows, selected, lut, overlay_dir / f"{row.patient_id}_{row.image_id}_before_after.png")
                overlays_done += 1
        del payload, color, gradient, entropy, wavelet, fft_low, fft_high, high_low
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    writer.flush()
    image_manifest = pd.DataFrame(writer.images); patch_manifest = pd.DataFrame(writer.patches)
    image_manifest.to_csv(out / "manifest.csv", index=False); patch_manifest.to_csv(out / "selected_patches_manifest.csv", index=False)
    shutil.copy2(args.lut_file, out / "alizams_luts_usada.h")
    formula = "0.25*fuzzy_color_score + 0.25*fuzzy_spectral_score + 0.15*gradiente_score + 0.15*entropia_score + 0.10*wavelet_high_score + 0.10*fft_high_score" if args.ranking_mode == "spectral" else "0.40*fuzzy_color_score + 0.25*gradiente_score + 0.20*entropia_score + 0.15*wavelet_high_score"
    parameters = {"method": f"fuzzy {args.ranking_mode} patch ranking from precomputed shards", "ranking_mode": args.ranking_mode, "selection_strategy": args.selection_strategy, "greedy_distance_penalty": args.greedy_distance_penalty, "greedy_distance_scale": args.greedy_distance_scale, "input_manifest": str(args.manifest), "output_manifest": str(out / "manifest.csv"), "lut_file": str(args.lut_file), "lut_name": "AlizaMS RainbowB 1536 black_rainbow_lut", "device": str(device), "input_mmap": not args.no_mmap, "input_candidates_per_image": sorted(manifest["count"].unique().astype(int).tolist()), "selected_top_k": args.top_k, "score_formula": formula, "spectral_features": ["entropy channel mean", "FFT high channel mean", "FFT low channel mean", "FFT high/low ratio", "three-level Haar high energy"] if args.ranking_mode == "spectral" else [], "structural_score_scaling": "per-image min-max among cached candidate patches; no labels", "wavelet_high_score": "offline three-level Haar detail energy computed from cached grayscale patch", "selection_uses_cancer_label": False, "dicom_read_during_selection": False, "candidate_limitation": "Selection is restricted to candidate patches already present in the source shards.", "patient_split_leakage": bool(set(image_manifest.loc[image_manifest.split == 'train', 'patient_id']) & set(image_manifest.loc[image_manifest.split == 'test', 'patient_id'])), "images": int(len(image_manifest)), "patches": int(len(patch_manifest)), "shards": int(image_manifest.shard_id.nunique())}
    if parameters["patient_split_leakage"]:
        raise RuntimeError("Patient leakage detected in fuzzy-selected output.")
    (out / "selection_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    print(json.dumps(parameters, indent=2))


if __name__ == "__main__":
    main()
