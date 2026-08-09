#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from features_pixelwise import read_dicom_gray_with_metadata

try:
    import alizams_color_omp
except ImportError:
    alizams_color_omp = None


COLOR_BINS = [
    ("vermelho", "#e53935"),
    ("laranja", "#fb8c00"),
    ("amarelo", "#fdd835"),
    ("verde", "#43a047"),
    ("ciano", "#00acc1"),
    ("azul", "#1e88e5"),
    ("roxo", "#8e24aa"),
    ("rosa", "#d81b60"),
]
LUT_NAME = "black_rainbow_lut"
LUT_LABEL = "AlizaMS RainbowB 1536"


def parse_args() -> argparse.Namespace:
    root = Path("/media/angualberto/423ff97c-bf80-4bc9-9621-cf3ec081fd25")
    project = root / "iaimgem"
    p = argparse.ArgumentParser(description="Compare AlizaMS RainbowB colors in RSNA DICOM cancer vs non-cancer images.")
    p.add_argument("--csv", default=str(root / "rsna_kaggle_oficial/train.csv"))
    p.add_argument("--images-dir", default=str(root / "rsna_kaggle_oficial/train_images"))
    p.add_argument("--lut-file", default=str(project / "sistema_integrado_rsna_csv_treino/alizams_rainbowb_cores_todas_imagens/alizams_luts.h"))
    p.add_argument("--out-dir", default=str(project / "saida_analise_cores_alizams_dicom_rsna"))
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--backend", choices=["auto", "python", "fortran_openmp"], default="auto")
    p.add_argument("--max-images", type=int, default=0, help="Development limit, stratified by cancer label; zero processes all images.")
    p.add_argument("--save-filtered", action="store_true", help="Save pseudocolor PNGs; disabled by default for the full dataset.")
    p.add_argument("--panel-per-class", type=int, default=6)
    return p.parse_args()


def load_lut(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(rf"constexpr unsigned char {LUT_NAME}\[[^\]]+\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not match:
        raise RuntimeError(f"LUT {LUT_NAME} not found in {path}")
    values = np.asarray([int(value) for value in re.findall(r"\d+", match.group(1))], dtype=np.uint8)
    if values.size % 3:
        raise RuntimeError(f"Invalid LUT size: {values.size}")
    return values.reshape(-1, 3)


def resize_image(gray: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(gray.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def breast_mask(gray: np.ndarray) -> np.ndarray:
    mask = gray > max(0.03, float(np.quantile(gray, 0.08)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = labels == largest
    return clean if clean.mean() >= 0.01 else mask


def apply_lut(gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
    index = np.clip(np.floor(gray * len(lut)).astype(np.int32), 0, len(lut) - 1)
    return lut[index]


def color_metrics(rgb: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], str, str]:
    pixels = rgb[mask]
    if pixels.size == 0:
        pixels = rgb.reshape(-1, 3)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0].astype(np.float32) * 2.0
    sat = hsv[:, 1].astype(np.float32) / 255.0
    val = hsv[:, 2].astype(np.float32) / 255.0
    weight = np.clip(sat * val, 1e-4, None)
    total = float(weight.sum() + 1e-12)
    ranges = {
        "vermelho": (hue < 15) | (hue >= 345),
        "laranja": (hue >= 15) & (hue < 45),
        "amarelo": (hue >= 45) & (hue < 75),
        "verde": (hue >= 75) & (hue < 165),
        "ciano": (hue >= 165) & (hue < 195),
        "azul": (hue >= 195) & (hue < 255),
        "roxo": (hue >= 255) & (hue < 300),
        "rosa": (hue >= 300) & (hue < 345),
    }
    metrics = {
        "hue_medio_graus": float(np.average(hue, weights=weight)),
        "saturacao_media": float(np.average(sat, weights=weight)),
        "brilho_medio": float(np.average(val, weights=weight)),
    }
    for color, selected in ranges.items():
        metrics[f"freq_{color}"] = float(weight[selected].sum() / total)
    ordered = sorted([name for name, _hex in COLOR_BINS], key=lambda name: metrics[f"freq_{name}"], reverse=True)
    return metrics, ordered[0], ordered[1]


def lut_profile(lut: np.ndarray) -> tuple[np.ndarray, ...]:
    hsv = cv2.cvtColor(lut.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0].astype(np.float64) * 2.0
    sat = hsv[:, 1].astype(np.float64) / 255.0
    val = hsv[:, 2].astype(np.float64) / 255.0
    weight = np.clip(sat * val, 1e-4, None)
    bins = np.zeros(len(lut), dtype=np.int32)
    selections = [
        (hue < 15) | (hue >= 345), (hue >= 15) & (hue < 45),
        (hue >= 45) & (hue < 75), (hue >= 75) & (hue < 165),
        (hue >= 165) & (hue < 195), (hue >= 195) & (hue < 255),
        (hue >= 255) & (hue < 300), (hue >= 300) & (hue < 345),
    ]
    for index, selection in enumerate(selections, start=1):
        bins[selection] = index
    return bins, hue, sat, val, weight


def color_metrics_from_gray(gray: np.ndarray, mask: np.ndarray, lut: np.ndarray, backend: str) -> tuple[dict[str, float], str, str]:
    if backend == "python":
        return color_metrics(apply_lut(gray, lut), mask)
    if alizams_color_omp is None:
        raise RuntimeError("Fortran OpenMP backend requested but alizams_color_omp is not compiled")
    bins, hue, sat, val, weight = lut_profile(lut)
    result = alizams_color_omp.alizams_metrics_omp(
        gray.ravel().astype(np.float32), mask.ravel().astype(np.int32), bins, hue, sat, val, weight,
        gray.size, len(lut),
    )
    frequencies, hue_mean, sat_mean, val_mean = result
    metrics = {
        "hue_medio_graus": float(hue_mean), "saturacao_media": float(sat_mean), "brilho_medio": float(val_mean),
        **{f"freq_{name}": float(frequencies[index]) for index, (name, _hex) in enumerate(COLOR_BINS)},
    }
    ordered = sorted([name for name, _hex in COLOR_BINS], key=lambda name: metrics[f"freq_{name}"], reverse=True)
    return metrics, ordered[0], ordered[1]


def label_name(label: int) -> str:
    return "cancer_verdadeiro" if int(label) == 1 else "sem_cancer"


def process_record(record: dict[str, object], lut: np.ndarray, image_size: int, filtered_dir: str | None, backend: str) -> dict[str, object]:
    path = Path(str(record["image_path"]))
    try:
        gray, dicom_meta = read_dicom_gray_with_metadata(path)
        gray = resize_image(gray, image_size)
        metrics, primary, secondary = color_metrics_from_gray(gray, breast_mask(gray), lut, backend)
        output_path = ""
        if filtered_dir:
            rgb = apply_lut(gray, lut)
            output = Path(filtered_dir) / label_name(int(record["cancer"])) / f"{record['patient_id']}_{record['image_id']}_alizams_rainbowb.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(output)
            output_path = str(output)
        return {
            **record,
            "classe": label_name(int(record["cancer"])),
            "lut": LUT_LABEL,
            "cor_predominante": primary,
            "segunda_cor": secondary,
            "filtered_path": output_path,
            "monochrome1_inverted": bool(dicom_meta["monochrome1_inverted"]),
            "windowing_applied": bool(dicom_meta["windowing_applied"]),
            **metrics,
        }
    except Exception as exc:
        return {**record, "classe": "erro", "erro": str(exc)}


def select_source(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.csv, dtype=str)
    # Support two CSV formats:
    # 1) RSNA-style with columns patient_id,image_id,cancer
    # 2) CBIS-DDMS pixelwise style with image_path,cancer
    if "image_path" in frame.columns:
        required = {"image_path", "cancer"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        frame["image_path"] = frame["image_path"].astype(str)
        # Derive patient_id and image_id from the stored path (works for CBIS-DDSM layout)
        frame["patient_id"] = frame["image_path"].apply(lambda p: Path(str(p)).parts[-4])
        frame["image_id"] = frame["image_path"].apply(lambda p: Path(str(p)).stem)
        frame["cancer"] = frame["cancer"].astype(int)
    else:
        required = {"patient_id", "image_id", "cancer"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        frame = frame.astype({"patient_id": str, "image_id": str, "cancer": int})
        frame["cancer"] = frame["cancer"].astype(int)
        frame["image_path"] = frame.apply(lambda row: str(Path(args.images_dir) / str(row.patient_id) / f"{row.image_id}.dcm"), axis=1)
    if args.max_images > 0:
        positives = frame[frame.cancer == 1].head(max(1, args.max_images // 2))
        negatives = frame[frame.cancer == 0].head(max(1, args.max_images - len(positives)))
        frame = pd.concat([positives, negatives], ignore_index=True)
    return frame


def process_all(frame: pd.DataFrame, args: argparse.Namespace, lut: np.ndarray) -> pd.DataFrame:
    filtered_dir = str(Path(args.out_dir) / "imagens_filtradas_256") if args.save_filtered else None
    records = frame.to_dict("records")
    results: list[dict[str, object]] = []
    start = time.perf_counter()
    if args.workers <= 1:
        for index, record in enumerate(records, 1):
            results.append(process_record(record, lut, args.image_size, filtered_dir, args.backend))
            if index % 100 == 0 or index == len(records):
                print_progress(index, len(records), start)
        return pd.DataFrame(results)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = set()
        iterator = iter(records)
        for _ in range(min(len(records), args.workers * 2)):
            pending.add(executor.submit(process_record, next(iterator), lut, args.image_size, filtered_dir, args.backend))
        done_count = 0
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
                done_count += 1
                try:
                    pending.add(executor.submit(process_record, next(iterator), lut, args.image_size, filtered_dir, args.backend))
                except StopIteration:
                    pass
                if done_count % 100 == 0 or done_count == len(records):
                    print_progress(done_count, len(records), start)
    return pd.DataFrame(results)


def print_progress(done: int, total: int, start: float) -> None:
    elapsed = time.perf_counter() - start
    rate = done / max(elapsed, 1e-9)
    eta = (total - done) / max(rate, 1e-9)
    print(f"processadas={done}/{total} tempo={elapsed:.1f}s taxa={rate:.2f} imagens/s eta={eta/60:.1f}min", flush=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    freq_columns = [f"freq_{color}" for color, _hex in COLOR_BINS]
    rows = []
    for class_name, group in frame.groupby("classe"):
        means = group[freq_columns].mean()
        rows.append({
            "classe": class_name,
            "n_imagens": int(len(group)),
            "cor_predominante_grupo": means.idxmax().replace("freq_", ""),
            "hue_medio_graus": float(group["hue_medio_graus"].mean()),
            **{f"{column}_media": float(means[column]) for column in freq_columns},
        })
    return pd.DataFrame(rows)


def save_plots(frame: pd.DataFrame, out: Path) -> None:
    colors = [name for name, _hex in COLOR_BINS]
    hex_colors = [hex_code for _name, hex_code in COLOR_BINS]
    frequencies = [f"freq_{color}" for color in colors]
    average = frame.groupby("classe")[frequencies].mean().T
    average.index = colors
    ax = average.plot(kind="bar", figsize=(12, 6), width=0.82)
    ax.set(title="Cores AlizaMS RainbowB por classe - DICOM RSNA", xlabel="Cor", ylabel="Frequencia media ponderada")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(out / "barras_cores_cancer_vs_sem_cancer.png", dpi=170); plt.close()
    dominant = pd.crosstab(frame["classe"], frame["cor_predominante"], normalize="index").reindex(columns=colors, fill_value=0.0)
    ax = dominant.plot(kind="bar", stacked=True, figsize=(11, 5.5), color=hex_colors)
    ax.set(title="Cor predominante por imagem - DICOM RSNA", xlabel="Classe", ylabel="Proporcao de imagens")
    ax.legend(title="Cor", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(out / "pilha_cor_predominante_por_classe.png", dpi=170); plt.close()


def save_panels(frame: pd.DataFrame, lut: np.ndarray, out: Path, count: int, image_size: int) -> None:
    selected = []
    for class_name in ["cancer_verdadeiro", "sem_cancer"]:
        group = frame[frame.classe == class_name].copy()
        group["freq_dominante"] = group.apply(lambda row: row[f"freq_{row.cor_predominante}"], axis=1)
        selected.extend(group.sort_values("freq_dominante", ascending=False).head(count).to_dict("records"))
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 2, figsize=(8, max(4, 2.3 * len(selected))))
    axes = np.atleast_2d(axes)
    for axis, record in zip(axes, selected):
        gray, _meta = read_dicom_gray_with_metadata(record["image_path"])
        gray = resize_image(gray, image_size)
        rgb = apply_lut(gray, lut)
        axis[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axis[0].set_title(f"{record['classe']} | original")
        axis[1].imshow(rgb)
        axis[1].set_title(f"RainbowB | dominante: {record['cor_predominante']}")
        for item in axis:
            item.axis("off")
    plt.tight_layout(); fig.savefig(out / "painel_top_cor_predominante.png", dpi=160); plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lut_path = Path(args.lut_file)
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT source not found: {lut_path}")
    shutil.copy2(lut_path, out / "alizams_luts_usada.h")
    lut = load_lut(lut_path)
    if args.backend == "auto":
        args.backend = "fortran_openmp" if alizams_color_omp is not None else "python"
    if args.backend == "fortran_openmp" and alizams_color_omp is None:
        raise RuntimeError("Compile the OpenMP backend first: bash build_alizams_fortran_openmp.sh")
    source = select_source(args)
    print(f"DICOM a processar: {len(source)} | workers: {args.workers} | backend: {args.backend} | LUT: {LUT_LABEL}", flush=True)
    per_image = process_all(source, args, lut)
    ok = per_image[per_image.classe != "erro"].copy()
    errors = per_image[per_image.classe == "erro"].copy()
    if ok.empty:
        raise RuntimeError("No DICOM images processed successfully")
    summary = summarize(ok)
    per_image.to_csv(out / "cores_alizams_por_imagem.csv", index=False)
    summary.to_csv(out / "resumo_cores_por_classe.csv", index=False)
    if not errors.empty:
        errors.to_csv(out / "erros_leitura_dicom.csv", index=False)
    save_plots(ok, out)
    save_panels(ok, lut, out, args.panel_per_class, args.image_size)
    payload = {
        "method": "LUT AlizaMS black_rainbow_lut applied to normalized DICOM pixels inside breast mask",
        "scientific_note": "These colors are pseudocolors derived from grayscale intensity, not biological tissue colors or ground truth.",
        "input_csv": str(args.csv), "images_dir": str(args.images_dir), "lut_file": str(lut_path), "backend": args.backend,
        "lut_label": LUT_LABEL, "lut_size": int(len(lut)), "image_size": args.image_size,
        "images_requested": int(len(source)), "images_success": int(len(ok)), "read_errors": int(len(errors)),
        "summary": summary.to_dict("records"),
    }
    (out / "resumo_cores_alizams_dicom.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nResumo das cores por classe:")
    print(summary.to_string(index=False))
    print(f"\nSaida: {out}")


if __name__ == "__main__":
    main()
