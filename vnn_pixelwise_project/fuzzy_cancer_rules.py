#!/usr/bin/env python3
"""Interpretable fuzzy tendency layer for RSNA mammography experiments.

The output is a statistical tendency score, not a cancer diagnosis. Rules are
fixed before evaluation and must not be tuned on the held-out test split without
recording that calibration decision.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_SETS = {
    "hue_medio": {"baixo": (230.0, 238.0, 244.0), "medio": (238.0, 246.0, 254.0), "alto": (247.0, 254.0, 275.0)},
    "freq_roxo": {"baixo": (0.12, 0.20, 0.27), "medio": (0.18, 0.30, 0.43), "alto": (0.32, 0.45, 0.62)},
    "freq_rosa": {"baixo": (0.08, 0.15, 0.24), "medio": (0.14, 0.25, 0.39), "alto": (0.28, 0.42, 0.62)},
    "freq_azul": {"baixo": (0.12, 0.20, 0.30), "medio": (0.18, 0.30, 0.43), "alto": (0.32, 0.47, 0.64)},
    "freq_verde": {"baixo": (0.03, 0.08, 0.14), "medio": (0.06, 0.14, 0.25), "alto": (0.18, 0.30, 0.48)},
    "magenta_ratio": {"baixo": (0.65, 0.95, 1.20), "medio": (0.85, 1.20, 1.70), "alto": (1.35, 1.85, 2.60)},
    "roxo_menos_azul": {"baixo": (-0.18, -0.06, 0.03), "medio": (-0.05, 0.03, 0.14), "alto": (0.08, 0.19, 0.36)},
    "magenta_minus_cold": {"baixo": (-0.35, -0.14, 0.02), "medio": (-0.10, 0.04, 0.22), "alto": (0.12, 0.30, 0.55)},
    "gradiente_medio": {"baixo": (0.05, 0.18, 0.32), "medio": (0.18, 0.38, 0.62), "alto": (0.48, 0.72, 1.00)},
    "gradiente_top": {"baixo": (0.18, 0.48, 0.85), "medio": (0.45, 0.95, 1.75), "alto": (1.25, 2.10, 4.50)},
    "entropia_local": {"baixo": (2.2, 3.3, 4.2), "medio": (3.2, 4.3, 5.2), "alto": (4.6, 5.5, 6.5)},
    "wavelet_high": {"baixo": (0.001, 0.004, 0.009), "medio": (0.004, 0.010, 0.022), "alto": (0.016, 0.032, 0.075)},
    "fft_high": {"baixo": (0.015, 0.035, 0.065), "medio": (0.035, 0.080, 0.145), "alto": (0.105, 0.175, 0.30)},
    "high_low_ratio": {"baixo": (0.01, 0.04, 0.10), "medio": (0.05, 0.14, 0.28), "alto": (0.20, 0.42, 0.80)},
    "score_vnn": {"baixo": (0.20, 0.38, 0.52), "medio": (0.38, 0.55, 0.70), "alto": (0.62, 0.78, 0.93)},
    "score_cnn": {"baixo": (0.20, 0.38, 0.52), "medio": (0.38, 0.55, 0.70), "alto": (0.62, 0.78, 0.93)},
    "score_hybrid": {"baixo": (0.20, 0.38, 0.52), "medio": (0.38, 0.55, 0.70), "alto": (0.62, 0.78, 0.93)},
    "attention_max": {"baixo": (0.15, 0.25, 0.38), "medio": (0.25, 0.43, 0.62), "alto": (0.52, 0.70, 0.90)},
    "attention_entropy": {"baixo": (0.15, 0.32, 0.48), "medio": (0.32, 0.52, 0.70), "alto": (0.62, 0.80, 0.98)},
    "attention_concentration": {"baixo": (0.15, 0.27, 0.39), "medio": (0.28, 0.45, 0.62), "alto": (0.52, 0.70, 0.92)},
}

OUTPUT_SETS = {
    "baixa": ("trap", (0.0, 0.0, 0.22, 0.44)),
    "intermediaria": ("tri", (0.26, 0.50, 0.74)),
    "alta": ("trap", (0.56, 0.78, 1.0, 1.0)),
}

UNIVERSE = np.linspace(0.0, 1.0, 1001)

RULE_LABELS = {
    "R1": "magenta elevado com entropia e gradiente top elevados",
    "R2": "roxo e rosa elevados com azul baixo",
    "R3": "score VNN alto com atencao concentrada",
    "R4": "score hibrido alto com predominio magenta sobre cores frias",
    "R5": "detalhe espectral wavelet/FFT e entropia elevados",
    "R6": "magenta, gradiente top e entropia baixos",
    "R7": "density B/C reforca sinal magenta",
    "R8": "view MLO reforca score VNN elevado",
    "R9": "difficult negative com VNN alto sem suporte magenta reduz tendencia",
    "R10": "discordancia CNN baixa/VNN alta com suporte magenta",
}


def triangular(x: float, a: float, b: float, c: float) -> float:
    if not np.isfinite(x) or x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    return float((x - a) / (b - a) if x < b else (c - x) / (c - b))


def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    if not np.isfinite(x) or x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a == b and x <= b:
        return 1.0
    if c == d and x >= c:
        return 1.0
    return float((x - a) / (b - a) if x < b else (d - x) / (d - c))


OUTPUT_MEMBERSHIP = {
    "baixa": np.asarray([trapezoidal(x, *OUTPUT_SETS["baixa"][1]) for x in UNIVERSE]),
    "intermediaria": np.asarray([triangular(x, *OUTPUT_SETS["intermediaria"][1]) for x in UNIVERSE]),
    "alta": np.asarray([trapezoidal(x, *OUTPUT_SETS["alta"][1]) for x in UNIVERSE]),
}


def baixo(x: float, knots: tuple[float, float, float]) -> float:
    value = float(x)
    if not np.isfinite(value):
        return 0.0
    if value <= knots[1]:
        return 1.0
    if value >= knots[2]:
        return 0.0
    return float((knots[2] - value) / (knots[2] - knots[1]))


def medio(x: float, knots: tuple[float, float, float]) -> float:
    return triangular(float(x), *knots)


def alto(x: float, knots: tuple[float, float, float]) -> float:
    value = float(x)
    if not np.isfinite(value):
        return 0.0
    if value <= knots[0]:
        return 0.0
    if value >= knots[1]:
        return 1.0
    return float((value - knots[0]) / (knots[1] - knots[0]))


def membership(feature: str, level: str, value: Any, feature_sets: dict[str, Any] | None = None) -> float:
    sets = feature_sets or FEATURE_SETS
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    knots = tuple(sets[feature][level])
    return {"baixo": baixo, "medio": medio, "alto": alto}[level](number, knots)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _min(*values: float) -> float:
    return float(min(values))


def evaluate_fuzzy(features: dict[str, Any], parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    sets = (parameters or {}).get("feature_sets", FEATURE_SETS)
    m = lambda name, level: membership(name, level, features.get(name, np.nan), sets)
    density = str(features.get("density", "UNKNOWN")).upper()
    view = str(features.get("view", "UNKNOWN")).upper()
    rules = [
        ("R1", _min(m("magenta_ratio", "alto"), m("entropia_local", "alto"), m("gradiente_top", "alto")), "alta"),
        ("R2", _min(m("freq_roxo", "alto"), m("freq_rosa", "alto"), m("freq_azul", "baixo")), "alta"),
        ("R3", _min(m("score_vnn", "alto"), m("attention_concentration", "alto")), "alta"),
        ("R4", _min(m("score_hybrid", "alto"), m("magenta_minus_cold", "alto")), "alta"),
        ("R5", _min(m("wavelet_high", "alto"), m("fft_high", "alto"), m("entropia_local", "alto")), "media_alta"),
        ("R6", _min(m("magenta_ratio", "baixo"), m("gradiente_top", "baixo"), m("entropia_local", "baixo")), "baixa"),
        ("R7", (0.35 * m("magenta_ratio", "alto")) if density in {"B", "C"} else 0.0, "alta"),
        ("R8", (0.35 * m("score_vnn", "alto")) if view == "MLO" else 0.0, "alta"),
        ("R9", _min(1.0 if _bool(features.get("difficult_negative_case", False)) else 0.0, m("score_vnn", "alto"), m("magenta_ratio", "baixo")), "baixa"),
        ("R10", _min(m("score_cnn", "baixo"), m("score_vnn", "alto"), m("magenta_ratio", "alto")), "media_alta"),
    ]
    output = OUTPUT_MEMBERSHIP
    aggregate = np.zeros_like(UNIVERSE)
    active = []
    for rule_id, strength, consequence in rules:
        if strength <= 0:
            continue
        consequent = output[consequence] if consequence != "media_alta" else np.maximum(output["intermediaria"], output["alta"] * 0.75)
        aggregate = np.maximum(aggregate, np.minimum(strength, consequent))
        active.append({"regra": rule_id, "intensidade": float(strength), "consequencia": consequence, "descricao": RULE_LABELS[rule_id]})
    if aggregate.sum() <= 1e-12:
        score = 0.5
    else:
        score = float(np.sum(UNIVERSE * aggregate) / np.sum(aggregate))
    fuzzy_class = "baixa" if score < 0.34 else "intermediaria" if score < 0.67 else "alta"
    active.sort(key=lambda item: item["intensidade"], reverse=True)
    drivers = [item["descricao"] for item in active[:3]]
    explanation = f"Tendencia {fuzzy_class} ({score:.3f})"
    if drivers:
        explanation += " por " + "; ".join(drivers) + "."
    else:
        explanation += "; nenhuma regra apresentou ativacao relevante."
    return {"tendencia_cancer": score, "classe_fuzzy": fuzzy_class, "regras_ativas": active, "explicacao": explanation}


def parameters_payload() -> dict[str, Any]:
    return {
        "inference": "Mamdani", "and_operator": "min", "or_operator": "max", "aggregation": "max", "defuzzification": "centroid",
        "feature_sets": FEATURE_SETS, "output_sets": OUTPUT_SETS, "rules": RULE_LABELS,
        "intended_use": "statistical cancer tendency; not a medical diagnosis",
        "calibration": "fixed expert-defined rules; no test-set fitting performed by these scripts",
        "attention_note": "attention is an MIL weighting signal and is not segmentation ground truth",
    }


def save_parameters(path: str | Path) -> None:
    Path(path).write_text(json.dumps(parameters_payload(), indent=2), encoding="utf-8")
