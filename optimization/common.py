"""Shared utilities for model optimization and benchmarking.

This module intentionally keeps the implementation generic so it can run with a
real trained Keras/MobileNet model if one is present in the project. The scripts
are designed to fail gracefully with a clear message when the required training
artifacts are absent.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION_DIR = REPO_ROOT / "optimization"
RESULTS_DIR = OPTIMIZATION_DIR / "results"


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def find_model_file() -> Path:
    """Locate a trained Keras model in the repository.

    The project currently does not contain a trained model file. The script keeps
    the search generic so it can work as soon as the user adds a model artifact
    such as my_model.keras or my_model.h5.
    """
    search_dirs = [REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "artifacts", REPO_ROOT / "checkpoints"]
    candidates = ["*.keras", "*.h5", "*.hdf5", "*.weights.h5"]

    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in candidates:
            matches = sorted(directory.rglob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "No trained Keras model was found. Add a model file such as my_model.keras "
        "or my_model.h5 under the project and rerun the optimization script."
    )


def find_dataset_roots() -> List[Path]:
    """Locate image folders used for representative dataset generation."""
    roots = [
        REPO_ROOT,
        REPO_ROOT / "data",
        REPO_ROOT / "dataset",
        REPO_ROOT / "datasets",
        REPO_ROOT / "images",
        REPO_ROOT / "samples",
        REPO_ROOT / "training",
        REPO_ROOT / "validation",
    ]
    dataset_roots: List[Path] = []
    for root in roots:
        if root.exists():
            dataset_roots.append(root)
    return dataset_roots


def collect_image_paths(dataset_roots: Iterable[Path], extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    """Gather all image files under likely dataset directories."""
    image_paths: List[Path] = []
    seen = set()
    for root in dataset_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions and path not in seen:
                image_paths.append(path)
                seen.add(path)
    return image_paths


def load_label_mapping(label_file: Path | None = None) -> dict:
    if label_file is None:
        return {0: "Narrow", 1: "Medium", 2: "Wide"}
    with open(label_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_image_for_keras(image_path: Path, target_size=(224, 224)) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for image preprocessing.") from exc

    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    array = np.asarray(img, dtype=np.float32) / 255.0
    return array


def preprocess_images(image_paths: List[Path], target_size=(224, 224)) -> np.ndarray:
    if not image_paths:
        raise ValueError("No images were found in the project dataset folders.")
    batch = [preprocess_image_for_keras(path, target_size) for path in image_paths[:256]]
    return np.stack(batch, axis=0)


def format_mb(size_bytes: int) -> float:
    return round(size_bytes / (1024 * 1024), 3)
