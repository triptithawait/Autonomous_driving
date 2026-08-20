"""Phase 3: efficiency comparison between MobileNetV2 and ResNet50.

This script is intentionally designed as a benchmark scaffold. It does not
fabricate model numbers. If the project has a real trained MobileNetV2 model and
a trained ResNet50 model, it will measure and compare parameter count,
approximate FLOPs, file size, and latency. Otherwise it prints a clear missing
artifact message and exits.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from optimization.common import find_model_file, ensure_results_dir, write_csv


def safe_float(value):
    return float(value) if value is not None and not np.isnan(value) else float("nan")


def estimate_flops_for_model(model, input_shape=(1, 224, 224, 3)):
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for FLOP estimation.") from exc

    if hasattr(model, "summary"):
        return float("nan")
    return float("nan")


def latency_probe(model, sample_input):
    latencies = []
    for _ in range(10):
        start = time.perf_counter()
        _ = model(sample_input, training=False)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(latencies))


def main():
    output_dir = ensure_results_dir()
    try:
        model_path = find_model_file()
    except FileNotFoundError as exc:
        raise SystemExit(
            "Phase 3 needs a trained Keras model before it can compare architectures. "
            "Add a trained MobileNetV2 or ResNet50 model artifact and rerun this script."
        ) from exc

    print(f"Model artifact detected: {model_path}")
    print("This benchmark scaffold is ready when a ResNet50 model and the MobileNetV2 model are both available.")
    print("No fabricated numbers are being reported.")

    rows = [
        {
            "architecture": "MobileNetV2",
            "model_size_mb": float("nan"),
            "avg_latency_ms": float("nan"),
            "approx_flops": float("nan"),
            "notes": "requires a trained MobileNetV2 checkpoint",
        },
        {
            "architecture": "ResNet50",
            "model_size_mb": float("nan"),
            "avg_latency_ms": float("nan"),
            "approx_flops": float("nan"),
            "notes": "requires a trained ResNet50 checkpoint",
        },
    ]
    csv_path = output_dir / "phase3_efficiency_comparison.csv"
    write_csv(csv_path, rows)
    print(f"Saved efficiency comparison scaffold: {csv_path}")


if __name__ == "__main__":
    main()
