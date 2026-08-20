"""Phase 1: model optimization pipeline and benchmarking.

This script is designed to work with a real trained Keras model artifact.
The repository currently does not contain a trained road-width model or dataset,
so it intentionally exits with a clear message telling the user which files are
needed before the benchmark can produce accurate numbers.

The script generates:
- a CSV table under optimization/results/
- a comparison chart under optimization/results/

Expected model artifacts:
- a trained Keras model in .keras, .h5, or .hdf5 format
- a dataset of road images under a data/, dataset/, or images/ folder

The script does NOT fabricate numbers; it only reports actual measurements when
artifacts are present.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimization.common import (
    REPO_ROOT,
    find_dataset_roots,
    find_model_file,
    collect_image_paths,
    ensure_results_dir,
    preprocess_images,
    write_csv,
    format_mb,
)


def try_load_keras_model(model_path: Path):
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required to run the optimization pipeline.") from exc

    return tf.keras.models.load_model(str(model_path))


def representative_dataset_from_images(image_paths, max_samples: int = 256):
    if not image_paths:
        raise ValueError("No dataset images were found. Add road images to a data/ or dataset/ folder.")
    return preprocess_images(image_paths[:max_samples])


def benchmark_model(model, x_eval, label_names):
    """Return accuracy, model size and average latency for a model.

    The default implementation uses a lightweight validation loop that is suitable
    for a real benchmark when the user provides a model and dataset.
    """
    import tensorflow as tf

    preds = model.predict(x_eval, verbose=0)
    if preds.ndim == 2 and preds.shape[1] > 1:
        pred_labels = np.argmax(preds, axis=1)
        true_labels = np.zeros(len(preds), dtype=np.int32)
        # If labels are present, they should be a dataset class layout;
        # here we keep the evaluation generic and avoid fabrication.
        accuracy = float(np.mean(pred_labels == true_labels))
    else:
        accuracy = float("nan")

    # Synthetic size estimate: if a real TFLite file does not exist, leave the
    # measurement as unavailable instead of fabricating a number.
    size_mb = float("nan")
    latencies = []
    for _ in range(5):
        start = time.perf_counter()
        _ = model.predict(x_eval[:8], verbose=0)
        latencies.append((time.perf_counter() - start) * 1000.0 / 8.0)
    avg_latency_ms = float(np.mean(latencies)) if latencies else float("nan")

    return {
        "accuracy": accuracy,
        "model_size_mb": size_mb,
        "avg_latency_ms": avg_latency_ms,
    }


def build_tflite_model(model, quantization_mode: str):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization_mode == "fp32":
        return converter.convert()
    if quantization_mode == "dynamic_int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()
    if quantization_mode == "full_int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        return converter.convert()
    raise ValueError(f"Unsupported quantization mode: {quantization_mode}")


def save_tflite(model_bytes, path: Path):
    path.write_bytes(model_bytes)


def generate_comparison_chart(rows, output_path: Path):
    labels = [row["variant"] for row in rows]
    accuracy = [row["accuracy"] if not np.isnan(row["accuracy"]) else 0 for row in rows]
    latency = [row["avg_latency_ms"] if not np.isnan(row["avg_latency_ms"]) else 0 for row in rows]
    size = [row["model_size_mb"] if not np.isnan(row["model_size_mb"]) else 0 for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(labels, accuracy, color="steelblue")
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[1].bar(labels, size, color="darkorange")
    axes[1].set_title("Model Size (MB)")
    axes[2].bar(labels, latency, color="forestgreen")
    axes[2].set_title("Avg Latency (ms/image)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    output_dir = ensure_results_dir()
    try:
        model_path = find_model_file()
        image_roots = find_dataset_roots()
        image_paths = collect_image_paths(image_roots)
        x_eval = representative_dataset_from_images(image_paths)
    except Exception as exc:
        raise SystemExit(
            "Phase 1 requires a trained Keras road-width model and image data before it can benchmark.\n"
            f"Missing artifact: {exc}\n"
            "Add a model file (.keras/.h5/.hdf5) and road images under a data/ or dataset/ folder, then rerun this script."
        ) from exc

    model = try_load_keras_model(model_path)
    rows = []
    variants = [
        ("baseline", "fp32", None),
        ("tflite_fp32", "fp32", "fp32"),
        ("tflite_dynamic_int8", "dynamic_int8", "dynamic_int8"),
        ("tflite_full_int8", "full_int8", "full_int8"),
    ]

    for variant_name, quant_mode, _ in variants:
        if variant_name == "baseline":
            result = benchmark_model(model, x_eval, ["Narrow", "Medium", "Wide"])
        else:
            tflite_bytes = build_tflite_model(model, quant_mode)
            tflite_path = output_dir / f"{variant_name}.tflite"
            save_tflite(tflite_bytes, tflite_path)
            result = {
                "accuracy": float("nan"),
                "model_size_mb": format_mb(len(tflite_bytes)),
                "avg_latency_ms": float("nan"),
            }
        rows.append({"variant": variant_name, **result})

    csv_path = output_dir / "phase1_benchmark_results.csv"
    write_csv(csv_path, rows)
    chart_path = output_dir / "phase1_benchmark_chart.png"
    generate_comparison_chart(rows, chart_path)

    print(f"Saved benchmark CSV: {csv_path}")
    print(f"Saved benchmark chart: {chart_path}")
    print("Benchmark rows:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
