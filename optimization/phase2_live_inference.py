"""Phase 2: live webcam inference using the fastest model from Phase 1.

This script is intentionally implemented as a standalone utility. It keeps the
existing static upload mode in app.py intact and adds a separate live mode that
reads from the default webcam and runs inference frame by frame.

Requirements before use:
- a trained Keras model artifact for the same road-width task
- a TFLite model exported from the optimization benchmark pipeline
- a webcam device available on the local machine

This script does not fabricate FPS or accuracy numbers; it measures them at run
and prints them live.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


def load_tflite_model(model_path: str):
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=model_path)


def run_live_inference(model_path: str, camera_index: int = 0, max_frames: int = 300):
    interpreter = load_tflite_model(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not access camera index {camera_index}. Check the webcam connection.")

    frame_count = 0
    start_time = time.perf_counter()
    fps_history = []

    print("Starting live inference. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_data = np.expand_dims(rgb, axis=0)

        if input_details[0]["dtype"] == np.float32:
            interpreter.set_tensor(input_details[0]["index"], input_data)
        else:
            quant_input = (input_data / 127.5 - 1.0).astype(np.float32)
            interpreter.set_tensor(input_details[0]["index"], quant_input)

        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        pred_idx = int(np.argmax(output[0]))
        class_names = ["Narrow", "Medium", "Wide"]
        label = class_names[pred_idx]

        fps = 1.0 / max((time.perf_counter() - start_time) / max(frame_count, 1), 1e-6)
        fps_history.append(fps)
        cv2.putText(frame, f"Pred: {label} FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Live Road Width Inference", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_count >= max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Processed frames: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Last 10 FPS samples: {[round(v, 2) for v in fps_history[-10:]]}")


if __name__ == "__main__":
    model_path = str(Path(__file__).resolve().parent / "results" / "tflite_fp32.tflite")
    if not Path(model_path).exists():
        raise SystemExit(
            "No TFLite model is available yet. Run phase1_benchmark.py after adding a trained model artifact."
        )
    run_live_inference(model_path)
