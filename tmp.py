import numpy as np
import matplotlib.pyplot as plt

methods = ["CurVer", "GA", "NUTS"]

# Key numbers
time_s = np.array([4.943, 36.394, 878.821], dtype=float)
model_eval = np.array([20399, 70286, 20399], dtype=float)
sse_raw = np.array([5.45030e7, 5.45031e7, 5.45030e7], dtype=float)

# 1) Runtime (log scale)
plt.figure(figsize=(8, 4.5))
plt.bar(methods, time_s)
plt.yscale("log")
plt.xlabel("Method")
plt.ylabel("Total fit time (s) [log scale]")
plt.title("Runtime Comparison")
plt.grid(True, axis="y", which="both", linestyle="--", linewidth=0.7)
plt.tight_layout()
plt.show()

# 2) Model evaluation count
plt.figure(figsize=(8, 4.5))
plt.bar(methods, model_eval)
plt.xlabel("Method")
plt.ylabel("Model evaluations")
plt.title("Compute Cost Comparison (Model Evaluations)")
plt.grid(True, axis="y", linestyle="--", linewidth=0.7)
plt.tight_layout()
plt.show()

# 3) Accuracy (SSE_raw) — show relative difference vs CurVer (ppm)
rel_ppm = (sse_raw - sse_raw[0]) / sse_raw[0] * 1e6  # parts per million vs CurVer
plt.figure(figsize=(8, 4.5))
plt.bar(methods, rel_ppm)
plt.xlabel("Method")
plt.ylabel("ΔSSE_raw vs CurVer (ppm)")
plt.title("Accuracy Comparison (SSE_raw relative to CurVer)")
plt.grid(True, axis="y", linestyle="--", linewidth=0.7)
plt.tight_layout()
plt.show()