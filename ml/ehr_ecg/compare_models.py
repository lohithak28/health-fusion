
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "features" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Previously obtained test-set results.
# These are recorded results, not recalculated by this script.
results = [
    {
        "Model": "Majority baseline",
        "Accuracy": 0.87097,
        "Balanced accuracy": 0.50000,
        "Precision": 0.00000,
        "Recall": 0.00000,
        "F1-score": 0.00000,
        "AUROC": 0.50000,
    },
    {
        "Model": "EHR-only",
        "Accuracy": 0.12900,
        "Balanced accuracy": 0.50000,
        "Precision": 0.12900,
        "Recall": 1.00000,
        "F1-score": 0.22860,
        "AUROC": 0.74540,
    },
    {
        "Model": "ECG-only",
        "Accuracy": 0.61290,
        "Balanced accuracy": 0.56480,
        "Precision": 0.16670,
        "Recall": 0.50000,
        "F1-score": 0.25000,
        "AUROC": 0.53700,
    },
    {
        "Model": "EHR-ECG fusion",
        "Accuracy": 0.80645,
        "Balanced accuracy": 0.46296,
        "Precision": 0.00000,
        "Recall": 0.00000,
        "F1-score": 0.00000,
        "AUROC": 0.46296,
    },
]

df = pd.DataFrame(results)

csv_path = RESULTS_DIR / "model_comparison.csv"
df.to_csv(csv_path, index=False)

print("\nTest-set model comparison:")
print(df.round(4).to_string(index=False))
print(f"\nSaved CSV: {csv_path}")

# Plot selected metrics. AUROC is plotted separately because
# it measures ranking rather than threshold-based classification.
threshold_metrics = [
    "Accuracy",
    "Balanced accuracy",
    "Precision",
    "Recall",
    "F1-score",
]

ax = df.set_index("Model")[threshold_metrics].plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1),
    rot=15,
)

ax.set_title("Test-set comparison: EHR, ECG and fusion models")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
chart_path = RESULTS_DIR / "model_comparison.png"
plt.savefig(chart_path, dpi=200, bbox_inches="tight")
plt.close()

# Separate AUROC chart
ax = df.set_index("Model")[["AUROC"]].plot(
    kind="bar",
    figsize=(8, 5),
    ylim=(0, 1),
    legend=False,
    rot=15,
)

ax.axhline(0.5, linestyle="--", label="Chance reference")
ax.set_title("Test-set AUROC comparison")
ax.set_ylabel("AUROC")
ax.set_xlabel("")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
auroc_path = RESULTS_DIR / "model_auroc.png"
plt.savefig(auroc_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved metrics chart: {chart_path}")
print(f"Saved AUROC chart: {auroc_path}")
print("\nNote: Results are preliminary; the test set has only 4 deaths.")
