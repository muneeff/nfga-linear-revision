import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "/content/drive/MyDrive/Colab_Projects/Cholera_Forecasting/data/processed"

FORECAST = os.path.join(BASE_DIR,"final_tables","forecast_table.csv")
ABLATION = os.path.join(BASE_DIR,"final_tables","ablation_table.csv")
WILCOXON = os.path.join(BASE_DIR,"wilcoxon_results.csv")

OUT_DIR = os.path.join(BASE_DIR,"paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# =====================================================
# FIGURE 2
# RMSE comparison
# =====================================================

df = pd.read_csv(FORECAST)

pivot = df.pivot(
    index="model",
    columns="dataset",
    values="RMSE_mean"
)

# ترتيب حسب أسوأ RMSE لكل نموذج
pivot["max_rmse"] = pivot.max(axis=1)

pivot = pivot.sort_values(
    by="max_rmse",
    ascending=False
)

pivot = pivot.drop(
    columns=["max_rmse"]
)

fig, ax = plt.subplots(
    figsize=(11,6)
)

pivot.plot(
    kind="bar",
    ax=ax,
    width=0.85
)

# Log scale
ax.set_yscale("log")

ax.set_ylabel(
    "RMSE (log scale)",
    fontsize=12
)

ax.set_xlabel(
    "Model",
    fontsize=12
)

ax.set_title(
    "Forecasting Performance Across Datasets",
    fontsize=14,
    fontweight="bold"
)

# Grid
ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# إظهار القيم فوق الأعمدة
for container in ax.containers:
    labels = []

    for v in container.datavalues:

        if v >= 10:
            labels.append(f"{v:.0f}")

        elif v >= 1:
            labels.append(f"{v:.2f}")

        else:
            labels.append(f"{v:.3f}")

    ax.bar_label(
        container,
        labels=labels,
        fontsize=6,
        padding=2
    )

plt.xticks(
    rotation=25,
    ha="right"
)

plt.legend(
    title="Dataset",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "figure2_rmse_comparison.png"
    ),
    bbox_inches="tight"
)

plt.close()

# =====================================================
# FIGURE 3
# Ablation RMSE
# =====================================================

ab = pd.read_csv(ABLATION)

x = np.arange(len(ab))
width = 0.35

fig, ax = plt.subplots(figsize=(7,5))

ax.bar(
    x-width/2,
    ab["NFGA_Core_RMSE"],
    width,
    label="NFGA-Core"
)

ax.bar(
    x+width/2,
    ab["NFGA_LINEAR_RMSE"],
    width,
    label="NFGA-LINEAR"
)

ax.set_xticks(x)
ax.set_xticklabels(ab["dataset"])

ax.set_ylabel("RMSE")
ax.set_title("Ablation Study")

ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "figure3_ablation_rmse.png"
    ),
    bbox_inches="tight"
)

plt.close()

# =====================================================
# FIGURE 4
# Improvement %
# =====================================================

fig, ax = plt.subplots(figsize=(6,4))

bars = ax.bar(
    ab["dataset"],
    ab["Improvement_%"]
)

ax.set_ylabel("Improvement (%)")
ax.set_title(
    "Improvement of NFGA-LINEAR over NFGA-Core"
)

for bar in bars:
    h = bar.get_height()

    ax.text(
        bar.get_x()+bar.get_width()/2,
        h+1,
        f"{h:.1f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "figure4_improvement.png"
    ),
    bbox_inches="tight"
)

plt.close()

# =====================================================
# FIGURE 5
# Wilcoxon (Ordered)
# =====================================================

# قراءة الملف
wil = pd.read_csv(WILCOXON)

# ترتيب البيانات حسب p_value تصاعديًا
#wil_sorted = wil.sort_values(by="p_value", ascending=True).reset_index(drop=True)
wil_sorted = wil.sort_values(
    by=["dataset", "p_value"]
)
# تسميات الأعمدة بعد الترتيب
labels = [f"{d}\n{c}" for d, c in zip(wil_sorted["dataset"], wil_sorted["comparison"])]

# ألوان الأعمدة حسب الدلالة
#colors = ["tab:red" if p < 0.05 else "tab:blue" for p in wil_sorted["p_value"]]
colors = [
    "tab:red" if p < 0.05
    else "lightgray"
    for p in wil["p_value"]
]
# رسم الرسم البياني
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(labels, wil_sorted["p_value"], color=colors,edgecolor="black")
ax.axhline(y=0.05, linestyle="--", linewidth=1, label="alpha=0.05")

ax.set_ylabel("p-value")
ax.set_title("Wilcoxon Signed-Rank Test Results (Ordered)")
ax.legend()
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.close()

print("\nSaved to:")
print(OUT_DIR)