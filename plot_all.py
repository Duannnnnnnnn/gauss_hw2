import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

# ───────────────────────────────────────────
# 圖1：收斂性分析
# ───────────────────────────────────────────
df = pd.read_csv("../gauss_pool_or_not/converge.tsv", sep="\t", header=None,
                 names=["label", "iter", "value"])

m1 = df[df["label"] == "CONVERGE_M1"].copy()
m2 = df[df["label"] == "CONVERGE_M2"].copy()
half = len(m1) // 2

m1_d1 = m1.iloc[:half].reset_index(drop=True)
m2_d1 = m2.iloc[:half].reset_index(drop=True)
m1_d2 = m1.iloc[half:].reset_index(drop=True)
m2_d2 = m2.iloc[half:].reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(m1_d1["iter"], m1_d1["value"], label="P[X|M₁]", color="steelblue", linewidth=1.5)
ax.plot(m2_d1["iter"], m2_d1["value"], label="P[X|M₂]", color="tomato",    linewidth=1.5)
ax.set_xlabel("Number of samples")
ax.set_ylabel("Integral estimate")
ax.set_title("Data generated from M1 (1 Gaussian)\nConvergence of Bayesian Integral")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(m1_d2["iter"], m1_d2["value"], label="P[X|M1]", color="steelblue", linewidth=1.5)
ax.plot(m2_d2["iter"], m2_d2["value"], label="P[X|M2]", color="tomato",    linewidth=1.5)
ax.set_xlabel("Number of samples")
ax.set_ylabel("Integral estimate")
ax.set_title("Data generated from M2 (2 Gaussians)\nConvergence of Bayesian Integral")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("Monte Carlo Convergence Analysis (fixed data, increasing sample count)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("fig1_convergence.png", dpi=150, bbox_inches="tight")
print("fig1_convergence.png 儲存完成")
plt.close()


# ───────────────────────────────────────────
# 圖2：Prior 參數實驗結果（準確率長條圖）
# ───────────────────────────────────────────

experiments = {
    "Default\nmu Prior s=4": {
        "sampling_m1": 44/50, "sampling_m2": 41/50,
        "summing_m1":  38/50, "summing_m2":  42/50,
    },
    "Narrow Prior\nmu Prior s=0.5": {
        "sampling_m1": None,  "sampling_m2": 16/20,
        "summing_m1":  18/20, "summing_m2":  16/20,
    },
    "Wide Prior\nmu Prior s=20": {
        "sampling_m1": 20/20, "sampling_m2": 17/20,
        "summing_m1":  17/20, "summing_m2":  19/20,
    },
    "Large sigma Prior\na=2.0": {
        "sampling_m1": 20/20, "sampling_m2": 17/20,
        "summing_m1":  16/20, "summing_m2":  17/20,
    },
}

labels = list(experiments.keys())
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

samp_m1 = [experiments[k]["sampling_m1"] or 0 for k in labels]
samp_m2 = [experiments[k]["sampling_m2"] or 0 for k in labels]
summ_m1 = [experiments[k]["summing_m1"]  or 0 for k in labels]
summ_m2 = [experiments[k]["summing_m2"]  or 0 for k in labels]

bars1 = ax.bar(x - 1.5*width, samp_m1, width, label="Sampling, M1 data", color="steelblue",      alpha=0.85)
bars2 = ax.bar(x - 0.5*width, samp_m2, width, label="Sampling, M2 data", color="cornflowerblue", alpha=0.85)
bars3 = ax.bar(x + 0.5*width, summ_m1, width, label="Summing,  M1 data", color="tomato",         alpha=0.85)
bars4 = ax.bar(x + 1.5*width, summ_m2, width, label="Summing,  M2 data", color="lightsalmon",    alpha=0.85)

ax.set_ylabel("Accuracy (correct selections)")
ax.set_title("Bayesian Model Selection Accuracy under Different Prior Settings (20 datasets each)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline (50%)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3, axis="y")

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.0%}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("fig2_prior_experiment.png", dpi=150, bbox_inches="tight")
print("fig2_prior_experiment.png 儲存完成")
plt.close()


# ───────────────────────────────────────────
# 圖3：高斯分布示意圖（M1 vs M2）
# ───────────────────────────────────────────
from scipy.stats import norm

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x_range = np.linspace(-15, 15, 1000)

# M1
ax = axes[0]
ax.plot(x_range, norm.pdf(x_range, 0, 3), color="steelblue", linewidth=2.5)
ax.fill_between(x_range, norm.pdf(x_range, 0, 3), alpha=0.2, color="steelblue")
ax.set_title("M1: Single Gaussian\nN(mu=0, sigma=3)")
ax.set_xlabel("x")
ax.set_ylabel("Probability density")
ax.grid(True, alpha=0.3)

# M2
ax = axes[1]
g1 = 0.4 * norm.pdf(x_range, -5, 1.5)
g2 = 0.6 * norm.pdf(x_range,  5, 1.5)
mix = g1 + g2
ax.plot(x_range, g1,  color="tomato",     linewidth=1.5, linestyle="--", label="Gaussian A (40%)")
ax.plot(x_range, g2,  color="steelblue",  linewidth=1.5, linestyle="--", label="Gaussian B (60%)")
ax.plot(x_range, mix, color="purple",     linewidth=2.5,                 label="Mixture")
ax.fill_between(x_range, mix, alpha=0.15, color="purple")
ax.set_title("M2: 2-Component Gaussian Mixture\nb=0.6, N(-5,1.5) + N(5,1.5)")
ax.set_xlabel("x")
ax.set_ylabel("Probability density")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("Model Illustration: Single Gaussian vs Gaussian Mixture", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("fig3_model_illustration.png", dpi=150, bbox_inches="tight")
print("fig3_model_illustration.png 儲存完成")
plt.close()

print("\n全部圖表儲存完成！")
