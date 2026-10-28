"""Rebuild the README figure from the notebook's saved numerical summaries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": "#C9D6E3",
    "xtick.color": "#526B80",
    "ytick.color": "#526B80",
    "text.color": "#193248",
    "axes.titlecolor": "#193248",
    "axes.labelcolor": "#526B80",
})
ink = "#193248"
muted = "#526B80"
colors = {"Analytical": "#087F8C", "Direct sampling": "#467AC1", "Gibbs MCMC": "#C46A36"}
fig = plt.figure(figsize=(16, 9.2), facecolor="#F6F9FD")
fig.text(.045, .953, "Better point predictions. Consistent posterior inference.", size=23, weight="bold")
fig.text(.045, .913, "One Bayesian linear regression model | 331 training observations | 83 held-out observations", color=muted, size=12)
fig.text(.055, .845, "A   Predictive performance", weight="bold", size=17)
fig.text(.515, .845, "B   Posterior agreement", weight="bold", size=17)
fig.text(.055, .801, "Training-mean baseline", color="#7B8FA4", size=11, weight="bold")
fig.text(.258, .801, "Bayesian predictive mean", color=colors["Analytical"], size=11, weight="bold")
for i, (name, color) in enumerate(colors.items()):
    fig.text(.515 + i * .152, .801, name, color=color, size=11, weight="bold")

ax = fig.add_axes([.085, .235, .355, .485], facecolor="#F6F9FD")
for center, metric in [(3.1, "MAE"), (1.35, "RMSE")]:
    values = [data["baseline"][metric], data["methods"]["Analytical"][metric]]
    ax.barh([center + .23, center - .23], values, height=.34, color=["#AFBECD", colors["Analytical"]], zorder=3)
    for ypos, value in zip([center + .23, center - .23], values):
        ax.text(value + .18, ypos, f"{value:.2f}", va="center", weight="bold", size=12)
ax.set_yticks([3.1, 1.35], ["MAE", "RMSE"], weight="bold", size=13)
ax.tick_params(axis="y", length=0, pad=15)
ax.set_ylim(.55, 3.9)
ax.set_xlim(0, 15.3)
ax.set_xticks([0, 5, 10, 15])
ax.set_xlabel("Prediction error (10,000 NTD per ping)", labelpad=12, size=10)
ax.grid(axis="x", color="#DFE7EF", linewidth=.8, zorder=0)
fig.text(.085, .151, "Lower is better. Both methods use the same test set.", color=muted, size=10)

for j, entry in enumerate(data["coefficients"]):
    col, row = j % 2, j // 2
    small = fig.add_axes([.53 + .236 * col, .542 - .292 * row, .188, .169], facecolor="#F6F9FD")
    for i, (name, color) in enumerate(colors.items()):
        summary = entry[name]
        mean = summary["mean"]
        lo, hi = summary["interval"]
        small.errorbar(mean, 2-i, xerr=[[mean-lo], [hi-mean]], fmt="o", color=color, markersize=5, capsize=4, elinewidth=1.8, capthick=1.5, zorder=3)
    small.set_title(entry["parameter"], loc="left", size=12, weight="bold", pad=11)
    small.set_yticks([])
    small.set_ylim(-.6, 2.6)
    small.xaxis.set_major_locator(MaxNLocator(3))
    small.tick_params(axis="x", labelsize=9, length=3)
    if entry["parameter"] == "Distance to MRT":
        small.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    small.grid(axis="x", color="#DFE7EF", linewidth=.8, zorder=0)
fig.text(.53, .182, "Points: posterior means. Lines: marginal 95% credible intervals.\nEach coefficient uses its own x-axis scale.", color=muted, size=9, linespacing=1.6)

band = FancyBboxPatch((.045, .025), .91, .106, boxstyle="round,pad=0.012,rounding_size=0.012", transform=fig.transFigure, facecolor="#E9F1F8", edgecolor="none", zorder=0)
fig.patches.append(band)
for x, value, label in [
    (.075, f'{data["rmse_reduction_percent"]:.1f}%', "lower RMSE than the baseline"),
    (.385, "80 / 83", "held-out outcomes covered by 95% PIs"),
    (.705, "37.74", "mean predictive interval width"),
]:
    fig.text(x, .087, value, size=25, weight="bold", color=ink)
    fig.text(x, .048, label, size=10, color=muted)
fig.savefig(ROOT / "assets" / "results.png", dpi=160, facecolor=fig.get_facecolor())
fig.savefig(ROOT / "assets" / "results.svg", facecolor=fig.get_facecolor())
plt.close(fig)
print("Created assets/results.png and assets/results.svg from results.json.")
