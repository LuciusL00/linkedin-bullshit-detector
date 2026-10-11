"""
Charts for the concreteness EDA. Run after eda/concreteness/concreteness_eda.py.

    python eda/concreteness/concreteness_charts.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EDA = Path(__file__).resolve().parent
FIG = EDA / "figures"
FIG.mkdir(exist_ok=True)

# Reference palette (dataviz skill), light mode. Slots 1-2 validated together.
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BAND = "#f0efec"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

summary = json.loads((EDA / "concreteness_summary.json").read_text(encoding="utf-8"))
scores = pd.read_csv(ROOT / "data/processed/concreteness_scores_influencers.csv")
rel = scores[scores.reliable]
p10, p90 = rel.conc_mean.quantile([.1, .9])
mean = rel.conc_mean.mean()


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def title(fig, main, sub):
    fig.text(0.02, 0.97, main, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.915, sub, fontsize=9.5, color=INK2, va="top")


# ---------------------------------------------------------------- Figure 1
EX = summary["examples"]
rows = [  # (label, key, is_bullshit)
    ("Pseudo-profound: \"Hidden meaning transforms unparalleled abstract beauty\"", "Pseudo-profound (Pennycook stimulus)", True),
    ("Pseudo-profound: \"Wholeness quiets infinite phenomena\"", "Pseudo-profound (Pennycook stimulus) 2", True),
    ("LinkedIn maxim: \"Leadership is not about titles...\"", "LinkedIn maxim", True),
    ("Corporate buzzwords: \"leverage synergies to unlock value\"", "Corporate buzzwords", True),
    ("Coffee + inflated lesson", "Coffee + inflated lesson", True),
    ("Technical, vague: \"Inflation expectations remain anchored\"", "Technical, vague", False),
    ("Technical, specific: \"...near 2.3%... yields swung 40bp\"", "Technical, specific", False),
    ("Motivational quote: \"A river cuts through rock...\"", "Motivational quote", True),  # generic message: bullshit under rubric v2
    ("Mundane: \"Had coffee with the team this morning\"", "Mundane", False),
]
rows = sorted(rows, key=lambda r: EX[r[1]]["conc_mean"])
fig, ax = plt.subplots(figsize=(9.5, 5.2))
fig.subplots_adjust(left=0.47, right=0.96, top=0.78, bottom=0.12)
ax.axvspan(p10, p90, color=BAND, zorder=0)
ax.axvline(mean, color=BASE, linewidth=1, zorder=1, ymax=0.88)
ax.text(mean, len(rows) - 0.35, f"Real posts: middle 80%\n({p10:.2f}-{p90:.2f}, mean {mean:.2f})",
        ha="center", va="bottom", fontsize=8.5, color=MUTED)
for i, (lab, key, bs) in enumerate(rows):
    v = EX[key]["conc_mean"]
    ax.scatter(v, i, s=70, color=ORANGE if bs else BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlim(1.9, 4.2)
ax.set_ylim(-0.6, len(rows) + 0.6)
ax.set_xlabel("Mean concreteness of content words (1 = abstract, 5 = concrete)")
style(ax)
h1 = plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=ORANGE, label="Bullshit (claims depth, says nothing)")
h2 = plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=BLUE, label="Not bullshit (mundane or meaningful)")
fig.subplots_adjust(top=0.74)
fig.legend(handles=[h1, h2], loc="upper left", bbox_to_anchor=(0.015, 0.84), ncol=2, frameon=False,
           fontsize=9, labelcolor=INK2)
title(fig, "Concreteness alone does not separate bullshit from meaningful text",
      "Illustrative sentences scored with the Brysbaert norms. Bullshit and non-bullshit overlap:\n"
      "technical language scores as abstract, and concrete metaphors score as concrete.")
fig.savefig(FIG / "fig1_examples_on_scale.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fd = rel.dropna(subset=["flowery_per100", "conc_resid"]).copy()
nz_med = fd.loc[fd.flowery_per100 > 0, "flowery_per100"].median()
fd["grp"] = np.where(fd.flowery_per100 == 0, "None",
                     np.where(fd.flowery_per100 <= nz_med, "Light", "Heavy"))
order = ["None", "Light", "Heavy"]
stats = fd.groupby("grp").agg(n=("conc_mean", "size"), full=("conc_mean", "mean"),
                              resid=("conc_resid", "mean"), dens=("flowery_per100", "mean")).loc[order]


def ci(s, reps=1000, rng=np.random.default_rng(1)):
    a = s.values
    m = [rng.choice(a, len(a)).mean() for _ in range(reps)]
    return np.percentile(m, [2.5, 97.5])


fig, ax = plt.subplots(figsize=(9, 4.2))
fig.subplots_adjust(left=0.30, right=0.96, top=0.74, bottom=0.16)
for i, g in enumerate(order):
    sub = fd[fd.grp == g]
    for val, col, dy in [("conc_mean", BLUE, 0.13), ("conc_resid", ORANGE, -0.13)]:
        lo, hi = ci(sub[val])
        ax.plot([lo, hi], [i + dy, i + dy], color=col, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter(sub[val].mean(), i + dy, s=70, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(hi + 0.006, i + dy, f"{sub[val].mean():.2f}", va="center", fontsize=9, color=INK2)
pct = (stats.n / stats.n.sum() * 100).round(0)
ax.set_yticks(range(3))
GLAB = {"None": "No flowery words", "Light": "Light use", "Heavy": "Heavy use"}
ax.set_yticklabels([f"{GLAB[g]}\n(avg {stats.loc[g,'dens']:.1f} per 100 words; {pct[g]:.0f}% of posts)" for g in order], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Mean concreteness (95% CI)")
style(ax)
h1 = plt.Line2D([], [], marker="o", linestyle="-", markersize=8, color=BLUE, label="All content words")
h2 = plt.Line2D([], [], marker="o", linestyle="-", markersize=8, color=ORANGE, label="Excluding the flowery words themselves")
ax.legend(handles=[h1, h2], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
          fontsize=9, labelcolor=INK2)
title(fig, "Flowery posts are less concrete mostly because flowery words are abstract",
      f"{len(fd):,} influencer posts. Heavy-flowery posts drop {stats.loc['None','full']-stats.loc['Heavy','full']:.2f} on all words, "
      f"but only {stats.loc['None','resid']-stats.loc['Heavy','resid']:.2f} on their other words.\n"
      f"For scale: one standard deviation across posts = {rel.conc_mean.std():.2f}.")
fig.savefig(FIG / "fig2_flowery_density.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
NAMES = {
    "maxim_contrast": "\"It's not about X, it's about Y\"",
    "inspirational_hashtag": "Inspirational hashtags (#motivation...)",
    "lesson_framing": "Lesson framing (\"here's what I learned\")",
    "announce_humblebrag": "Announcement (\"thrilled / humbled to\")",
    "multi_exclamation": "Two or more exclamation marks",
    "has_number": "Contains a number (specificity check)",
}
mk = [(k, v["conc_mean"]) for k, v in summary["markers"].items()]
mk = sorted(mk, key=lambda kv: kv[1]["diff"])
fig, ax = plt.subplots(figsize=(9, 4.4))
fig.subplots_adjust(left=0.36, right=0.95, top=0.75, bottom=0.15)
ax.axvline(0, color=BASE, linewidth=1, zorder=1)
for i, (k, d) in enumerate(mk):
    ax.plot([d["lo"], d["hi"]], [i, i], color=BLUE, linewidth=2, solid_capstyle="round", zorder=2)
    ax.scatter(d["diff"], i, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(d["hi"] + 0.004, i, f"{d['diff']:+.3f}  (n={d['n_with']:,})", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(mk)))
ax.set_yticklabels([NAMES[k] for k, _ in mk], fontsize=9)
ax.set_xlim(-0.15, 0.11)
ax.set_xlabel("Difference in concreteness vs the same author's other posts (95% CI)")
style(ax)
title(fig, "Only the \"not X, but Y\" maxim is noticeably more abstract",
      "Within-author comparison across 68 influencers; CIs from resampling authors.\n"
      f"Effects are small: the largest is about a quarter of one standard deviation ({rel.conc_mean.std():.2f}).")
fig.savefig(FIG / "fig3_framing_markers.png", dpi=200)
plt.close(fig)

stats.to_csv(EDA / "flowery_groups.csv")
print("saved figures to", FIG)
print(stats.round(3))
