"""
Overview EDA on the cleaned posts. Run after src/clean.py.

    python eda/overview/overview_eda.py

Reads  : data/processed/posts_clean.csv, data/processed/concreteness_scores_influencers.csv,
         annotation/sample_key.csv and the round 1-2 label sheets
Writes : eda/overview/overview_summary.json, eda/overview/figures/*.png
Write-up in plain words: eda/overview/overview_eda.md
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

# Reference palette (dataviz skill), light mode; slots 1-3 validated all-pairs.
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

df = pd.read_csv(ROOT / "data/processed/posts_clean.csv", encoding="utf-8-sig", low_memory=False)
df["group"] = np.where(df.source == "influencers", "Influencers",
                       np.where(df.is_repost, "Ordinary users: reposts", "Ordinary users: own posts"))
inf = df[df.group == "Influencers"]
own = df[df.group == "Ordinary users: own posts"]
low = df.text.str.lower()
out = {}

# ------------------------------------------------------------------ 1. size and data quality
out["size"] = df.group.value_counts().to_dict()
out["authors"] = df.groupby("group").author_id.nunique().to_dict()
cols = ["author_info", "followers", "media_type", "posted", "reactions", "comments", "reposts"]
out["missing_share"] = {g: d[cols].isna().mean().round(3).to_dict() for g, d in df.groupby("source")}

# ------------------------------------------------------------------ 2. time coverage
rel = inf.posted.astype(str).str.extract(r"(\d+)\s+(day|week|month|year)")
years_ago = pd.to_numeric(rel[0], errors="coerce") * rel[1].map({"day": 1 / 365, "week": 1 / 52, "month": 1 / 12,
                                                                  "year": 1})
out["influencer_age_years_ago"] = {
    "under_1": float((years_ago < 1).mean()), "1": float((years_ago == 1).mean()),
    "2_to_3": float(years_ago.between(2, 3).mean()), "4_plus": float((years_ago >= 4).mean()),
    "unparsed": float(years_ago.isna().mean()),
}
dates = pd.to_datetime(df[df.source == "figshare"].posted, errors="coerce")
out["figshare_dates"] = {"parsed_share": float(dates.notna().mean()),
                         "by_year": dates.dt.year.value_counts().sort_index().astype(int).to_dict()}

# ------------------------------------------------------------------ 3. length
out["words"] = {g: d.n_words.describe(percentiles=[.25, .5, .75, .9]).round(1).to_dict() for g, d in df.groupby("group")}
out["share_under_10_words"] = df.groupby("group").n_words.apply(lambda s: float((s < 10).mean())).to_dict()
out["share_no_words_only_tags_or_links"] = df.groupby("group").n_words.apply(lambda s: float((s == 0).mean())).to_dict()
out["influencer_truncated_share"] = float(inf.truncated.mean())
out["words_truncated_vs_not"] = inf.groupby("truncated").n_words.median().to_dict()
out["media_type"] = inf.media_type.value_counts(normalize=True).round(3).to_dict()

# ------------------------------------------------------------------ 4. authors
per_author = inf.groupby("author_id").size().sort_values(ascending=False)
out["influencer_posts_per_author"] = {
    "median": float(per_author.median()), "min": int(per_author.min()), "max": int(per_author.max()),
    "top5_share_of_posts": float(per_author.head(5).sum() / len(inf)),
}

# ------------------------------------------------------------------ 5. patterns (rough text searches, never labels)
PATTERNS = {
    "Lesson framing (\"lessons\", \"taught me\", \"here's what\")":
        r"\b(?:lessons?|learn(?:ed|t)|taught me|here'?s what|here is what|takeaways?|remember this)\b",
    "\"It's not about X, it's about Y\"":
        r"\b(?:is not|isn'?t|it'?s not|it is not|not about)\b.{0,80}?\b(?:it'?s|it is|about)\b",
    "Announcement (\"thrilled / honoured to\")": r"\b(?:thrilled|humbled|delighted|honou?red|excited|proud) to\b",
    "Asks readers a question (\"Agree?\", \"Thoughts?\")": r"\b(?:agree|thoughts|what do you think|your thoughts)\s*\?",
    "Hiring post": r"\b(?:we'?re hiring|i'?m hiring|is hiring|now hiring|#hiring)\b",
    "Contains a number": r"\d",
    "Contains a link": r"https?://|lnkd\.in|www\.",
    "Has hashtags": r"#\w+",
    "Has an emoji": "[\U0001F300-\U0001FAFF☀-➿]",
}
FLOWERY = (r"\b(?:journey|passion|purpose|inspir|empower|transform|unlock|unleash|ignite|elevate|mindset"
           r"|authentic|thrive|embrace|resilien|legacy|wisdom|courage|destiny|abundance|soul|profound"
           r"|synerg|leverag|disrupt|holistic|paradigm|mindful)\w*")
flags = {name: low.str.contains(p, regex=True) for name, p in PATTERNS.items()}
flags["Inspirational / buzz words (journey, empower, unlock...)"] = low.str.contains(FLOWERY, regex=True)
flags["Cut off at \"...see more\""] = df.truncated
groups2 = ["Influencers", "Ordinary users: own posts"]
share = {k: {g: float(v[df.group == g].mean()) for g in groups2} for k, v in flags.items()}
out["patterns"] = share

# ------------------------------------------------------------------ 6. vocabulary
clean = df.text.str.replace(r"https?://\S+|www\.\S+|lnkd\.in/\S+|@\w+|#\w+", " ", regex=True)
STOP = list(ENGLISH_STOP_WORDS | {"ve", "ll", "don", "just", "like", "amp", "s", "t", "m", "re", "d"})
vocab = {}
for g in groups2:
    for n, key in [(1, "words"), (2, "phrases")]:
        cv = CountVectorizer(stop_words=STOP, ngram_range=(n, n), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
                             binary=True)
        m = cv.fit_transform(clean[df.group == g])
        counts = np.asarray(m.sum(0)).ravel() / m.shape[0]
        top = np.argsort(-counts)[:15]
        vocab.setdefault(g, {})[key] = {cv.get_feature_names_out()[i]: round(float(counts[i]), 3) for i in top}
out["top_words_share_of_posts"] = vocab
tags = pd.Series([t.lower() for x in inf.text for t in re.findall(r"#(\w+)", x)])
out["top_hashtags_influencers"] = tags.value_counts().head(15).to_dict()

# ------------------------------------------------------------------ 7. post types (topic clusters)
# TF-IDF + SVD ("LSA") is a cheap stand-in for sentence embeddings; sentence-transformers isn't installed.
vec = TfidfVectorizer(stop_words="english", min_df=10, max_df=0.4, ngram_range=(1, 2), sublinear_tf=True,
                      max_features=20000)
X = vec.fit_transform(df.text.str.replace(r"https?://\S+|www\.\S+|lnkd\.in/\S+|@\w+", " ", regex=True)
                      .str.replace("#", " ", regex=False))
svd = TruncatedSVD(100, random_state=0)
Z = Normalizer().fit_transform(svd.fit_transform(X))
km = KMeans(10, n_init=10, random_state=0).fit(Z)
terms = np.array(vec.get_feature_names_out())
centres = svd.inverse_transform(km.cluster_centers_)
NAMES = {  # first top term -> readable name (checked by reading example posts)
    "fintech": "Banking & fintech", "people": "Opinions about people & work",
    "time": "Work, productivity & life advice", "marketing": "Marketing & personal branding",
    "radicalcpa": "Accounting (#radicalcpa)", "healthcare": "Digital health",
    "new": "General musings & advice", "markets": "Markets & economy news",
    "women": "Global health, women & world news", "great": "Business news, events & announcements",
}
df["post_type"] = km.labels_
types = {}
for j in range(10):
    top = terms[np.argsort(-centres[j])[:10]]
    m = df.post_type == j
    types[j] = {
        "name": NAMES.get(top[0], ", ".join(top[:3])),
        "top_terms": top.tolist(), "posts": int(m.sum()),
        "share": {g: float(m[df.group == g].mean()) for g in df.group.unique()},
        "influencer_authors": int(df.author_id[m & (df.group == "Influencers")].nunique()),
        "top_influencer_share": float(df.author_id[m & (df.group == "Influencers")].value_counts(normalize=True).iloc[0]),
    }
df["post_type_name"] = df.post_type.map(lambda j: types[j]["name"])
out["post_types"] = types

# ------------------------------------------------------------------ 8. pilot labels (rounds 1-2, first answers)
def sheet(path, n=30):
    return pd.read_excel(ROOT / path, sheet_name="Labels").set_index("post_id").label.head(n)


key = pd.read_csv(ROOT / "annotation/sample_key.csv", encoding="utf-8-sig").set_index("post_id")
L = pd.concat([pd.DataFrame({w: sheet(f"annotation/round1/round1_{w}.xlsx") for w in "ABC"}),
               pd.DataFrame({w: sheet(f"annotation/round2/round2_{w}.xlsx") for w in "AB"})])
L.loc[["R2-06", "R2-11"], "A"] = "not bullshit"  # A's first answers (changed after seeing B's sheet)
lab = pd.DataFrame({"bs": (L == "bullshit").sum(1) / L.notna().sum(1) > 0.5})
lab["source_row"], lab["source"], lab["pool"] = key.loc[L.index, "source_row"], key.loc[L.index, "source"], \
    key.loc[L.index, "pool"]
lab = lab.merge(df[["source", "source_row", "media_type", "truncated", "n_words", "post_type_name"]],
                on=["source", "source_row"], how="left")
conc = pd.read_csv(ROOT / "data/processed/concreteness_scores_influencers.csv")
lab = lab.merge(conc[["row_id", "conc_mean", "reliable"]].rename(columns={"row_id": "source_row"})
                .assign(source="influencers"), on=["source", "source_row"], how="left")
lab["length"] = pd.cut(lab.n_words, [0, 30, 60, 10_000], labels=["10-30 words", "31-60 words", "61+ words"])
pilot = {"posts": int(len(lab)), "bullshit_posts": int(lab.bs.sum())}
for col in ["pool", "media_type", "truncated", "length", "post_type_name"]:
    g = lab.groupby(col, observed=True).bs.agg(["mean", "size"])
    pilot[col] = {str(k): {"bullshit_share": round(float(v["mean"]), 2), "posts": int(v["size"])} for k, v in g.iterrows()}
rel = lab[lab.reliable == True]  # noqa: E712
pilot["concreteness_influencer_posts"] = {
    "bullshit_mean": float(rel[rel.bs].conc_mean.mean()), "not_bullshit_mean": float(rel[~rel.bs].conc_mean.mean()),
    "n_bullshit": int(rel.bs.sum()), "n_not": int((~rel.bs).sum()),
}
out["pilot"] = pilot

# ------------------------------------------------------------------ 9. engagement (within the same influencer)
eng = inf.dropna(subset=["reactions"]).copy()
eng["lr"] = np.log1p(eng.reactions)
eng["dm"] = eng.lr - eng.groupby("author_id").lr.transform("mean")
r_ = eng.lr - eng.groupby("author_id").lr.transform("mean")
out["engagement"] = {
    "reactions_median": float(eng.reactions.median()), "reactions_mean": float(eng.reactions.mean()),
    "reactions_max": float(eng.reactions.max()),
    "author_explains_share_of_log_reactions_variance": float(1 - (r_ ** 2).sum() / ((eng.lr - eng.lr.mean()) ** 2).sum()),
    "top5_authors_share_of_reactions": float(eng.groupby("author_id").reactions.sum().nlargest(5).sum()
                                             / eng.reactions.sum()),
    "ordinary_own_reactions_median": float(own.reactions.median()),
}
ENG_PATTERNS = ["Lesson framing (\"lessons\", \"taught me\", \"here's what\")", "\"It's not about X, it's about Y\"",
                "Inspirational / buzz words (journey, empower, unlock...)", "Announcement (\"thrilled / honoured to\")",
                "Asks readers a question (\"Agree?\", \"Thoughts?\")", "Contains a number", "Contains a link"]
eng_rows = {}
for name in ENG_PATTERNS:
    f = flags[name][eng.index]
    agg = eng.assign(f=f).groupby(["author_id", "f"]).dm.agg(["sum", "count"]).unstack(fill_value=0)
    s1, n1, s0, n0 = (agg[("sum", True)].values, agg[("count", True)].values,
                      agg[("sum", False)].values, agg[("count", False)].values)
    point = s1.sum() / n1.sum() - s0.sum() / n0.sum()
    draws = []
    for _ in range(1000):
        i = RNG.integers(0, len(s1), len(s1))
        if n1[i].sum() and n0[i].sum():
            draws.append(s1[i].sum() / n1[i].sum() - s0[i].sum() / n0[i].sum())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    eng_rows[name] = {"pct": float(np.expm1(point)), "lo": float(np.expm1(lo)), "hi": float(np.expm1(hi)),
                      "posts_with": int(n1.sum())}
out["engagement"]["within_author_pct_difference"] = eng_rows
(OUT / "overview_summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


# ------------------------------------------------------------------ figures
def style(ax, grid="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def title(fig, main, sub):
    fig.text(0.02, 0.97, main, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, sub, fontsize=9.5, color=INK2, va="top")


# Figure 4: post length
fig, ax = plt.subplots(figsize=(9, 4.2))
fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.14)
bins = np.concatenate([[0, 1], np.logspace(0, np.log10(1000), 40)[1:]])
for d, col, lab_ in [(inf, BLUE, "Influencers"), (own, ORANGE, "Ordinary users (own posts)")]:
    w = np.ones(len(d)) / len(d) * 100
    ax.hist(d.n_words.clip(lower=0.5), bins=np.logspace(np.log10(0.5), np.log10(1000), 45), weights=w,
            histtype="step", linewidth=2, color=col, label=f"{lab_}: median {int(d.n_words.median())} words")
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 3, 10, 30, 100, 300, 1000])
ax.set_xticklabels(["0", "1", "3", "10", "30", "100", "300", "1,000"])
ax.set_xlabel("Words in the visible text, not counting links, hashtags and @mentions (log scale)")
ax.set_ylabel("% of posts")
style(ax, grid="y")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
title(fig, "Most posts are short; influencer posts stop early because they are previews",
      f"{out['share_under_10_words']['Influencers']:.0%} of influencer posts have under 10 words. "
      f"{inf.truncated.mean():.0%} are cut off at \"...see more\" (none over {int(inf.n_words.max())} words).")
fig.savefig(FIG / "fig4_post_length.png", dpi=200)
plt.close(fig)

# Figure 5: posts per influencer
fig, ax = plt.subplots(figsize=(9, 3.8))
fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.14)
ax.bar(range(len(per_author)), per_author.values, color=BLUE, width=0.8)
ax.set_xlim(-1, len(per_author))
ax.set_xticks([])
ax.set_xlabel(f"Each bar is one influencer ({len(per_author)} in total), sorted by number of posts")
ax.set_ylabel("Posts after cleaning")
style(ax, grid="y")
ax.axhline(per_author.median(), color=INK2, linewidth=1, linestyle=(0, (3, 3)))
ax.text(len(per_author) - 1, per_author.median() + 15, f"median {int(per_author.median())}", ha="right",
        fontsize=9, color=INK2)
title(fig, "A few influencers write a large share of the posts",
      f"The top 5 write {out['influencer_posts_per_author']['top5_share_of_posts']:.0%} of influencer posts. "
      "Samples cap posts per author so no one dominates the labels.")
fig.savefig(FIG / "fig5_posts_per_author.png", dpi=200)
plt.close(fig)

# Figure 6: patterns by source
order = sorted(share, key=lambda k: share[k]["Influencers"])
fig, ax = plt.subplots(figsize=(9.5, 5.8))
fig.subplots_adjust(left=0.43, right=0.95, top=0.81, bottom=0.09)
y = np.arange(len(order))
for dy, g, col, lab_ in [(0.18, "Influencers", BLUE, "Influencers"),
                         (-0.18, "Ordinary users: own posts", ORANGE, "Ordinary users (own posts)")]:
    vals = [share[k][g] * 100 for k in order]
    ax.barh(y + dy, vals, height=0.33, color=col, label=lab_)
    for yi, v in zip(y, vals):
        ax.text(v + 0.8, yi + dy, f"{v:.0f}%", va="center", fontsize=8, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels(order, fontsize=9)
ax.set_xlabel("% of posts")
ax.set_xlim(0, 105)
style(ax)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
title(fig, "Ordinary users post more announcements and emoji; influencers share more links and hashtags",
      "Share of cleaned posts showing each pattern. Patterns are rough text searches, not labels.")
fig.savefig(FIG / "fig6_patterns_by_source.png", dpi=200)
plt.close(fig)

# Figure 7: post types
order = sorted(types, key=lambda j: types[j]["share"]["Influencers"])
fig, ax = plt.subplots(figsize=(9.5, 5.4))
fig.subplots_adjust(left=0.36, right=0.83, top=0.8, bottom=0.1)
y = np.arange(len(order))
for dy, g, col, lab_ in [(0.18, "Influencers", BLUE, "Influencers"),
                         (-0.18, "Ordinary users: own posts", ORANGE, "Ordinary users (own posts)")]:
    vals = [types[j]["share"][g] * 100 for j in order]
    ax.barh(y + dy, vals, height=0.33, color=col, label=lab_)
    for yi, v in zip(y, vals):
        ax.text(v + 0.6, yi + dy, f"{v:.0f}%", va="center", fontsize=8, color=INK2)
for yi, j in zip(y, order):
    ax.text(1.02, yi, f"{types[j]['top_influencer_share']:.0%}", transform=ax.get_yaxis_transform(),
            va="center", fontsize=8.5, color=INK2)
ax.text(1.02, len(order) - 0.3, "top influencer's\nshare of type", transform=ax.get_yaxis_transform(),
        va="bottom", fontsize=8, color=MUTED)
ax.set_yticks(y)
ax.set_yticklabels([types[j]["name"] for j in order], fontsize=9)
ax.set_xlabel("% of posts")
style(ax)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
title(fig, "Some post types are basically one influencer",
      "Posts grouped by the words they use (TF-IDF + k-means, 10 groups; names given by reading examples).\n"
      "Right column: share of that type's influencer posts written by its most frequent author.")
fig.savefig(FIG / "fig7_post_types.png", dpi=200)
plt.close(fig)

# Figure 8: engagement within author
names = sorted(eng_rows, key=lambda k: eng_rows[k]["pct"])
fig, ax = plt.subplots(figsize=(9.5, 4.6))
fig.subplots_adjust(left=0.43, right=0.93, top=0.76, bottom=0.14)
ax.axvline(0, color=BASE, linewidth=1)
for i, k in enumerate(names):
    d = eng_rows[k]
    ax.plot([d["lo"] * 100, d["hi"] * 100], [i, i], color=BLUE, linewidth=2, solid_capstyle="round")
    ax.scatter(d["pct"] * 100, i, s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(d["hi"] * 100 + 2, i, f"{d['pct']:+.0%}  (n={d['posts_with']:,})", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlim(min(d["lo"] for d in eng_rows.values()) * 100 - 5, max(d["hi"] for d in eng_rows.values()) * 100 + 24)
ax.set_xlabel("Difference in reactions vs the same influencer's other posts (%, 95% CI)")
style(ax)
title(fig, "Within the same influencer, announcements, lessons and buzz words get more reactions; links get fewer",
      "Influencer posts only. A first look at the engagement extension: patterns, not bullshit labels.\n"
      "CIs come from resampling influencers.")
fig.savefig(FIG / "fig8_engagement_within_author.png", dpi=200)
plt.close(fig)

print(json.dumps({k: v for k, v in out.items() if k not in ("top_words_share_of_posts", "top_hashtags_influencers")},
                 indent=1, default=float)[:6000])
print("vocab:", json.dumps(out["top_words_share_of_posts"], indent=0)[:2500])
