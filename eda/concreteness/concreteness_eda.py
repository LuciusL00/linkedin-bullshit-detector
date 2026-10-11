"""
Concreteness EDA for the bullshit-detector project.

Scores every post with the Brysbaert et al. (2014) concreteness norms and tests
whether "flowery" language is less concrete.

    python eda/concreteness/concreteness_eda.py

Reads  : data/raw/brysbaert_concreteness.txt, data/raw/linkedin_*.csv/xlsx
Writes : data/processed/concreteness_scores_influencers.csv
         eda/concreteness/concreteness_summary.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP

ROOT = Path(__file__).resolve().parents[2]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
EDA = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)

# ------------------------------------------------------------------ lexicon
lex = pd.read_csv(RAW / "brysbaert_concreteness.txt", sep="\t")
CONTENT_POS = {"Noun", "Adjective", "Verb", "Adverb"}
uni = lex[(lex.Bigram == 0) & lex.Dom_Pos.isin(CONTENT_POS)]
UNI = dict(zip(uni.Word.str.lower(), uni["Conc.M"]))
big = lex[lex.Bigram == 1]
BIG = dict(zip(big.Word.str.lower(), big["Conc.M"]))


def lemma(w):
    """Lexicon-guided suffix stripping: return the first form found in the norms."""
    if w in UNI:
        return w
    cands = []
    if w.endswith("ies"):
        cands.append(w[:-3] + "y")
    if w.endswith("ied"):
        cands.append(w[:-3] + "y")
    if w.endswith("es"):
        cands.append(w[:-2])
    if w.endswith("s") and not w.endswith("ss"):
        cands.append(w[:-1])
    if w.endswith("ed"):
        cands += [w[:-2], w[:-1]]
        if len(w) > 4 and w[-3] == w[-4]:
            cands.append(w[:-3])          # stopped -> stop
    if w.endswith("ing"):
        cands += [w[:-3], w[:-3] + "e"]
        if len(w) > 5 and w[-4] == w[-5]:
            cands.append(w[:-4])          # running -> run
    for c in cands:
        if c in UNI:
            return c
    return None


# Curated list of inspirational / grandiose / corporate-buzz words (base forms).
FLOWERY_BASE = """journey passion passionate purpose purposeful inspire inspiration
empower empowerment transform transformation transformative unlock unleash ignite
elevate mindset growth authentic authenticity vision visionary impact impactful thrive
embrace resilience resilient potential success successful excellence greatness humble
grateful gratitude blessed thrilled honored honoured incredible amazing extraordinary
remarkable meaningful legacy wisdom courage belief believe dream destiny abundance
kindness soul spirit magic beautiful powerful profound mission value leadership
innovation innovative synergy leverage disrupt disruptive disruption holistic ecosystem
paradigm strategic alignment mindful mindfulness""".split()
FLOWERY = {lemma(w) or w for w in FLOWERY_BASE}

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#\w+")
SEE_MORE = re.compile(r"\s*…?\s*see more\s*$", re.I)
TOKEN = re.compile(r"[a-z]+")


def score(text):
    """Return per-post concreteness measures."""
    t = SEE_MORE.sub("", str(text))
    t = HASHTAG.sub(" ", MENTION.sub(" ", URL.sub(" ", t))).lower()
    toks = [w for w in TOKEN.findall(t)]
    content = [w for w in toks if w not in STOP and len(w) > 1]
    vals, vals_resid, n_flow = [], [], 0
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and f"{toks[i]} {toks[i+1]}" in BIG:
            vals.append(BIG[f"{toks[i]} {toks[i+1]}"]); vals_resid.append(vals[-1]); i += 2
            continue
        w = toks[i]; i += 1
        if w in STOP or len(w) < 2:
            continue
        lw = lemma(w)
        if lw is None:
            continue
        v = UNI[lw]
        vals.append(v)
        if lw in FLOWERY:
            n_flow += 1
        else:
            vals_resid.append(v)
    n = len(vals)
    return dict(
        n_tokens=len(toks), n_content=len(content), n_matched=n,
        coverage=n / len(content) if content else np.nan,
        conc_mean=np.mean(vals) if n else np.nan,
        conc_resid=np.mean(vals_resid) if vals_resid else np.nan,
        abstract_ratio=np.mean(np.array(vals) < 2.5) if n else np.nan,
        flowery_per100=100 * n_flow / len(content) if content else np.nan,
    )


# Structural "flowery framing" markers - independent of word concreteness.
MARKERS = {
    "maxim_contrast": r"\b(?:is not|isn'?t|it'?s not|it is not|not about)\b.{0,80}?\b(?:it'?s|it is|about)\b",
    "lesson_framing": r"\b(?:lessons?|learn(?:ed|t)|taught me|here'?s what|here is what|takeaways?|remember this)\b",
    "announce_humblebrag": r"\b(?:thrilled|humbled|delighted|honou?red|excited|proud) to\b",
    "inspirational_hashtag": r"#(?:motivation|inspiration|leadership|success|mindset|growth|personaldevelopment|entrepreneurship|believe|positivity|gratitude|selfimprovement|inspire|goals|purpose|hustle)\b",
    "multi_exclamation": r"!.*!",
    "has_number": r"\d",   # specificity contrast, expected to go the other way
}


def add_markers(df, col):
    low = df[col].astype(str).str.lower()
    for k, pat in MARKERS.items():
        df[k] = low.str.contains(pat, regex=True)
    return df


def cluster_boot_diff(d, flag, val, group, reps=1000):
    """Within-group (author-demeaned) difference in means, CI by bootstrapping groups."""
    d = d[[flag, val, group]].dropna().copy()
    d["dm"] = d[val] - d.groupby(group)[val].transform("mean")
    agg = d.groupby([group, flag])["dm"].agg(["sum", "count"]).unstack(fill_value=0)
    s1, n1 = agg[("sum", True)].values, agg[("count", True)].values
    s0, n0 = agg[("sum", False)].values, agg[("count", False)].values
    point = s1.sum() / n1.sum() - s0.sum() / n0.sum()
    k = len(s1); draws = []
    for _ in range(reps):
        idx = RNG.integers(0, k, k)
        if n1[idx].sum() == 0 or n0[idx].sum() == 0:
            continue
        draws.append(s1[idx].sum() / n1[idx].sum() - s0[idx].sum() / n0[idx].sum())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi, int(n1.sum()), int(n0.sum())


# ------------------------------------------------------------------ data
inf = pd.read_csv(RAW / "linkedin_influencers_34k.csv", low_memory=False)
inf = inf.dropna(subset=["content"]).reset_index().rename(columns={"index": "row_id"})
inf["truncated"] = inf["content"].astype(str).str.contains(r"see more\s*$", regex=True)
S = pd.DataFrame([score(t) for t in inf["content"]])
inf = pd.concat([inf[["row_id", "name", "media_type", "truncated", "content"]], S], axis=1)
inf = add_markers(inf, "content")
inf["reliable"] = inf["n_matched"] >= 5

fig = pd.read_excel(RAW / "linkedin_published_1000.xlsx", sheet_name="General")
fig.columns = [str(c).strip() for c in fig.columns]
fig = fig[fig["POST"].notna() & (fig["POST"].astype(str).str.upper().str.strip() != "NO POSTS")].copy()
fig = pd.concat([fig.reset_index(drop=True), pd.DataFrame([score(t) for t in fig["POST"]])], axis=1)
p98 = pd.read_csv(RAW / "linkedin_posts_98.csv")
p98 = pd.concat([p98, pd.DataFrame([score(t) for t in p98["post_text"]])], axis=1)

out = {}
rel = inf[inf.reliable]
out["coverage"] = dict(
    posts=len(inf), reliable_posts=int(inf.reliable.sum()),
    median_matched=float(inf.n_matched.median()), median_coverage=float(inf.coverage.median()),
)
out["distribution"] = {
    name: dict(n=int((d.n_matched >= 5).sum()),
               mean=float(d.loc[d.n_matched >= 5, "conc_mean"].mean()),
               sd=float(d.loc[d.n_matched >= 5, "conc_mean"].std()),
               p10=float(d.loc[d.n_matched >= 5, "conc_mean"].quantile(.1)),
               p90=float(d.loc[d.n_matched >= 5, "conc_mean"].quantile(.9)))
    for name, d in [("influencers", inf), ("figshare", fig), ("zenodo98", p98)]
}
# share of variance explained by author
g = rel.groupby("name")["conc_mean"].transform("mean")
out["author_r2"] = float(1 - ((rel.conc_mean - g) ** 2).sum() / ((rel.conc_mean - rel.conc_mean.mean()) ** 2).sum())
# truncation
out["truncation"] = {str(k): float(v) for k, v in rel.groupby("truncated")["conc_mean"].mean().items()}

# flowery density vs concreteness of the REST of the words (non-mechanical test)
fd = rel.dropna(subset=["flowery_per100", "conc_resid"])
rho, p = stats.spearmanr(fd.flowery_per100, fd.conc_resid)
fdw = fd.copy()
for c in ["flowery_per100", "conc_resid", "conc_mean"]:
    fdw[c] = fdw[c] - fdw.groupby("name")[c].transform("mean")
rho_w, p_w = stats.spearmanr(fdw.flowery_per100, fdw.conc_resid)
rho_full, _ = stats.spearmanr(fd.flowery_per100, fd.conc_mean)
out["flowery_vs_concreteness"] = dict(
    spearman_full_words=float(rho_full),
    spearman_residual_words=float(rho), p=float(p),
    spearman_residual_within_author=float(rho_w), p_within=float(p_w),
    share_posts_with_flowery=float((fd.flowery_per100 > 0).mean()),
)
# flowery quintiles
fd = fd.assign(fq=pd.qcut(fd.flowery_per100.rank(method="first"), 5, labels=False))
out["flowery_quintiles"] = fd.groupby("fq")[["flowery_per100", "conc_mean", "conc_resid"]].mean().round(3).to_dict("index")

# structural markers, within author
out["markers"] = {}
for k in MARKERS:
    for val in ["conc_mean", "conc_resid"]:
        pt, lo, hi, n1, n0 = cluster_boot_diff(rel, k, val, "name")
        out["markers"].setdefault(k, {})[val] = dict(diff=pt, lo=lo, hi=hi, n_with=n1, n_without=n0)

# extremes (enough words to be meaningful)
ext = rel[rel.n_matched >= 12]
clip = lambda s: re.sub(r"\s+", " ", s)[:170]
out["lowest"] = [(round(r.conc_mean, 2), clip(r.content)) for r in ext.nsmallest(6, "conc_mean").itertuples()]
out["highest"] = [(round(r.conc_mean, 2), clip(r.content)) for r in ext.nlargest(6, "conc_mean").itertuples()]

# illustrative sentences (hand-written, NOT data)
EXAMPLES = {
    "Pseudo-profound (Pennycook stimulus)": "Hidden meaning transforms unparalleled abstract beauty.",
    "Pseudo-profound (Pennycook stimulus) 2": "Wholeness quiets infinite phenomena.",
    "Corporate buzzwords": "We leverage synergies to unlock transformative value across our ecosystem.",
    "LinkedIn maxim": "Leadership is not about titles. It is about presence, purpose and courage.",
    "Motivational quote": "A river cuts through rock not because of its power but its persistence.",
    "Mundane": "Had coffee with the team this morning before the meeting.",
    "Coffee + inflated lesson": "Had coffee with my team today. It taught me that leadership is about presence, not titles.",
    "Technical, vague": "Inflation expectations remain anchored despite rate volatility.",
    "Technical, specific": "Five-year inflation expectations held near 2.3% even as ten-year yields swung 40 basis points.",
}
out["examples"] = {k: {kk: (None if pd.isna(vv) else round(float(vv), 2)) for kk, vv in score(v).items()
                       if kk in ("conc_mean", "n_matched", "abstract_ratio")} for k, v in EXAMPLES.items()}

PROC.mkdir(exist_ok=True, parents=True)
inf.drop(columns=["content"]).to_csv(PROC / "concreteness_scores_influencers.csv", index=False)
(EDA / "concreteness_summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("lowest", "highest")}, indent=1, default=float))
print("\nLOWEST concreteness posts:")
for c, t in out["lowest"]:
    print(f"  {c}  {t}")
print("\nHIGHEST concreteness posts:")
for c, t in out["highest"]:
    print(f"  {c}  {t}")
