# Concreteness EDA

Run on 2026-10-06. Scripts: `eda/concreteness/concreteness_eda.py` (scores and statistics), `eda/concreteness/concreteness_charts.py` (figures).
Per-post scores: `data/processed/concreteness_scores_influencers.csv`. Raw numbers: `eda/concreteness/concreteness_summary.json`.

## Question

Is word concreteness useful for a bullshit detector, and is "flowery" language less concrete?

## Method

- **Lexicon:** Brysbaert et al. (2014) concreteness norms. 39,954 entries, rated from 1 (abstract) to 5 (concrete).
- **Which words count:** only content words (nouns, verbs, adjectives, adverbs), selected using the lexicon's own part-of-speech column, with stopwords removed. Function words would distort the score: "the" is rated 1.43.
- **Matching word forms:** inflected words are matched by stripping common endings until a dictionary entry is found ("humbled" → "humble", "hiring" → "hire"). spaCy was not available. Two-word expressions are matched first.
- **Cleaning:** URLs, @mentions, hashtags and the trailing "…see more" are removed before scoring.
- **Post score:** the mean concreteness of matched content words. A score is treated as reliable only when at least 5 words matched.
- **Flowery words:** a hand-curated list of 79 inspirational, grandiose or corporate-buzz words (e.g. *journey, purpose, empower, unlock, synergy, leadership*). The list is subjective; it is defined in the script.
- **Fair test:** flowery words are abstract by definition, so flowery posts would score as abstract even if nothing else differed. Concreteness is therefore also computed **with the flowery words removed**.
- **Comparing framing patterns:** each post is compared with the same author's other posts, so author style is held constant. 95% confidence intervals come from resampling authors.

## Findings

**1. Lexicon coverage is good.**
26,578 of 31,996 influencer posts (83%) have at least 5 matched words. The median post has 13 matched words, and 87.5% of its content words are in the lexicon.

**2. Post concreteness falls in a narrow band.**
Influencer posts average 3.00 (SD 0.30), and the middle 80% fall between 2.63 and 3.38. Ordinary users (Figshare) average 3.08, and the Zenodo software-engineering posts average 3.07. Averaging over words squeezes most posts towards the middle of the scale.

**3. Concreteness reflects the post, not the author.**
Which author wrote the post explains only 6.4% of the variance. Truncated posts score the same as complete ones (3.00 vs 3.00).

**4. Flowery posts are less concrete, but mostly because the flowery words themselves are abstract** (Figure 2).

| Flowery words per 100 content words | Share of posts | Concreteness (all words) | Concreteness (flowery words removed) |
|---|---|---|---|
| None | 69% | 3.03 | 3.03 |
| Light (avg 2.9) | 16% | 2.98 | 3.00 |
| Heavy (avg 9.3) | 15% | 2.88 | 2.96 |

- **All words:** heavy users score 0.15 lower, about half a standard deviation.
- **Flowery words removed:** the gap shrinks to 0.07.
- **Rank correlation between flowery density and concreteness:** −0.17 on all words, −0.07 with flowery words removed, and −0.05 within the same author.

**5. Among the framing patterns, only "not X, but Y" maxims are clearly more abstract** (Figure 3). Differences are measured against the same author's other posts.

| Pattern | Difference | 95% CI | Posts |
|---|---|---|---|
| "It's not about X, it's about Y" | **−0.081** | [−0.126, −0.041] | 290 |
| Inspirational hashtags | −0.021 | [−0.031, −0.011] | 2,020 |
| Lesson framing | −0.019 | [−0.042, −0.001] | 816 |
| Announcement ("thrilled/humbled to") | +0.010 | [−0.013, +0.035] | 627 |
| Two or more exclamation marks | +0.034 | [+0.016, +0.052] | 1,840 |
| Contains a number | +0.037 | [+0.027, +0.051] | 12,323 |

- **Every effect is small:** the largest is about a quarter of a standard deviation.
- **Some patterns go the other way.** Announcements and exclamation-heavy posts tend to be about concrete events, such as a new job or a celebration.

**6. Both ends of the scale are mixed.**
- **Lowest-scoring posts:** motivational filler ("TAKE RESPONSIBILITY FOR YOUR ACTIONS…", "Have you gotten outside your comfort zone?"), alongside **substantive** abstract posts about epistemology and AI regulation.
- **Highest-scoring posts:** a vivid workplace anecdote (the parable genre), mundane posts ("Happy Fried Chicken Day"), and a substantive post listing a seminar's named officials.

**7. On illustrative sentences, bullshit and non-bullshit overlap** (Figure 1; hand-written examples, not data).
- **Low scores, both kinds:** pseudo-profound sentences score 2.21–2.30, but a meaningful technical sentence scores 2.42. That is lower than corporate buzzwords at 2.58.
- **High scores, both kinds:** a concrete metaphor ("A river cuts through rock…") scores 3.72, and the mundane coffee post scores 3.84.
- **Middle:** coffee plus an inflated lesson scores 3.12, inside the normal range.

## Conclusions

- **Do not use concreteness as a rule or threshold that decides the label.** It wrongly penalises technical language, and concrete metaphors and anecdotes get past it.
- **Keep it as one model feature.** It carries a weak but real signal, and is strongest for maxim-style framing.
- **"Flowery = abstract" is mostly circular.** It holds largely because the flowery words themselves are abstract. Counting flowery words directly and also using concreteness partly measures the same thing twice.
- **The real test is still to come.** Once human gold labels exist, measure how well concreteness separates bullshit from non-bullshit (ROC-AUC) on the gold set.

## Caveats

- Word forms are matched approximately (suffix rules, not a full lemmatiser), and irregular forms are missed.
- The flowery list is hand-picked and subjective.
- The influencer data comes from about 70 prominent accounts posting before 2021, so it may not represent LinkedIn today.
- All results are correlations. The illustrative sentences were written by hand, not sampled.
