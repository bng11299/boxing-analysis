"""Uncertainty for the numbers this project reports.

A 0.10 macro-F1 gap between two models on 976 validation clips is not a claim
until it carries an interval. These are paired bootstraps: both models are
resampled on the *same* draw of validation clips, so the interval describes the
difference between them rather than the sum of their separate variances.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


# Which metric the interval is about has to be stated, because the two can
# disagree about whether a claim survives. ST-GCN beats the static-pose floor
# by 0.189 macro F1 but only 0.061 accuracy; an interval on the first says
# nothing about the second, and the second is the one that looked fragile.
METRICS = {"macro_f1": _macro_f1, "accuracy": _accuracy}


def bootstrap_gap_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    metric: str = "macro_f1",
) -> dict:
    """Paired bootstrap CI for metric(a) - metric(b).

    Resamples validation clips with replacement, recomputing both models'
    scores on each identical draw. Returns the point estimate, the percentile
    interval, the metric it applies to, and whether it excludes zero.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of "
                         f"{sorted(METRICS)}")
    score = METRICS[metric]
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, pred_a and pred_b must be the same length")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = score(y_true, pred_a) - score(y_true, pred_b)

    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # A resample can omit a rare class entirely; zero_division=0 keeps that
        # draw usable rather than discarding it and biasing the interval.
        diffs[i] = score(y_true[idx], pred_a[idx]) - score(y_true[idx], pred_b[idx])

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
        "alpha": alpha,
    }


def sign_test_p(n_pos: int, n_neg: int) -> float:
    """Exact two-sided sign test; ties are dropped before calling.

    With nine videos the smallest p it can return is 2/512 = 0.0039, so it can
    confirm a gap every video agrees on, and is silent about anything less.
    """
    from math import comb
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def binomial_greater_p(k: int, n: int, p: float) -> float:
    """Exact one-sided binomial test: P(X >= k) for X ~ Binomial(n, p), the chance
    of k or more successes in n if each succeeds with probability p.

    Summed term by term in log space, so it holds for large n where comb(n, i)
    would overflow a float.
    """
    from math import exp, fsum, lgamma, log, log1p
    for name, v in (("k", k), ("n", n)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} must be a whole number, got {v!r}")
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise ValueError(f"p must be a number in [0, 1], got {p!r}")
    if k == 0 or p == 1:
        return 1.0
    if p == 0:
        return 0.0
    lp, lq, top = log(p), log1p(-p), lgamma(n + 1)
    tail = fsum(exp(top - lgamma(i + 1) - lgamma(n - i + 1) + i * lp + (n - i) * lq) for i in range(k, n + 1))
    return float(min(1.0, tail))


def fisher_greater_p(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact test on the 2x2 table [[a, b], [c, d]]: the chance, with
    both row and column totals held fixed, of a diagonal at least this heavy (`a` this
    large). A table with an empty row or column carries no evidence: p = 1."""
    from scipy.stats import fisher_exact
    for name, v in (("a", a), ("b", b), ("c", c), ("d", d)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"cell {name} must be a whole number >= 0, got {v!r}")
    return float(fisher_exact([[a, b], [c, d]], alternative="greater").pvalue)


def per_group_gaps(y_true, pred_a, pred_b, groups, score) -> dict:
    """score(a) - score(b) inside each group, with a sign test across groups.

    The clip-level bootstrap treats clips as independent, and clips from one
    video are not. The unit that actually varies independently here is the
    video, so this is the check on that interval: does the gap point the same
    way video by video? A gap of exactly zero is counted as a tie, not as a
    disagreement.
    """
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    if not (len(y_true) == len(pred_a) == len(pred_b) == len(groups)):
        raise ValueError("y_true, pred_a, pred_b and groups must be the same length")
    gaps = {}
    for g in dict.fromkeys(groups.tolist()):  # first-seen order, deterministic
        m = groups == g
        gaps[g] = float(score(y_true[m], pred_a[m]) - score(y_true[m], pred_b[m]))
    pos = sum(v > 0 for v in gaps.values())
    neg = sum(v < 0 for v in gaps.values())
    return {"gaps": gaps, "positive": pos, "negative": neg,
            "ties": len(gaps) - pos - neg,
            "sign_test_p": sign_test_p(pos, neg)}


def stratified_subsample(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Indices for a class-proportional subsample, at least one clip per class.

    Plain uniform sampling at small fractions can drop the rarest class
    entirely, which would make the learning curve measure class coverage rather
    than training-set size.
    """
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * frac)))
        keep.append(idx[:k])
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out
