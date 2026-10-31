"""Alternative metrics (reviewer comment 2). Every metric is oriented higher-is-better,
so 'price = reference - constrained' keeps the same sign convention as macro-F1."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score, balanced_accuracy_score
from mib_audit import macro_f1

NAN = float("nan")
LABEL_METRICS = ("macro_f1", "qwk", "neg_mae", "bal_acc")
PROB_METRICS = ("neg_rps", "neg_logloss", "neg_brier", "cum_auc")
ALL_METRICS = LABEL_METRICS + PROB_METRICS


def _norm(P):
    P = np.clip(np.asarray(P, float), 1e-12, 1.0)
    return P / P.sum(1, keepdims=True)


def neg_rps(y, P, K):
    """Ranked probability score (proper and ordinal), negated."""
    C = np.cumsum(P, 1)[:, :-1]
    T = (y[:, None] <= np.arange(K - 1)[None, :]).astype(float)
    return -float(((C - T) ** 2).sum(1).mean() / (K - 1))


def neg_logloss(y, P, K):
    return float(np.log(P[np.arange(len(y)), y]).mean())


def neg_brier(y, P, K):
    return -float(((P - np.eye(K)[y]) ** 2).sum(1).mean())


def cum_auc(y, P, K):
    """Mean AUC of the cumulative splits y > k, scored by P(y > k)."""
    v = []
    for k in range(K - 1):
        t = (y > k).astype(int)
        if 0 < t.mean() < 1:
            v.append(roc_auc_score(t, P[:, k + 1:].sum(1)))
    return float(np.mean(v)) if v else NAN


def qwk(y, pred, K):
    with np.errstate(all="ignore"):
        v = cohen_kappa_score(y, pred, weights="quadratic", labels=list(range(K)))
    return float(np.nan_to_num(v))


def all_metrics(y, pred, P, K) -> dict[str, float]:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    out = dict(macro_f1=macro_f1(y, pred, K), qwk=qwk(y, pred, K),
               neg_mae=-float(np.abs(y - pred).mean()),
               bal_acc=float(balanced_accuracy_score(y, pred)))
    if P is not None:
        P = _norm(P)
        out.update(neg_rps=neg_rps(y, P, K), neg_logloss=neg_logloss(y, P, K),
                   neg_brier=neg_brier(y, P, K), cum_auc=cum_auc(y, P, K))
    return out