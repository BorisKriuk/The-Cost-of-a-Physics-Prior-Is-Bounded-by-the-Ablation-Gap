"""Ordinal GBM with HARD monotone constraints + context residualization + pipeline.

Revision (MAIS):
  * Pipeline / MajorityPipeline expose predict_proba -> proper and ordinal scores;
  * new arm MC_c = unconstrained on drivers+geo+ctx: the feature-matched reference for
    MIB* (Remark 24). With an empty context its signature equals MC's, so dedupe_zoo
    turns it into an exact alias of MC.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from mib_data import design, DRIVER_SIGN

HGB = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
           l2_regularization=1.0, early_stopping=False, random_state=0)


class _Const:
    def __init__(self, p): self.p = float(p)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


class OrdinalGBM:
    """Cumulative model: K-1 binary P(y>k). monotonic_cst is a GUARANTEE, not a penalty."""

    def __init__(self, features, signs, K, **kw):
        self.features, self.signs, self.K = list(features), dict(signs), int(K)
        self.kw = {**HGB, **kw}

    @property
    def cst(self) -> np.ndarray:
        return np.array([int(self.signs.get(f, 0)) for f in self.features], dtype=int)

    def fit(self, X: pd.DataFrame, y):
        Xv, y = X[self.features].to_numpy(float), np.asarray(y)
        cst = self.cst
        mc = cst if np.any(cst != 0) else None
        self.models_ = []
        for k in range(self.K - 1):
            t = (y > k).astype(int)
            if t.min() == t.max():
                self.models_.append(_Const(t.mean()))
                continue
            m = HistGradientBoostingClassifier(monotonic_cst=mc, **self.kw)
            m.fit(Xv, t)
            self.models_.append(m)
        return self

    def cum(self, X: pd.DataFrame) -> np.ndarray:
        Xv = X[self.features].to_numpy(float)
        S = np.column_stack([m.predict_proba(Xv)[:, 1] for m in self.models_])
        return np.minimum.accumulate(S, axis=1)   # isotonization in k keeps monotonicity in x

    def score(self, X) -> np.ndarray:
        return self.cum(X).sum(1)

    def predict_proba(self, X) -> np.ndarray:
        S = self.cum(X)
        P = np.empty((len(S), self.K))
        P[:, 0] = 1 - S[:, 0]
        for k in range(1, self.K - 1):
            P[:, k] = S[:, k - 1] - S[:, k]
        P[:, self.K - 1] = S[:, -1]
        P = np.clip(P, 0, None)
        return P / P.sum(1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(1)


class CtxResidualizer:
    """ctx_c = c - A b, A = quadratic design in the drivers. Coefficients from train only."""

    def __init__(self, keep: list[str], drivers: list[str]):
        self.keep, self.drivers, self.coef = list(keep), list(drivers), {}

    @property
    def cols(self) -> list[str]:
        return ["ctx_" + c for c in self.keep]

    def fit(self, df: pd.DataFrame):
        if not self.keep:
            return self
        A = design(df, self.drivers)
        for c in self.keep:
            self.coef[c] = np.linalg.lstsq(A, df[c].to_numpy(float), rcond=None)[0]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.keep:
            return pd.DataFrame(index=df.index)
        A = design(df, self.drivers)
        return pd.DataFrame({"ctx_" + c: df[c].to_numpy(float) - A @ self.coef[c]
                             for c in self.keep}, index=df.index)


class Pipeline:
    """raw -> [base_feats | ctx_*] -> OrdinalGBM. The e2e audit goes through transform."""
    kind = "gbm"

    def __init__(self, name, base_feats, ctx_keep, signs, K, drivers, ctx_sign=0):
        self.name, self.base_feats, self.ctx_keep = name, list(base_feats), list(ctx_keep)
        self.signs = {k: int(v) for k, v in dict(signs).items() if k in set(base_feats)}
        self.K, self.drivers, self.ctx_sign = int(K), list(drivers), int(ctx_sign)
        self.res = CtxResidualizer(self.ctx_keep, self.drivers)

    @property
    def signature(self) -> tuple:
        cs = self.ctx_sign if self.ctx_keep else 0
        return (self.kind, tuple(self.base_feats), tuple(self.ctx_keep), cs,
                tuple(sorted((k, v) for k, v in self.signs.items() if v != 0)))

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([raw[self.base_feats], self.res.transform(raw)], axis=1)

    @property
    def feature_signs(self) -> dict[str, int]:
        return {**self.signs, **{c: self.ctx_sign for c in self.res.cols}}

    def fit(self, raw: pd.DataFrame, y):
        self.res.fit(raw)
        X = self.transform(raw)
        self.model = OrdinalGBM(list(X.columns), self.feature_signs, self.K).fit(X, y)
        return self

    def score(self, raw) -> np.ndarray:
        return self.model.score(self.transform(raw))

    def predict_proba(self, raw) -> np.ndarray:
        return self.model.predict_proba(self.transform(raw))

    def predict(self, raw) -> np.ndarray:
        return self.model.predict(self.transform(raw))


class MajorityPipeline:
    """Baseline: constant = majority class of the train fold; probabilities = train prior."""
    kind = "majority"

    def __init__(self, K):
        self.name, self.K = "MAJORITY", int(K)
        self.base_feats, self.ctx_keep, self.signs, self.ctx_sign = [], [], {}, 0

    @property
    def signature(self) -> tuple:
        return (self.kind, (), (), 0, ())

    def fit(self, raw, y):
        cnt = np.bincount(np.asarray(y), minlength=self.K).astype(float)
        self.cls_ = int(cnt.argmax())
        self.prior_ = cnt / cnt.sum()
        return self

    def score(self, raw) -> np.ndarray:
        return np.zeros(len(raw))

    def predict_proba(self, raw) -> np.ndarray:
        return np.tile(self.prior_, (len(raw), 1))

    def predict(self, raw) -> np.ndarray:
        return np.full(len(raw), self.cls_, int)


def build_zoo(ds, screened: dict[str, int], ctx_keep: list[str]) -> dict:
    """WX_ONLY: unconstrained weather, reference for PHYS*. MC: unconstrained weather+geo,
    reference for MONO_*. MC_c: unconstrained weather+geo+ctx, feature-matched reference
    for MIB*. MC must precede MC_c so that MC_c -> MC is the alias when ctx is empty."""
    naive = {d: s for d, s in DRIVER_SIGN.items() if d in ds.drivers}
    P, base, drv, geo, K = Pipeline, ds.base_feats, ds.drivers, ds.geo, ds.K
    return {
        "MAJORITY":   MajorityPipeline(K),
        "GEO_ONLY":   P("GEO_ONLY",   geo,  [],       {},       K, drv),
        "WX_ONLY":    P("WX_ONLY",    drv,  [],       {},       K, drv),
        "MC":         P("MC",         base, [],       {},       K, drv),
        "MONO_naive": P("MONO_naive", base, [],       naive,    K, drv),
        "MONO_sc":    P("MONO_sc",    base, [],       screened, K, drv),
        "PHYS":       P("PHYS",       drv,  [],       naive,    K, drv),
        "PHYS_sc":    P("PHYS_sc",    drv,  [],       screened, K, drv),
        "MIB":        P("MIB",        base, ctx_keep, screened, K, drv, ctx_sign=0),
        "MIB_fp":     P("MIB_fp",     base, ctx_keep, screened, K, drv, ctx_sign=+1),
        "MIB_fm":     P("MIB_fm",     base, ctx_keep, screened, K, drv, ctx_sign=-1),
        "MC_c":       P("MC_c",       base, ctx_keep, {},       K, drv, ctx_sign=0),
    }


def dedupe_zoo(zoo: dict) -> tuple[dict, dict]:
    """Identical signature => fit once. Catches MIB*==MONO_sc and MC_c==MC with a closed
    gate, and MONO_sc==MONO_naive in folds where screening agrees with the textbook."""
    canon, alias, seen = {}, {}, {}
    for n, p in zoo.items():
        s = p.signature
        if s in seen:
            alias[n] = seen[s]
        else:
            seen[s] = n
            canon[n] = p
    return canon, alias