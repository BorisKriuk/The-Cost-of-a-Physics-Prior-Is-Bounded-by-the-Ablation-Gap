"""Exogeneity screen, sign screen, VR_partial / VR_e2e, paired cluster bootstrap.

Revision (MAIS): vr_profile_* = VR at several activity thresholds, active fraction,
magnitude-weighted VR and mean effect (comment 3); bootstrap can be disabled (B=0)
for repeated-CV seeds (comment 1).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from mib_data import design

NAN = float("nan")
VR_EPS = (1e-3, 1e-2, 5e-2)


# ---------- exogeneity of the context channel ----------
def exogeneity_screen(df, ctx_candidates, drivers, max_r2=0.90, min_rel=0.10) -> pd.DataFrame:
    if not ctx_candidates:
        return pd.DataFrame(columns=["feature", "r2_on_drivers", "resid_rel_std", "exogenous"])
    A, rows = design(df, drivers), []
    for c in ctx_candidates:
        y = df[c].to_numpy(float)
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        r = y - A @ b
        r2 = float(1 - r.var() / max(y.var(), 1e-300))
        rel = float(r.std() / max(y.std(), 1e-300))
        rows.append(dict(feature=c, r2_on_drivers=r2, resid_rel_std=rel,
                         exogenous=bool(r2 <= max_r2 and rel >= min_rel)))
    return pd.DataFrame(rows)


# ---------- partial effects ----------
def _fd(pipe, raw, col, h) -> np.ndarray:
    up, dn = raw.copy(), raw.copy()
    up[col] = up[col].to_numpy(float) + h
    dn[col] = dn[col].to_numpy(float) - h
    return pipe.score(up) - pipe.score(dn)


def partial_effects(pipe, raw, drivers, h) -> dict[str, np.ndarray]:
    return {j: _fd(pipe, raw, j, h[j]) for j in drivers}


def screen_signs(effects: dict[str, np.ndarray]) -> dict[str, int]:
    return {j: (int(np.sign(d.mean())) or +1) for j, d in effects.items()}


def step_sizes(df, drivers, frac=0.05) -> dict[str, float]:
    h = {}
    for j in drivers:
        x = df[j].to_numpy(float)
        q = np.subtract(*np.percentile(x, [75, 25]))
        h[j] = float(frac * (q if q > 0 else max(x.std(), 1e-6)))
    return h


def vr_counts_from_effects(effects, score_range, declared, eps=1e-3) -> dict[str, tuple[int, int]]:
    tau = eps * max(float(score_range), 1e-12)
    out = {}
    for j, d in effects.items():
        sg = int(declared.get(j, 0))
        if sg == 0:
            out[j] = (0, 0)
            continue
        act = np.abs(d) > tau
        out[j] = (int((act & (np.sign(d) != sg)).sum()), int(act.sum()))
    return out


def vr_from_counts(cnt: dict[str, tuple[int, int]]) -> tuple[float, dict[str, float], int]:
    per = {j: (v / a if a else NAN) for j, (v, a) in cnt.items()}
    V = sum(v for v, _ in cnt.values())
    A = sum(a for _, a in cnt.values())
    return (V / A if A else NAN), per, int(A)


# ---------- VR profile (revision, comment 3) ----------
def vr_profile_counts(effects, score_range, declared, eps_grid=VR_EPS) -> dict[str, np.ndarray]:
    """Per (eps, feature): [n_eval, n_active, n_viol, sum|D| active, sum|D| violating, sum D].
    At eps=1e-3 the counts are identical to vr_counts_from_effects."""
    out = {}
    for eps in eps_grid:
        tau = float(eps) * max(float(score_range), 1e-12)
        for j, d in effects.items():
            sg = int(declared.get(j, 0))
            if sg == 0:
                continue
            d = np.asarray(d, float)
            a = np.abs(d)
            act = a > tau
            viol = act & (np.sign(d) != sg)
            out[f"{float(eps)!r}|{j}"] = np.array(
                [d.size, act.sum(), viol.sum(), a[act].sum(), a[viol].sum(), d.sum()], float)
    return out


def add_profiles(acc: dict, new: dict) -> dict:
    for k, v in new.items():
        acc[k] = acc[k] + v if k in acc else v.copy()
    return acc


def _profile_row(eps, feat, v, with_mean=True) -> dict:
    n, a, vi, wa, wv, sd = (float(x) for x in v)
    return dict(eps=eps, driver=feat, n_eval=int(n), n_active=int(a), n_viol=int(vi),
                active_frac=a / n if n else NAN,
                vr=vi / a if a else NAN,
                vr_weighted=wv / wa if wa > 0 else NAN,
                mean_effect=(sd / n if n else NAN) if with_mean else NAN)


def vr_profile_rows(acc: dict) -> list[dict]:
    rows, tot = [], {}
    for key in sorted(acc):
        e, j = key.split("|", 1)
        e = float(e)
        rows.append(_profile_row(e, j, acc[key]))
        tot[e] = tot[e] + acc[key] if e in tot else acc[key].copy()
    rows += [_profile_row(e, "ALL", v, with_mean=False) for e, v in sorted(tot.items())]
    return rows


# ---------- bootstrap helpers ----------
def _units(groups, n):
    if groups is None:
        return [np.array([i]) for i in range(n)]
    return [np.flatnonzero(groups == u) for u in np.unique(groups)]


def effect_summary(d: np.ndarray, groups: np.ndarray, B: int = 400, seed: int = 0) -> dict:
    d = np.asarray(d, float)
    out = dict(mean=float(d.mean()), frac_pos=float((d > 0).mean()),
               ci_lo=NAN, ci_hi=NAN, identified=False)
    if B <= 0 or groups is None:
        return out
    rng = np.random.default_rng(seed)
    units = _units(groups, len(d))
    U = len(units)
    ms = np.empty(B)
    for b in range(B):
        ii = np.concatenate([units[k] for k in rng.integers(0, U, U)])
        ms[b] = d[ii].mean()
    lo, hi = np.percentile(ms, [2.5, 97.5])
    out.update(ci_lo=float(lo), ci_hi=float(hi), identified=bool(lo > 0 or hi < 0))
    return out


# ---------- metrics ----------
def macro_f1(y, p, K=None) -> float:
    """Macro-F1 over ALL K classes (an absent class contributes 0, like zero_division=0)."""
    y, p = np.asarray(y, int), np.asarray(p, int)
    K = int(max(y.max(), p.max()) + 1) if K is None else int(K)
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0).mean())


def per_class_f1(y, p, K) -> list[float]:
    y, p = np.asarray(y, int), np.asarray(p, int)
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return [float(v) for v in np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)]


def bootstrap_f1_matrix(y, preds: dict, groups, K, B=2000, seed=0):
    """B cluster replicates shared by all models. B<=0 disables the bootstrap."""
    if B <= 0:
        return None, 0
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    units = _units(groups, len(y))
    U = len(units)
    names = list(preds)
    acc = {n: [] for n in names}
    for _ in range(B):
        ii = np.concatenate([units[k] for k in rng.integers(0, U, U)])
        yy = y[ii]
        if np.bincount(yy, minlength=K).min() == 0:
            continue
        for n in names:
            acc[n].append(macro_f1(yy, preds[n][ii], K))
    mats = {n: np.asarray(v, float) for n, v in acc.items()}
    nb = len(next(iter(mats.values())))
    assert nb >= max(100, B // 10), f"bootstrap degenerated: {nb} valid replicates of {B}"
    return mats, nb


def paired_from_matrix(y, preds: dict, mats, K: int, ref: str) -> dict:
    out = {}
    f1_ref = macro_f1(y, preds[ref], K)
    for n in preds:
        if n == ref:
            continue
        if np.array_equal(preds[n], preds[ref]):
            out[n] = dict(delta=0.0, ci_lo=0.0, ci_hi=0.0, p=1.0, n_boot=0, identical=True)
            continue
        delta = float(macro_f1(y, preds[n], K) - f1_ref)
        if mats is None:
            out[n] = dict(delta=delta, ci_lo=NAN, ci_hi=NAN, p=NAN, n_boot=0, identical=False)
            continue
        d = mats[n] - mats[ref]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        out[n] = dict(delta=delta, ci_lo=float(lo), ci_hi=float(hi), p=p,
                      n_boot=int(d.size), identical=False)
    return out