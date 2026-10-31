#!/usr/bin/env python3
"""Revision aggregator over every results.json under --roots (seed-0 grid in results/,
repeated-CV seeds in results_rep/). Produces, per metric:
  ledger / slack s_hat per seed and per protocol, counting only non-aliased and DISTINCT
  prediction pairs (comment 1); price cells with mean/sd over seeds and resolution vs the
  per-protocol slack; robustness of the headline ratios across metrics, per-class
  decomposition, majority-collapse share, rare-class fold support (comment 2);
  Sec. 8.7 identity on unrounded values (comment 4).
Ordering convention: (relation, lo, hi) means class(lo) is contained in class(hi), so
we expect v(hi) >= v(lo); inversion = max(v(lo) - v(hi), 0)."""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mib_audit import per_class_f1
from mib_metrics import all_metrics, LABEL_METRICS

ARMS = ("MAJORITY", "GEO_ONLY", "WX_ONLY", "MC", "MONO_naive", "MONO_sc",
        "PHYS", "PHYS_sc", "MIB", "MIB_fp", "MIB_fm", "MC_c")
ORDERINGS = [
    ("R1_ablation", "GEO_ONLY", "MONO_naive"), ("R1_ablation", "GEO_ONLY", "MONO_sc"),
    ("R1_ablation", "GEO_ONLY", "MIB"), ("R1_ablation", "GEO_ONLY", "MIB_fp"),
    ("R1_ablation", "GEO_ONLY", "MIB_fm"),
    ("R2_ablation", "MAJORITY", "PHYS"), ("R2_ablation", "MAJORITY", "PHYS_sc"),
    ("R3_free_ctx", "MONO_sc", "MIB"), ("R3_free_ctx", "MONO_sc", "MIB_fp"),
    ("R3_free_ctx", "MONO_sc", "MIB_fm"),
    ("R4_ctx_sign", "MIB_fp", "MIB"), ("R4_ctx_sign", "MIB_fm", "MIB"),
    ("R5_unconstrained", "MONO_naive", "MC"), ("R5_unconstrained", "MONO_sc", "MC"),
    ("R5_unconstrained", "PHYS", "WX_ONLY"), ("R5_unconstrained", "PHYS_sc", "WX_ONLY"),
]
EXTENDED = [("R6_matched", "MIB", "MC_c"), ("R6_matched", "MIB_fp", "MC_c"),
            ("R6_matched", "MIB_fm", "MC_c"), ("R6_matched", "MC", "MC_c")]
# (cell, reference, constrained, ablated model giving the ceiling)
PRICES = [
    ("MONO_naive|MC", "MC", "MONO_naive", "GEO_ONLY"),
    ("MONO_sc|MC", "MC", "MONO_sc", "GEO_ONLY"),
    ("PHYS|WX_ONLY", "WX_ONLY", "PHYS", "MAJORITY"),
    ("PHYS_sc|WX_ONLY", "WX_ONLY", "PHYS_sc", "MAJORITY"),
    ("MIB|MC(paper,unmatched)", "MC", "MIB", "GEO_ONLY"),
    ("MIB|MC_c", "MC_c", "MIB", "GEO_ONLY"),       # ceiling loose: S u {c} ablated
    ("MIB_fp|MC_c", "MC_c", "MIB_fp", "GEO_ONLY"),
    ("MIB_fm|MC_c", "MC_c", "MIB_fm", "GEO_ONLY"),
]
IDX = ["tag", "seed", "protocol"]


def sig(a) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def discover(roots, tags):
    runs, seen = [], set()
    for root in roots:
        for fn in sorted(glob.glob(os.path.join(root, "**", "results.json"), recursive=True)):
            R = json.load(open(fn))
            c = R["config"]
            tag = c.get("tag") or os.path.basename(os.path.dirname(fn))
            seed = int(c.get("seed", 0))
            if tags and tag not in tags:
                continue
            if (tag, seed) in seen:
                print(f"[warn] duplicate {tag}/seed{seed}: {fn} ignored")
                continue
            seen.add((tag, seed))
            runs.append(dict(dir=os.path.dirname(fn), tag=tag, seed=seed, R=R))
    return runs


def collect(runs):
    M, PC, FO = [], [], []
    for r in runs:
        R = r["R"]
        K = len(R["classes"])
        cc = np.asarray(R["class_counts"])
        rare, major = int(cc.argmin()), int(cc.argmax())
        for p in R["protocols"]:
            fn = os.path.join(r["dir"], f"oof_{p}.npz")
            if not os.path.exists(fn):
                print(f"[warn] missing {fn}")
                continue
            z = np.load(fn)
            y = z["y"].astype(int)
            if not any(k.startswith("proba_") for k in z.files):
                print(f"[warn] {fn}: no probabilities (old run) -> label metrics only")
            for arm in ARMS:
                if f"pred_{arm}" not in z.files:
                    continue
                pred = z[f"pred_{arm}"].astype(int)
                P = z[f"proba_{arm}"].astype(np.float64) if f"proba_{arm}" in z.files else None
                s_lab = sig(pred)
                s_prb = sig(P) if P is not None else s_lab
                for k, v in all_metrics(y, pred, P, K).items():
                    M.append(dict(tag=r["tag"], seed=r["seed"], protocol=p, arm=arm, metric=k,
                                  value=v, sig=s_lab if k in LABEL_METRICS else s_prb))
                f1k = per_class_f1(y, pred, K)
                sh = np.bincount(pred, minlength=K) / len(pred)
                for k in range(K):
                    PC.append(dict(tag=r["tag"], seed=r["seed"], protocol=p, arm=arm, cls=k,
                                   cls_name=R["classes"][k], K=K, is_major=(k == major),
                                   f1=f1k[k], pred_share=sh[k], true_share=float(np.mean(y == k))))
            for g in R["protocols"][p].get("fold_ctx", []):
                ts = g.get("test_support")
                if ts is not None:
                    FO.append(dict(tag=r["tag"], seed=r["seed"], protocol=p, fold=g["fold"],
                                   rare_test=int(ts[rare]), test_n=int(sum(ts)),
                                   gate_open=bool(g["ctx_keep"])))
    return pd.DataFrame(M), pd.DataFrame(PC), pd.DataFrame(FO)


def wide(M, metric):
    m = M[M.metric == metric]
    V = m.pivot_table(index=IDX, columns="arm", values="value", aggfunc="first")
    S = m.pivot_table(index=IDX, columns="arm", values="sig", aggfunc="first")
    return V, S


def ledger(V, S, extended):
    parts = []
    for rel, lo, hi in ORDERINGS + (EXTENDED if extended else []):
        if lo not in V.columns or hi not in V.columns:
            continue
        d = pd.DataFrame({"v_lo": V[lo], "v_hi": V[hi], "sig_lo": S[lo], "sig_hi": S[hi]})
        d = d.dropna(subset=["v_lo", "v_hi"])
        d.insert(0, "rel", rel); d.insert(1, "lo", lo); d.insert(2, "hi", hi)
        parts.append(d.reset_index())
    L = pd.concat(parts, ignore_index=True)
    L["gap"] = L.v_hi - L.v_lo
    L["inversion"] = (-L.gap).clip(lower=0.0)
    L["alias"] = L.sig_lo == L.sig_hi
    return L


def slack(L):
    NA = L[~L.alias]
    D = NA.drop_duplicates(["seed", "protocol", "sig_lo", "sig_hi"])
    recs = []
    for s, g in D.groupby("seed"):
        Ls = L[L.seed == s]
        rec = dict(seed=s, tags=",".join(sorted(Ls.tag.unique())), n_formal=len(Ls),
                   n_nonalias=int((~Ls.alias).sum()), n_distinct=len(g),
                   n_inverted=int((g.inversion > 0).sum()), s_hat=float(g.inversion.max()))
        for p, gp in g.groupby("protocol"):
            rec[f"s_hat_{p}"] = float(gp.inversion.max())
            rec[f"n_distinct_{p}"] = len(gp)
        recs.append(rec)
    sp = D.groupby(["seed", "protocol"]).inversion.max().rename("s_p")
    return pd.DataFrame(recs), D, sp


def prices(V, sp):
    W = V.reset_index().merge(sp.reset_index(), on=["seed", "protocol"], how="left")
    W["s_p"] = W["s_p"].fillna(0.0)
    parts = []
    for cell, ref, con, abl in PRICES:
        if not {ref, con, abl} <= set(W.columns):
            continue
        t = W[IDX + ["s_p"]].copy()
        t["cell"] = cell
        t["price"] = W[ref] - W[con]
        t["ceiling"] = W[ref] - W[abl]
        parts.append(t.dropna(subset=["price", "ceiling"]))
    C = pd.concat(parts, ignore_index=True)
    C["ratio"] = C.price / C.ceiling.where(C.ceiling.abs() > 1e-12)
    C["dominance_ok"] = C.price <= C.ceiling + 1e-12      # constrained >= its ablation
    C["resolved"] = C.price.abs() > C.s_p
    C["ceiling_below_slack"] = C.ceiling < C.s_p
    S = C.groupby(["tag", "protocol", "cell"]).agg(
        n_seeds=("seed", "nunique"), price_mean=("price", "mean"), price_sd=("price", "std"),
        price_min=("price", "min"), price_max=("price", "max"),
        ceiling_mean=("ceiling", "mean"), ratio_median=("ratio", "median"),
        s_p_median=("s_p", "median"), frac_resolved=("resolved", "mean"),
        frac_dominance_ok=("dominance_ok", "mean"),
        frac_ceiling_below_slack=("ceiling_below_slack", "mean")).reset_index()
    s0 = C[C.seed == 0].set_index(["tag", "protocol", "cell"]).price.rename("price_seed0")
    S = S.merge(s0.reset_index(), on=["tag", "protocol", "cell"], how="left")
    return C, S


def identity_87(V, S):
    need = {"MIB", "MIB_fp", "MIB_fm", "MONO_sc"}
    if not need <= set(V.columns):
        return pd.DataFrame()
    I = V[S["MIB"] != S["MONO_sc"]].copy()
    trio = I[["MIB", "MIB_fp", "MIB_fm"]]
    I["spread"] = trio.max(1) - trio.min(1)
    I["bound"] = I.MIB - I.MONO_sc
    I["excess"] = I.spread - I.bound
    I["R4_term"] = trio.max(1) - I.MIB        # = R4 inversion, always >= 0
    I["R3_term"] = I.MONO_sc - trio.min(1)    # signed
    I["identity_residual"] = I.excess - (I.R4_term + I.R3_term)
    return I[["spread", "bound", "excess", "R4_term", "R3_term", "identity_residual"]].reset_index()


def robustness(M):
    rows = []
    need = {"MC", "GEO_ONLY", "MONO_sc", "WX_ONLY", "PHYS_sc", "MAJORITY"}
    for metric, m in M.groupby("metric"):
        V = m.pivot_table(index=IDX, columns="arm", values="value", aggfunc="first")
        if not need <= set(V.columns):
            continue
        Q = pd.DataFrame(index=V.index)
        Q["Delta_G"] = V.MC - V.GEO_ONLY
        Q["P_mono_sc"] = V.MC - V.MONO_sc
        Q["Delta_M"] = V.WX_ONLY - V.MAJORITY
        Q["P_phys_sc"] = V.WX_ONLY - V.PHYS_sc
        Q["ratio_shielded"] = Q.P_mono_sc / Q.Delta_G.where(Q.Delta_G.abs() > 1e-12)
        Q["ratio_unshielded"] = Q.P_phys_sc / Q.Delta_M.where(Q.Delta_M.abs() > 1e-12)
        Q["shield_ratio"] = Q.P_phys_sc / Q.P_mono_sc.where(Q.P_mono_sc.abs() > 1e-9)
        Q["geo_recovery_over_majority"] = (V.GEO_ONLY - V.MAJORITY) / (V.MC - V.MAJORITY)
        Q["metric"] = metric
        rows.append(Q.reset_index())
    R = pd.concat(rows, ignore_index=True)
    cols = ["Delta_G", "P_mono_sc", "ratio_shielded", "Delta_M", "P_phys_sc",
            "ratio_unshielded", "shield_ratio", "geo_recovery_over_majority"]
    return R, R.groupby(["tag", "protocol", "metric"])[cols].agg(["mean", "std", "median"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["results", "results_rep"])
    ap.add_argument("--tags", default="", help="comma list of tags; default: all")
    ap.add_argument("--out", default="revision")
    ap.add_argument("--no-extended", action="store_true", help="drop R6 (MC_c) orderings")
    a = ap.parse_args()
    tags = set(filter(None, a.tags.split(",")))
    runs = discover([r for r in a.roots if os.path.isdir(r)], tags)
    assert runs, f"no results.json under {a.roots}"
    per_tag = pd.Series([r["tag"] for r in runs]).value_counts().to_dict()
    print(f"[revision] {len(runs)} runs; seeds per tag: {per_tag}")
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)

    M, PC, FO = collect(runs)
    M.to_csv(f"{a.out}/metrics_long.csv", index=False)
    PC.to_csv(f"{a.out}/per_class_long.csv", index=False)

    for metric in sorted(M.metric.unique()):
        V, S = wide(M, metric)
        L = ledger(V, S, not a.no_extended)
        B, D, sp = slack(L)
        C, Cs = prices(V, sp)
        L.to_csv(f"{a.out}/ledger_{metric}.csv", index=False)
        D[D.inversion > 0].sort_values("inversion", ascending=False).to_csv(
            f"{a.out}/ledger_distinct_inverted_{metric}.csv", index=False)
        B.to_csv(f"{a.out}/slack_by_seed_{metric}.csv", index=False)
        C.to_csv(f"{a.out}/price_cells_long_{metric}.csv", index=False)
        Cs.to_csv(f"{a.out}/price_cells_{metric}.csv", index=False)
        if metric in ("macro_f1", "neg_rps"):
            print(f"\n=== [{metric}] slack s_hat per split seed (distinct, non-aliased pairs) ===")
            print(B.round(4).to_string(index=False))
            sc = [c for c in B.columns if c.startswith("s_hat")]
            print(f"\n=== [{metric}] s_hat distribution over seeds ===")
            print(B[sc].describe(percentiles=[.5, .9]).round(4).to_string())
            print(f"\n=== [{metric}] price cells, tag=ctx (over seeds) ===")
            print(Cs[Cs.tag == "ctx"].round(4).to_string(index=False))
        if metric == "macro_f1":
            I = identity_87(V, S)
            I.to_csv(f"{a.out}/identity_8_7.csv", index=False)
            if len(I):
                print("\n=== Sec. 8.7: spread - bound = (max-MIB) + (MONO_sc-min), unrounded ===")
                print(f"max |identity residual| = {I.identity_residual.abs().max():.2e}")
                print(I[I.seed == I.seed.min()].round(5).to_string(index=False))

    Rr, Rs = robustness(M)
    Rr.to_csv(f"{a.out}/metric_robustness_long.csv", index=False)
    Rs.to_csv(f"{a.out}/metric_robustness.csv")
    print("\n=== headline quantities across metrics (tag=ctx; mean/sd/median over seeds) ===")
    if "ctx" in Rs.index.get_level_values("tag"):
        print(Rs.xs("ctx", level="tag").round(4).to_string())

    pv = PC.pivot_table(index=IDX + ["cls", "cls_name", "K"], columns="arm",
                        values="f1", aggfunc="first").reset_index()
    if {"WX_ONLY", "PHYS_sc", "MC", "MONO_sc"} <= set(pv.columns):
        pv["contrib_P_mono_sc"] = (pv.MC - pv.MONO_sc) / pv.K
        pv["contrib_P_phys_sc"] = (pv.WX_ONLY - pv.PHYS_sc) / pv.K
        dec = pv.groupby(["tag", "protocol", "cls", "cls_name"])[
            ["contrib_P_mono_sc", "contrib_P_phys_sc"]].agg(["mean", "std"])
        dec.to_csv(f"{a.out}/price_by_class.csv")
        print("\n=== macro-F1 price decomposed by class (sum over classes = price) ===")
        print(dec.round(4).to_string())

    maj = PC[PC.is_major].pivot_table(index=["tag", "protocol"], columns="arm",
                                      values="pred_share", aggfunc="mean")
    maj.to_csv(f"{a.out}/majority_prediction_share.csv")
    print("\n=== share of OOF predictions = majority class (collapse diagnostic) ===")
    print(maj.round(3).to_string())

    if len(FO):
        rare = FO.groupby(["tag", "protocol"]).rare_test.agg(
            min="min", median="median", max="max",
            folds_lt20=lambda v: int((v < 20).sum()),
            folds_zero=lambda v: int((v == 0).sum()), n_folds="count")
        rare.to_csv(f"{a.out}/rare_class_test_counts.csv")
        print("\n=== rarest-class records per test fold (all seeds) ===")
        print(rare.to_string())
    print(f"\n[revision] all tables -> {a.out}/")


if __name__ == "__main__":
    main()