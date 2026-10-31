"""Main run: validation-protocol ladder x model zoo -> <out>/results.json|csv|npz.

Revision (MAIS):
  * --seed is the CV *split* seed; HGB's seed is fixed (mib_models.HGB), so seed 0
    reproduces the paper's folds. --tag is stored in results.json;
  * --boot 0 / --boot-sign 0 skip the bootstraps (repeated-CV seeds);
  * OOF class probabilities are saved (proba_<arm>);
  * new arm MC_c (feature-matched reference for MIB*);
  * table_metrics.csv (8 metrics, per-class F1, predicted-class shares);
  * table_vr_profile.csv (VR at several eps, magnitude-weighted VR);
  * per-fold train/test class support in results.json -> fold_ctx.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import StratifiedKFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except ImportError:
    HAS_SGK = False

from mib_data import load_dataset, available_protocols, DRIVER_SIGN
from mib_models import build_zoo, dedupe_zoo
from mib_audit import (exogeneity_screen, screen_signs, step_sizes, partial_effects,
                       vr_counts_from_effects, vr_from_counts, effect_summary,
                       vr_profile_counts, add_profiles, vr_profile_rows, VR_EPS,
                       macro_f1, per_class_f1, bootstrap_f1_matrix, paired_from_matrix)
from mib_metrics import all_metrics, ALL_METRICS

TAGS = ("own", "naive", "screened")
REFS = ["MC", "WX_ONLY", "GEO_ONLY", "MONO_sc", "MC_c"]   # for CIs
TABLE_REFS = ["MC", "WX_ONLY", "GEO_ONLY"]               # for the main table
FREE_ARMS = ("MC", "WX_ONLY", "GEO_ONLY", "MC_c")
CONSTRAINED_ARMS = ("MONO_naive", "MONO_sc", "PHYS", "PHYS_sc")


def cv_groups(ds, protocol):
    if protocol == "iid":            return None
    if protocol == "point":          return ds.coord_blocks
    if protocol == "spatial":        return ds.blocks
    if protocol == "temporal":       return ds.time_blocks
    if protocol == "spatiotemporal":
        return pd.factorize(pd.Series(ds.blocks).astype(str) + "_" +
                            pd.Series(ds.time_blocks).astype(str))[0]
    raise ValueError(protocol)


def boot_groups(ds, protocol, mode):
    """Bootstrap cluster: spatial blocks (data are always spatially dependent)."""
    if mode == "none":
        return None
    if protocol in ("temporal",) and ds.time_blocks is not None:
        return ds.time_blocks
    return ds.blocks


def splits(ds, protocol, folds, seed, spatial_cv):
    g = cv_groups(ds, protocol)
    if g is None:
        return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(ds.df, ds.y))
    nu = len(np.unique(g))
    assert nu >= folds, f"groups {nu} < folds {folds} (protocol {protocol})"
    cv = (StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
          if (spatial_cv == "stratgroup" and HAS_SGK) else GroupKFold(folds))
    return list(cv.split(ds.df, ds.y, groups=g))


def run_protocol(ds, protocol, global_ctx, a, h):
    all_names = list(build_zoo(ds, {d: +1 for d in ds.drivers}, global_ctx).keys())
    N = len(ds.y)
    oof = {n: np.full(N, -1, int) for n in all_names}
    prob = {n: np.full((N, ds.K), np.nan) for n in all_names}
    vr = {n: {t: {} for t in TAGS} for n in all_names}
    prof = {n: {t: {} for t in TAGS} for n in all_names}
    naive = {d: s for d, s in DRIVER_SIGN.items() if d in ds.drivers}
    fold_rows, sign_rows, alias_hits, fold_signs, fold_ctx = [], [], {}, [], []

    for f, (tr, te) in enumerate(splits(ds, protocol, a.folds, a.seed, a.spatial_cv)):
        Rtr, Rte, ytr, yte = ds.df.iloc[tr], ds.df.iloc[te], ds.y[tr], ds.y[te]
        sup = np.bincount(yte, minlength=ds.K).tolist()
        sup_tr = np.bincount(ytr, minlength=ds.K).tolist()

        # --- exogeneity gate: decided on the TRAIN fold ---
        if a.force_gate == "closed" or not ds.ctx_candidates:
            ctxf, sfold = [], []
        elif a.screen_scope == "global":
            ctxf, sfold = list(global_ctx), []
        else:
            sd = exogeneity_screen(Rtr, ds.ctx_candidates, ds.drivers, a.max_r2, a.min_rel)
            ctxf = sd.loc[sd.exogenous, "feature"].tolist()
            sfold = sd.to_dict("records")
        fold_ctx.append(dict(fold=f + 1, ctx_keep=ctxf, screen=sfold,
                             test_support=sup, train_support=sup_tr))

        # --- unconstrained reference MC: also the basis of sign screening ---
        ref = build_zoo(ds, naive, ctxf)["MC"].fit(Rtr, ytr)
        eff_tr = partial_effects(ref, Rtr, ds.drivers, h)
        scr = screen_signs(eff_tr)
        fold_signs.append(scr)
        for j in ds.drivers:
            st = effect_summary(eff_tr[j], ds.blocks[tr], a.boot_sign, a.seed + f)
            sign_rows.append(dict(protocol=protocol, fold=f + 1, driver=j,
                                  sign_naive=naive[j], sign_screened=scr[j],
                                  agrees_with_naive=bool(naive[j] == scr[j]), **st))

        zoo = build_zoo(ds, scr, ctxf)
        canon, alias = dedupe_zoo(zoo)
        for n, c in alias.items():
            alias_hits[f"{n}->{c}"] = alias_hits.get(f"{n}->{c}", 0) + 1

        fitted, fold_vr, fold_prof = {}, {}, {}
        for n, pipe in canon.items():
            fitted[n] = ref if n == "MC" else pipe.fit(Rtr, ytr)
            oof[n][te] = fitted[n].predict(Rte)
            prob[n][te] = fitted[n].predict_proba(Rte)

        for n, pipe in fitted.items():
            s = pipe.score(Rte)
            rngS = float(s.max() - s.min())
            eff = partial_effects(pipe, Rte, ds.drivers, h)
            decl = {"own": {j: int(pipe.signs.get(j, 0)) for j in ds.drivers},
                    "naive": naive, "screened": scr}
            fold_vr[n] = {t: vr_counts_from_effects(eff, rngS, d) for t, d in decl.items()}
            fold_prof[n] = {t: vr_profile_counts(eff, rngS, d, a.vr_eps) for t, d in decl.items()}

        for n, c in alias.items():
            oof[n][te] = oof[c][te]
            prob[n][te] = prob[c][te]
            fold_vr[n] = fold_vr[c]
            fold_prof[n] = fold_prof[c]

        # Fold-level falsifier: empty context => MIB* == MONO_sc and MC_c == MC.
        if not ctxf:
            for m in ("MIB", "MIB_fp", "MIB_fm"):
                assert np.array_equal(oof[m][te], oof["MONO_sc"][te]), \
                    f"fold {f+1}: empty context but {m} != MONO_sc"
            assert np.array_equal(oof["MC_c"][te], oof["MC"][te]), \
                f"fold {f+1}: empty context but MC_c != MC"

        for n in all_names:
            for t in TAGS:
                for j, (v, q) in fold_vr[n][t].items():
                    pv, pq = vr[n][t].get(j, (0, 0))
                    vr[n][t][j] = (pv + v, pq + q)
                add_profiles(prof[n][t], fold_prof[n][t])
            fold_rows.append(dict(protocol=protocol, fold=f + 1, model=n, n_test=len(te),
                                  f1=macro_f1(yte, oof[n][te], ds.K)))

        print(f"  [{protocol}] fold {f+1}/{a.folds}  ctx={ctxf or '[]'}  signs={scr}  "
              f"test_support={sup}" + (f"  aliases={sorted(alias)}" if alias else ""))

    for n in all_names:
        assert (oof[n] >= 0).all(), f"{n}: incomplete OOF coverage"
        assert np.isfinite(prob[n]).all(), f"{n}: incomplete OOF probabilities"
    return oof, prob, vr, prof, fold_rows, sign_rows, alias_hits, fold_signs, fold_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--order", required=True, help="increasing severity, comma-separated")
    ap.add_argument("--merge", default="", help='"A+B=C;D+E=F"')
    ap.add_argument("--map", nargs="*", default=[], help="canon=column")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-deg", type=float, default=2.0)
    ap.add_argument("--time-block-days", type=int, default=7)
    ap.add_argument("--min-class", type=int, default=30)
    ap.add_argument("--dedup", choices=["none", "exact", "coord"], default="none")
    ap.add_argument("--solar-as-driver", action="store_true",
                    help="insolation as 5th driver (naive sign +1) instead of context channel")
    ap.add_argument("--protocols", default="auto",
                    help="auto|iid,point,spatial[,temporal,spatiotemporal]")
    ap.add_argument("--seed", type=int, default=0, help="CV split seed (0 = paper)")
    ap.add_argument("--tag", default="", help="run label (ctx, b1, ...)")
    ap.add_argument("--boot", type=int, default=2000, help="0 = no CI bootstrap")
    ap.add_argument("--boot-sign", type=int, default=400, help="0 = no sign bootstrap")
    ap.add_argument("--vr-eps", type=float, nargs="+", default=list(VR_EPS),
                    help="activity thresholds for table_vr_profile.csv")
    ap.add_argument("--max-r2", type=float, default=0.90)
    ap.add_argument("--min-rel", type=float, default=0.10)
    ap.add_argument("--force-gate", choices=["auto", "closed"], default="auto")
    ap.add_argument("--screen-scope", choices=["fold", "global"], default="fold")
    ap.add_argument("--spatial-cv", choices=["stratgroup", "group"], default="stratgroup")
    ap.add_argument("--cluster-boot", choices=["always", "none"], default="always")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if not a.tag:
        a.tag = os.path.basename(os.path.normpath(a.out))
    os.makedirs(a.out, exist_ok=True)

    ov = dict(kv.split("=", 1) for kv in a.map)
    ds = load_dataset(a.csv, a.order, a.merge, ov, a.block_deg, a.min_class,
                      a.solar_as_driver, a.time_block_days, a.dedup)
    h = step_sizes(ds.df, ds.drivers)

    avail = available_protocols(ds)
    protos = avail if a.protocols == "auto" else [p.strip() for p in a.protocols.split(",")]
    bad = [p for p in protos if p not in avail]
    assert not bad, f"protocols {bad} unavailable on this data; available {avail}"
    print(f"\n[protocols] {protos}  (available: {avail})  tag={a.tag}  split_seed={a.seed}")

    # --- global screen: report / figure 3 only ---
    scr_df = exogeneity_screen(ds.df, ds.ctx_candidates, ds.drivers, a.max_r2, a.min_rel)
    print("\n[screen] exogeneity of context candidates (full data, report only):")
    print(scr_df.to_string(index=False) if len(scr_df) else "  (no candidates)")
    global_ctx = [] if a.force_gate == "closed" else \
        (scr_df.loc[scr_df.exogenous, "feature"].tolist() if len(scr_df) else [])
    print(f"[gate] global {'CLOSED' if not global_ctx else 'OPEN'} -> ctx_keep={global_ctx}")
    print(f"[gate] in-CV decision scope='{a.screen_scope}'")
    if not global_ctx:
        print("[gate] consequence: MIB, MIB_fp, MIB_fm == MONO_sc and MC_c == MC (aliases)")
    if a.force_gate == "closed":
        assert global_ctx == [], f"gate closed but context non-empty: {global_ctx}"

    res = {"config": vars(a), "classes": ds.classes, "drivers": ds.drivers,
           "ctx_candidates": ds.ctx_candidates, "N": int(len(ds.y)),
           "n_blocks": int(len(np.unique(ds.blocks))),
           "n_coord_blocks": int(len(np.unique(ds.coord_blocks))),
           "has_time": bool(ds.has_time), "ctx_keep": global_ctx,
           "refs": REFS, "table_refs": TABLE_REFS,
           "class_counts": np.bincount(ds.y, minlength=ds.K).tolist(),
           "screen": scr_df.to_dict("records"), "protocols": {}}

    rows, sign_all, metric_rows, vrp_rows = [], [], [], []
    for proto in protos:
        print(f"\n=== protocol: {proto} ===")
        (oof, prob, vr, prof, fold_rows, sign_rows, alias_hits, fs, fctx) = \
            run_protocol(ds, proto, global_ctx, a, h)
        np.savez_compressed(f"{a.out}/oof_{proto}.npz", y=ds.y, blocks=ds.blocks,
                            coord_blocks=ds.coord_blocks,
                            **{f"pred_{k}": v for k, v in oof.items()},
                            **{f"proba_{k}": v.astype(np.float32) for k, v in prob.items()})

        gb = boot_groups(ds, proto, "none" if a.cluster_boot == "none" else "always")
        f1 = {n: macro_f1(ds.y, p, ds.K) for n, p in oof.items()}
        pcf = {n: per_class_f1(ds.y, p, ds.K) for n, p in oof.items()}
        mats, nb = bootstrap_f1_matrix(ds.y, oof, gb, ds.K, a.boot, a.seed)
        cost = {r: paired_from_matrix(ds.y, oof, mats, ds.K, r) for r in REFS if r in oof}
        vro = {n: {t: dict(zip(("overall", "per_driver", "n_active"), vr_from_counts(c)))
                   for t, c in d.items()} for n, d in vr.items()}
        met = {n: all_metrics(ds.y, oof[n], prob[n], ds.K) for n in oof}
        share = {n: (np.bincount(oof[n], minlength=ds.K) / len(ds.y)).tolist() for n in oof}

        for m in CONSTRAINED_ARMS:
            o = vro[m]["own"]["overall"]
            assert np.isnan(o) or o < 1e-9, f"certificate of {m} violated (VR_own={o})"
        for m in FREE_ARMS:
            assert np.isnan(vro[m]["own"]["overall"]), f"{m} declares no signs: VR_own must be NaN"

        # consistency: VR profile at eps=1e-3 must reproduce VR_partial exactly
        if any(abs(e - 1e-3) < 1e-15 for e in a.vr_eps):
            chk = [r for r in vr_profile_rows(prof["MC"]["naive"])
                   if r["driver"] == "ALL" and abs(r["eps"] - 1e-3) < 1e-15]
            v0 = vro["MC"]["naive"]["overall"]
            if chk and not (np.isnan(v0) and np.isnan(chk[0]["vr"])) and abs(chk[0]["vr"] - v0) > 1e-12:
                print(f"  [warn] VR profile mismatch: {chk[0]['vr']} vs {v0}")

        fdf = pd.DataFrame(fold_rows)
        agg = fdf.groupby("model").f1.agg(["mean", "std", "min", "max"])
        res["protocols"][proto] = dict(
            f1=f1, per_class_f1=pcf, cost=cost, vr=vro, n_boot=nb,
            vr_partial={t: vro["MC"][t] for t in ("naive", "screened")},
            fold_signs=fs, fold_ctx=fctx, alias_hits=alias_hits, fold_f1=fold_rows,
            metrics=met, pred_share=share)
        sign_all += sign_rows

        for n in oof:
            metric_rows.append(dict(tag=a.tag, seed=a.seed, protocol=proto, model=n, **met[n],
                                    **{f"f1_{c_}": v for c_, v in zip(ds.classes, pcf[n])},
                                    **{f"share_{c_}": v for c_, v in zip(ds.classes, share[n])}))
            for t in TAGS:
                for r_ in vr_profile_rows(prof[n][t]):
                    vrp_rows.append(dict(tag=a.tag, seed=a.seed, protocol=proto,
                                         model=n, decl=t, **r_))

        maj, geo = f1["MAJORITY"], f1["GEO_ONLY"]
        print(f"\n  baseline: MAJORITY={maj:.4f}  GEO_ONLY={geo:.4f}  "
              f"WX_ONLY={f1['WX_ONLY']:.4f}  (valid bootstrap replicates {nb})")
        print("  F1(macro), pooled OOF and fold spread:")
        for n in f1:
            print(f"    {n:11s} oof={f1[n]:.4f}  fold={agg.loc[n,'mean']:.4f}"
                  f"+-{agg.loc[n,'std']:.4f}  [{agg.loc[n,'min']:.4f},{agg.loc[n,'max']:.4f}]  "
                  f"perclass={np.round(pcf[n],3).tolist()}")
        for r, cd in cost.items():
            print(f"  paired differences vs {r} (95% cluster bootstrap):")
            for n, c in cd.items():
                tag = "  SIGNIF" if (c["ci_lo"] > 0 or c["ci_hi"] < 0) else ""
                ident = "  (identical)" if c["identical"] else ""
                print(f"    {n:11s} d={c['delta']:+.4f} "
                      f"[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] p={c['p']:.3f}{tag}{ident}")
        print("  metrics (higher = better): " + " ".join(f"{k:>11s}" for k in ALL_METRICS))
        for n in met:
            print(f"    {n:11s} " + " ".join(f"{met[n].get(k, np.nan):11.4f}" for k in ALL_METRICS))
        vp = res["protocols"][proto]["vr_partial"]
        print(f"  VR_partial(MC): naive={vp['naive']['overall']:.4f} "
              f"(active {vp['naive']['n_active']}) | screened="
              f"{vp['screened']['overall']:.4f} (active {vp['screened']['n_active']})")
        print("  VR_e2e (own / vs naive / vs screened):")
        for n, d in vro.items():
            fmt = lambda x: "n/a " if np.isnan(x) else f"{x:.4f}"
            print(f"    {n:11s} own={fmt(d['own']['overall'])} "
                  f"naive={fmt(d['naive']['overall'])} screened={fmt(d['screened']['overall'])}")
        if alias_hits:
            print(f"  aliases (model->canonical, #folds): {alias_hits}")

        alias_of = {k.split('->')[0]: k.split('->')[1]
                    for k, v in alias_hits.items() if v == a.folds}
        for n in f1:
            row = dict(protocol=proto, model=n, alias_of=alias_of.get(n, ""),
                       f1_oof=f1[n], f1_fold_mean=float(agg.loc[n, "mean"]),
                       f1_fold_sd=float(agg.loc[n, "std"]),
                       vr_own=vro[n]["own"]["overall"], n_act_own=vro[n]["own"]["n_active"],
                       vr_vs_naive=vro[n]["naive"]["overall"],
                       vr_vs_screened=vro[n]["screened"]["overall"],
                       **{f"f1_{c_}": v for c_, v in zip(ds.classes, pcf[n])})
            for r, cd in cost.items():
                c = cd.get(n)
                row[f"dF1_vs_{r}"] = 0.0 if c is None else c["delta"]
                row[f"ci_lo_{r}"]  = 0.0 if c is None else c["ci_lo"]
                row[f"ci_hi_{r}"]  = 0.0 if c is None else c["ci_hi"]
                row[f"p_{r}"]      = 1.0 if c is None else c["p"]
                row[f"ident_{r}"]  = bool(c["identical"]) if c is not None else False
            rows.append(row)

    res["sign_stability"] = sign_all
    with open(f"{a.out}/results.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    pd.DataFrame(rows).to_csv(f"{a.out}/table_main.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(f"{a.out}/table_metrics.csv", index=False)
    pd.DataFrame(vrp_rows).to_csv(f"{a.out}/table_vr_profile.csv", index=False)
    sdf = pd.DataFrame(sign_all)
    sdf.to_csv(f"{a.out}/table_signs.csv", index=False)
    scr_df.to_csv(f"{a.out}/table_screen.csv", index=False)
    pd.DataFrame([dict(protocol=p, **fc) for p in res["protocols"]
                  for fc in res["protocols"][p]["fold_ctx"]]).to_csv(
        f"{a.out}/table_gate_by_fold.csv", index=False)

    print("\n[signs] sign stability (folds of all protocols):")
    for j in ds.drivers:
        s = sdf[sdf.driver == j]
        print(f"   {j:7s} agrees with textbook {int(s.agrees_with_naive.sum())}/{len(s)} folds, "
              f"CI excludes 0 in {int(s.identified.sum())}/{len(s)}, "
              f"'+' signs {int((s.sign_screened > 0).sum())}/{len(s)}")
    print(f"\n[done] -> {a.out}/results.json, table_main.csv, table_metrics.csv, "
          f"table_vr_profile.csv, table_signs.csv, table_screen.csv, "
          f"table_gate_by_fold.csv, oof_*.npz")


if __name__ == "__main__":
    main()