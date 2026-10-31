"""Compact run summary for the manuscript + falsifier checklist.
Revision: MC_c arm, alternative metrics, per-fold rare-class support,
C4 only when the bootstrap is enabled, C7 (MC_c == MC when the gate is empty)."""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd

CONSTR = ("MONO_naive", "MONO_sc", "PHYS", "PHYS_sc", "MIB_fp", "MIB_fm")
FREE   = ("MC", "WX_ONLY", "GEO_ONLY", "MC_c")
KEY    = ("MAJORITY", "GEO_ONLY", "WX_ONLY", "MC", "MONO_naive", "MONO_sc",
          "PHYS", "PHYS_sc", "MIB", "MIB_fp", "MIB_fm", "MC_c")
MET    = ("macro_f1", "neg_rps", "neg_logloss", "neg_brier", "qwk", "neg_mae",
          "cum_auc", "bal_acc")


def f(x, n=4):
    return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    a = ap.parse_args()
    R = json.load(open(f"{a.res}/results.json"))
    c, protos = R["config"], list(R["protocols"])
    ok, bad, skip = [], [], []

    print("=" * 78)
    print(f"RES={a.res}   N={R['N']}  K={len(R['classes'])}  classes={R['classes']}")
    print(f"counts={R['class_counts']}  drivers={R['drivers']}")
    print(f"ctx_candidates={R['ctx_candidates']}  ctx_keep_global={R['ctx_keep']}  "
          f"has_time={R['has_time']}")
    print(f"folds={c['folds']} block_deg={c['block_deg']} blocks={R['n_blocks']} "
          f"coord_blocks={R['n_coord_blocks']} seed={c['seed']} boot={c['boot']}")
    print(f"gate={c['force_gate']}/{c['screen_scope']}  solar_as_driver={c['solar_as_driver']}  "
          f"spatial_cv={c['spatial_cv']}  protocols={protos}")
    if R["screen"]:
        print("SCREEN: " + " | ".join(
            f"{s['feature']}: R2={s['r2_on_drivers']:.4f} rel={s['resid_rel_std']:.4f} "
            f"exo={s['exogenous']}" for s in R["screen"]))
    else:
        print("SCREEN: кандидатов нет")

    for p in protos:
        P = R["protocols"][p]
        print("-" * 78)
        print(f"[{p}]  n_boot={P['n_boot']}  aliases={P['alias_hits'] or '{}'}")
        print("  gate по фолдам: " + ", ".join(
            f"f{g['fold']}:{g['ctx_keep'] or '[]'}" for g in P["fold_ctx"]))
        print("  signs по фолдам: " + " ".join(
            "".join(("+" if s[d] > 0 else "-") for d in R["drivers"]) for s in P["fold_signs"]))
        fd = pd.DataFrame(P["fold_f1"]).groupby("model").f1.agg(["mean", "std"])
        hdr = f"  {'model':11s} {'F1oof':>7s} {'fold+-sd':>16s}"
        for r in R["table_refs"]:
            hdr += f" {'d_'+r:>9s} {'[95% CI]':>19s}"
        print(hdr + f" {'VRown':>7s} {'VRnaive':>8s} {'VRscr':>7s}  perclass")
        for m in [x for x in KEY if x in P["f1"]]:
            line = (f"  {m:11s} {P['f1'][m]:7.4f} "
                    f"{fd.loc[m,'mean']:.4f}+-{fd.loc[m,'std']:.4f}")
            for r in R["table_refs"]:
                cc = P["cost"].get(r, {}).get(m)
                if m == r or cc is None:
                    line += f" {'—':>9s} {'':>19s}"
                elif cc["identical"]:
                    line += f" {'ident':>9s} {'':>19s}"
                else:
                    line += (f" {cc['delta']:+9.4f} "
                             f"[{cc['ci_lo']:+.4f},{cc['ci_hi']:+.4f}]"
                             f"{'*' if (cc['ci_lo']>0 or cc['ci_hi']<0) else ' '}")
            v = P["vr"][m]
            line += (f" {f(v['own']['overall']):>7s} {f(v['naive']['overall']):>8s} "
                     f"{f(v['screened']['overall']):>7s}  "
                     f"{np.round(P['per_class_f1'][m], 3).tolist()}")
            print(line)
        vp = P["vr_partial"]
        print(f"  VR_partial(MC): naive={f(vp['naive']['overall'])} "
              f"screened={f(vp['screened']['overall'])} n_act={vp['naive']['n_active']}")
        print("  VR_partial per driver (naive/screened): " + " ".join(
            f"{d}={f(vp['naive']['per_driver'].get(d), 3)}/"
            f"{f(vp['screened']['per_driver'].get(d), 3)}" for d in R["drivers"]))
        # ---- revision additions ----
        if "metrics" in P:
            print("  metrics (higher = better): " + " ".join(f"{k:>11s}" for k in MET))
            for m in [x for x in KEY if x in P["metrics"]]:
                print(f"    {m:11s} " + " ".join(
                    f"{P['metrics'][m].get(k, float('nan')):11.4f}" for k in MET))
        sup = [g["test_support"] for g in P["fold_ctx"] if "test_support" in g]
        if sup:
            rare = int(np.argmin(R["class_counts"]))
            print(f"  test support of rarest class '{R['classes'][rare]}' per fold: "
                  f"{[s[rare] for s in sup]}")

    s = pd.DataFrame(R["sign_stability"])
    print("-" * 78)
    print("SIGNS (все протоколы):")
    for d in R["drivers"]:
        sd = s[s.driver == d]
        print(f"  {d:7s} agree_naive={int(sd.agrees_with_naive.sum())}/{len(sd)} "
              f"identified={int(sd.identified.sum())}/{len(sd)} "
              f"n_plus={int((sd.sign_screened > 0).sum())}/{len(sd)} "
              f"mean_eff={sd['mean'].mean():+.5f}")

    # ---------- falsifiers ----------
    print("-" * 78)
    print("CHECKS:")
    (ok if "WX_ONLY" in R["protocols"][protos[0]]["f1"] else bad).append(
        "C1 WX_ONLY присутствует (код новый)")
    (ok if set(R["table_refs"]) <= set(R["protocols"][protos[0]]["cost"]) else bad).append(
        "C2 CI посчитаны против всех табличных референсов")
    v0 = R["protocols"][protos[0]]["vr"]
    (ok if all(v0[m]["own"]["overall"] < 1e-9 for m in CONSTR if m in v0) else bad).append(
        "C3a VR_own==0 у всех ограниченных моделей (сертификат цел)")
    (ok if all(np.isnan(v0[m]["own"]["overall"]) for m in FREE if m in v0) else bad).append(
        "C3b VR_own==NaN у неограниченных (знаки не заявлены)")
    if c["boot"] > 0:
        (ok if all(R["protocols"][p]["n_boot"] >= max(100, c["boot"] // 10)
                   for p in protos) else bad).append("C4 бутстрап не выродился")
    else:
        skip.append("C4 bootstrap disabled (--boot 0): no CIs in this run")
    for p in protos:
        z = np.load(f"{a.res}/oof_{p}.npz")
        empty = all(not g["ctx_keep"] for g in R["protocols"][p]["fold_ctx"])
        if empty:
            eq = all(np.array_equal(z[f"pred_{m}"], z["pred_MONO_sc"])
                     for m in ("MIB", "MIB_fp", "MIB_fm"))
            (ok if eq else bad).append(f"C5[{p}] шлюз пуст => MIB* тождественны MONO_sc")
            if "pred_MC_c" in z.files:
                (ok if np.array_equal(z["pred_MC_c"], z["pred_MC"]) else bad).append(
                    f"C7[{p}] empty gate => MC_c identical to MC")
        (ok if (z["pred_MC"] >= 0).all() else bad).append(f"C6[{p}] OOF покрывает все объекты")
    for t in ok: print(f"  PASS  {t}")
    for t in bad: print(f"  FAIL  {t}")
    for t in skip: print(f"  SKIP  {t}")

    # ---------- headline numbers ----------
    print("-" * 78)
    print("HEADLINE:")
    if "iid" in protos and "spatial" in protos:
        i, sp = R["protocols"]["iid"]["f1"]["MC"], R["protocols"]["spatial"]["f1"]["MC"]
        print(f"  MC: iid={i:.4f} -> spatial={sp:.4f}  (утечка разбиения {i-sp:+.4f})")
    for p in protos:
        P = R["protocols"][p]
        cm = P["cost"].get("MC", {})
        for m in ("MONO_sc", "MIB"):
            if m in cm:
                x = cm[m]
                print(f"  [{p}] {m} vs MC: {x['delta']:+.4f} "
                      f"[{x['ci_lo']:+.4f},{x['ci_hi']:+.4f}] p={x['p']:.3f}")
        cw = P["cost"].get("WX_ONLY", {})
        if "PHYS_sc" in cw:
            x = cw["PHYS_sc"]
            print(f"  [{p}] PHYS_sc vs WX_ONLY: {x['delta']:+.4f} "
                  f"[{x['ci_lo']:+.4f},{x['ci_hi']:+.4f}] p={x['p']:.3f}")
        cc = P["cost"].get("MC_c", {})
        if "MIB" in cc:
            x = cc["MIB"]
            print(f"  [{p}] MIB vs MC_c (feature-matched): {x['delta']:+.4f} "
                  f"[{x['ci_lo']:+.4f},{x['ci_hi']:+.4f}]")
        mib = [P["f1"][m] for m in ("MIB", "MIB_fp", "MIB_fm") if m in P["f1"]]
        if mib:
            print(f"  [{p}] разброс MIB по ctx_sign: {min(mib):.4f}..{max(mib):.4f} "
                  f"(размах {max(mib)-min(mib):.4f})")
    print("=" * 78)


if __name__ == "__main__":
    main()