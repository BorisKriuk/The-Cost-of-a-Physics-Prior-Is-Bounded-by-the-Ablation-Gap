#!/usr/bin/env python3
"""VR controls (reviewer comment 3), same OrdinalGBM / partial-effect machinery:
  real_plus_noise    : real target, drivers+geo + two known-null features
                       (permuted temp, Gaussian), declared sign +1 for the nulls;
  synthetic_monotone : ordinal target monotone in every driver BY CONSTRUCTION
                       (textbook signs), same class shares; scored against the true
                       signs (VR should be ~0) and against flipped signs (~1).
Reported per eps: active fraction, VR, magnitude-weighted VR, mean effect."""
from __future__ import annotations
import argparse, contextlib, io, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mib_data import load_dataset, DRIVER_SIGN
from mib_models import OrdinalGBM
from mib_audit import (step_sizes, partial_effects, vr_profile_counts, add_profiles,
                       vr_profile_rows, VR_EPS)
from mib_study import splits


def zrank(x):
    r = pd.Series(x).rank(pct=True).to_numpy()
    return (r - r.mean()) / r.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--merge", default="")
    ap.add_argument("--map", nargs="*", default=[])
    ap.add_argument("--block-deg", type=float, default=2.0)
    ap.add_argument("--protocol", default="spatial", choices=["iid", "point", "spatial"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spatial-cv", choices=["stratgroup", "group"], default="stratgroup")
    ap.add_argument("--eps", type=float, nargs="+", default=list(VR_EPS))
    ap.add_argument("--syn-noise", type=float, default=1.0,
                    help="noise sd in the synthetic latent (signal sd ~ sqrt(#drivers))")
    ap.add_argument("--out", default="revision")
    a = ap.parse_args()

    ov = dict(kv.split("=", 1) for kv in a.map)
    with contextlib.redirect_stdout(io.StringIO()):
        ds = load_dataset(a.csv, a.order, a.merge, ov, a.block_deg)
    rng = np.random.default_rng(a.seed)
    df = ds.df.copy()
    df["noise_perm"] = rng.permutation(df["temp"].to_numpy())
    df["noise_gauss"] = rng.standard_normal(len(df))
    noise = ["noise_perm", "noise_gauss"]
    naive = {d: DRIVER_SIGN[d] for d in ds.drivers}

    z = sum(naive[d] * zrank(df[d].to_numpy()) for d in ds.drivers)
    z = z + a.syn_noise * rng.standard_normal(len(df))
    shares = np.bincount(ds.y, minlength=ds.K) / len(ds.y)
    y_syn = np.searchsorted(np.quantile(z, np.cumsum(shares)[:-1]), z, side="right").astype(int)
    print(f"[vr-control] N={len(df)} K={ds.K} protocol={a.protocol} drivers={ds.drivers} "
          f"synthetic class counts={np.bincount(y_syn, minlength=ds.K).tolist()}")

    exps = [
        ("real_plus_noise", ds.drivers + ds.geo + noise, ds.y,
         {"textbook": {**naive, **{n: +1 for n in noise}}}),
        ("synthetic_monotone", ds.drivers + ds.geo, y_syn,
         {"true_signs": naive, "flipped_signs": {d: -s for d, s in naive.items()}}),
    ]
    folds = splits(ds, a.protocol, a.folds, a.seed, a.spatial_cv)
    h = step_sizes(df, ds.drivers + noise)

    rows = []
    for name, feats, y, decls in exps:
        acc = {k: {} for k in decls}
        cols = sorted({j for d in decls.values() for j in d})
        for f, (tr, te) in enumerate(folds):
            m = OrdinalGBM(feats, {}, ds.K).fit(df.iloc[tr], y[tr])
            Xte = df.iloc[te]
            s = m.score(Xte)
            rs = float(s.max() - s.min())
            eff = partial_effects(m, Xte, cols, h)
            for k, d in decls.items():
                add_profiles(acc[k], vr_profile_counts(eff, rs, d, a.eps))
            print(f"  [{name}] fold {f+1}/{len(folds)} done")
        for k in decls:
            rows += [dict(experiment=name, declared=k, **r) for r in vr_profile_rows(acc[k])]

    T = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"vr_control_{a.protocol}.csv")
    T.to_csv(fn, index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    for col in ("vr", "vr_weighted", "active_frac", "mean_effect"):
        print(f"\n=== {col} by activity threshold eps (protocol {a.protocol}) ===")
        print(T.pivot_table(index=["experiment", "declared", "driver"], columns="eps",
                            values=col, aggfunc="first").round(4).to_string())
    print("\nReading: if driver VR on the real target ~ noise VR (~0.5 or undefined), VR is"
          "\nnot evidence against the prior; the synthetic rows show what a correct prior"
          f"\n(VR~0) and a wrong one (VR~1) look like under the same machinery.\n-> {fn}")


if __name__ == "__main__":
    main()