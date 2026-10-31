#!/usr/bin/env python3
"""Data description for the paper (reviewer comment 5): filtering steps, class balance,
coordinate duplication, meteorological-join granularity, spatial blocks, units,
exogeneity of the auxiliary channel, absence of a date column."""
from __future__ import annotations
import argparse, contextlib, io, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mib_data import load_dataset, resolve_columns, _parse_merge, _majority_macro_f1
from mib_audit import exogeneity_screen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--merge", default="")
    ap.add_argument("--map", nargs="*", default=[])
    ap.add_argument("--block-deg", type=float, default=2.0)
    ap.add_argument("--out", default="revision")
    a = ap.parse_args()
    ov = dict(kv.split("=", 1) for kv in a.map)

    raw = pd.read_csv(a.csv)
    with contextlib.redirect_stdout(io.StringIO()):
        ds = load_dataset(a.csv, a.order, a.merge, ov, a.block_deg)
        col = resolve_columns(raw, ov)
    df, y = ds.df, ds.y
    order = [c.strip() for c in a.order.split(",")]
    mm = _parse_merge(a.merge)
    lab = raw[col["label"]].astype(str).str.strip()
    lab_m = lab.replace(mm) if mm else lab
    in_order = lab_m.isin(order)
    need = [col[k] for k in ds.drivers + ds.geo]
    complete = in_order & raw[need].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    cnt = np.bincount(y, minlength=ds.K)
    rare = int(cnt.argmin())

    L = ["# Data report", "",
         f"Source `{a.csv}`: {raw.shape[0]} rows, {raw.shape[1]} columns: "
         + ", ".join(f"`{c}`" for c in raw.columns), "", "## Column roles"]
    L += [f"- {k}: `{col[k]}`" for k in ds.drivers + ds.ctx_candidates + ds.geo + ["label"]
          if k in col]
    L += ["", "## Raw label values"] + [f"- {k}: {v}" for k, v in lab.value_counts().items()]
    if "type_id" in raw.columns:
        L += ["", "type_id x label:", "```",
              raw.groupby(["type_id", col["label"]]).size().to_string(), "```"]
    L += ["", "## Filtering",
          f"- raw records: {len(raw)}",
          f"- merge rule: {mm or 'none'}; ordinal order: {order}",
          f"- label in order (after merge): {int(in_order.sum())}",
          f"- ... with complete drivers and coordinates: {int(complete.sum())}",
          f"- final N after min-class filter: {len(y)}",
          "", "## Classes"]
    L += [f"- {c}: {n} ({n / len(y):.4f})" for c, n in zip(ds.classes, cnt)]
    L += [f"- majority-class macro-F1: {_majority_macro_f1(cnt):.4f}"]

    coord = df.lat.round(4).astype(str) + "_" + df.lon.round(4).astype(str)
    cg = pd.factorize(coord)[0]
    cm = np.bincount(cg)
    L += ["", "## Coordinates",
          f"- unique coordinate pairs (4 dp): {cm.size} for {len(df)} records",
          f"- records at a coordinate shared with others: {(cm[cg] > 1).mean():.4f}",
          f"- max records per coordinate: {cm.max()}",
          f"- lat [{df.lat.min():.3f}, {df.lat.max():.3f}], "
          f"lon [{df.lon.min():.3f}, {df.lon.max():.3f}]"]

    met = [c for c in ds.drivers + ds.ctx_candidates if c in df.columns]
    key = df[met].round(6).astype(str).agg("|".join, axis=1)
    mg = pd.factorize(key)[0]
    mc = np.bincount(mg)
    span = df.assign(_g=mg).groupby("_g").agg(
        n=("lat", "size"), la0=("lat", "min"), la1=("lat", "max"),
        lo0=("lon", "min"), lo1=("lon", "max"))
    multi = span[span.n > 1]
    per_pt = pd.Series(mg).groupby(cg).nunique()
    pts_multi = np.flatnonzero(cm > 1)
    L += ["", "## Meteorological join (records with identical weather tuples)",
          f"- variables: {met}",
          f"- distinct weather tuples: {mc.size} for {len(df)} records",
          f"- records sharing their tuple with others: {(mc[mg] > 1).mean():.4f}",
          f"- median / max records per tuple: {int(np.median(mc))} / {mc.max()}"]
    if len(multi):
        L += [f"- tuples with >1 record: {len(multi)}; median lat span "
              f"{(multi.la1 - multi.la0).median():.3f} deg, median lon span "
              f"{(multi.lo1 - multi.lo0).median():.3f} deg, max lat span "
              f"{(multi.la1 - multi.la0).max():.3f} deg"]
    if len(pts_multi):
        L += [f"- coordinates with >1 record: {len(pts_multi)}; share of them with a single "
              f"weather tuple: {(per_pt.loc[pts_multi] == 1).mean():.4f}"]
    L += ["- Reading: if repeated records at one coordinate share one tuple and tuples cover",
          "  a fixed lat/lon cell, the weather is a gridded or static join rather than a",
          "  time-of-detection join. State the source, resolution and matching rule."]

    L += ["", "## Spatial blocks",
          "| deg | blocks | blocks with rarest class | median records/block | largest block share |",
          "|---|---|---|---|---|"]
    for d in (1, 2, 5, 10):
        b = pd.factorize(np.floor(df.lat / d).astype(int).astype(str) + "_" +
                         np.floor(df.lon / d).astype(int).astype(str))[0]
        bc = np.bincount(b)
        L.append(f"| {d} | {bc.size} | {np.unique(b[y == rare]).size} | "
                 f"{int(np.median(bc))} | {bc.max() / len(b):.3f} |")

    L += ["", "## Variables (check units)", "```",
          df[met].describe().T.round(3).to_string(), "```",
          "- share of exact zeros: " + ", ".join(f"{c}={(df[c] == 0).mean():.3f}" for c in met)]
    scr = exogeneity_screen(df, ds.ctx_candidates, ds.drivers)
    L += ["", "## Exogeneity screen of auxiliary candidates (full data)", "```",
          scr.to_string(index=False) if len(scr) else "(no candidates)", "```"]
    dates = [c for c in raw.columns if re.search(r"date|time|day|month|year|acq|season", c, re.I)]
    L += ["", "## Time information"]
    L += ([f"- date-like columns: {dates}"] if dates else
          ["- No date/time column in the file: temporal and spatio-temporal protocols are",
           "  unavailable and seasonality cannot be controlled. The paper must explain how",
           "  weather was matched 'at the time of detection' (e.g. joined upstream, date dropped)."])

    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, "data_report.md")
    open(fn, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {fn}")


if __name__ == "__main__":
    main()