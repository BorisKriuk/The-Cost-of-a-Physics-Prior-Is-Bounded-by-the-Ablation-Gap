#!/usr/bin/env bash
# Reproduce the paper and the MAIS revision.
#   ./run.sh check | ctx | grid | repeats | data | vr | revision | digest | figs | all
# Env overrides: PY CSV ORDER3 MERGE3 ORDER4 R(split seeds incl. 0) BOOT FORCE=1
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python -u}
export PYTHONUNBUFFERED=1
CSV=${CSV:-data/FiresRu.csv}
ORDER3=${ORDER3:-"Controlled burn,Uncontrolled burn,Wildfire"}
MERGE3=${MERGE3:-"Forest fire+Natural fire+Peat fire=Wildfire"}
ORDER4=${ORDER4:-"Controlled burn,Uncontrolled burn,Forest fire,Natural fire"}
R=${R:-10}
BOOT=${BOOT:-2000}
FORCE=${FORCE:-0}
GRID="ctx b1 b2 b5 b10 closed drv glob k5"
mkdir -p results results_rep logs revision

study() {   # study TAG SEED OUTDIR [extra mib_study args]
  local tag=$1 seed=$2 out=$3
  shift 3
  local -a cls=(--order "$ORDER3" --merge "$MERGE3")
  local -a cfg
  case "$tag" in
    ctx|b2) cfg=(--block-deg 2) ;;
    b1)     cfg=(--block-deg 1) ;;
    b5)     cfg=(--block-deg 5) ;;
    b10)    cfg=(--block-deg 10) ;;
    closed) cfg=(--block-deg 2 --force-gate closed) ;;
    drv)    cfg=(--block-deg 2 --solar-as-driver) ;;
    glob)   cfg=(--block-deg 2 --screen-scope global) ;;
    k5)     cfg=(--block-deg 2); cls=(--order "$ORDER4") ;;
    *) echo "unknown tag: $tag" >&2; return 1 ;;
  esac
  if [[ -f "$out/results.json" && "$FORCE" != 1 ]]; then echo "[skip] $out"; return 0; fi
  mkdir -p "$out"
  echo "[run] $tag seed=$seed -> $out"
  $PY src/mib_study.py --csv "$CSV" "${cls[@]}" "${cfg[@]}" \
      --tag "$tag" --seed "$seed" --out "$out" "$@" 2>&1 | tee "logs/${tag}_seed${seed}.log"
}

check() {
  CSV="$CSV" ORDER3="$ORDER3" MERGE3="$MERGE3" ORDER4="$ORDER4" $PY - <<'EOF'
import contextlib, io, os, sys
import pandas as pd
sys.path.insert(0, "src")
from mib_data import load_dataset
raw = pd.read_csv(os.environ["CSV"])
print(raw.groupby(["type_id", "type_name"]).size().to_string(), "\n")
for name, order, merge, exp in [("K=3", os.environ["ORDER3"], os.environ["MERGE3"], [333, 2711, 23637]),
                                ("K=4", os.environ["ORDER4"], "", None)]:
    with contextlib.redirect_stdout(io.StringIO()):
        ds = load_dataset(os.environ["CSV"], order, merge)
    cnt = pd.Series(ds.y).value_counts().sort_index().tolist()
    tail = "" if exp is None else ("  OK (matches paper)" if cnt == exp else f"  MISMATCH: paper {exp}")
    print(f"{name}: classes={ds.classes} counts={cnt} N={len(ds.y)} "
          f"drivers={ds.drivers} ctx={ds.ctx_candidates}{tail}")
EOF
}

grid() { for t in $GRID; do study "$t" 0 "results/$t" --boot "$BOOT"; done; }

repeats() {   # seeds 1..R-1; seed 0 is the grid itself
  local s t
  for ((s=1; s<R; s++)); do
    study ctx "$s" "results_rep/ctx/seed$s" --boot 0 --boot-sign 0
    for t in b1 b5 b10; do
      study "$t" "$s" "results_rep/$t/seed$s" --boot 0 --boot-sign 0 --protocols spatial
    done
    for t in closed drv k5; do
      study "$t" "$s" "results_rep/$t/seed$s" --boot 0 --boot-sign 0
    done
  done
}

data() { $PY src/mib_data_report.py --csv "$CSV" --order "$ORDER3" --merge "$MERGE3" --out revision; }

vr() {
  local p
  for p in spatial iid; do
    $PY src/mib_vr_control.py --csv "$CSV" --order "$ORDER3" --merge "$MERGE3" \
        --protocol "$p" --out revision | tee "revision/vr_control_$p.txt"
  done
}

revision() { $PY src/mib_revision.py --roots results results_rep --out revision | tee revision/report.txt; }

digest() {
  : > digest_all.txt
  local d
  for d in results/*/; do
    [[ -f "${d}results.json" ]] && $PY src/mib_digest.py --res "${d%/}" >> digest_all.txt
  done
  echo "-> digest_all.txt"
}

figs() {
  $PY src/mib_figures_paper.py --script src/mib_figures.py --res results/ctx \
      --out paper/figs_ctx --with-ctx --ctx-res results/ctx --ctx-out paper/figs_ctx
  rm -rf _raw_results
}

case "${1:-}" in
  check) check ;;
  ctx) study ctx 0 results/ctx --boot "$BOOT" ;;
  grid) grid ;;
  repeats) repeats ;;
  data) data ;;
  vr) vr ;;
  revision) revision ;;
  digest) digest ;;
  figs) figs ;;
  all) check; grid; repeats; data; vr; revision; digest; figs ;;
  *) echo "usage: $0 {check|ctx|grid|repeats|data|vr|revision|digest|figs|all}"; exit 1 ;;
esac