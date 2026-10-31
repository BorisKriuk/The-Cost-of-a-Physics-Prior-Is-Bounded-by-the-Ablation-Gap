"""Фигуры и LaTeX-таблицы из results/. Запускать после mib_study.py."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 9,
                     "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
MODELS = ["MAJORITY", "GEO_ONLY", "WX_ONLY", "MC", "MONO_naive", "MONO_sc",
          "PHYS", "PHYS_sc", "MIB", "MIB_fp", "MIB_fm"]
PLAB = {"iid": "iid", "point": "по точкам", "spatial": "простр. блоки",
        "temporal": "врем. блоки", "spatiotemporal": "простр.-врем."}


def _p0(R): return list(R["protocols"])[0]
def _present(R, key, names=MODELS): return [m for m in names if m in R["protocols"][_p0(R)][key]]
def _pick(R, *prefer):
    for p in prefer:
        if p in R["protocols"]: return p
    return _p0(R)


def fig1_prior(R, out):
    p = _pick(R, "spatiotemporal", "spatial")
    pr = R["protocols"][p]["vr_partial"]; drv = R["drivers"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(len(drv)); w = .38
    for k, (tag, lab) in enumerate((("naive", "наивный (учебниковый) приор"),
                                    ("screened", "скринингованный приор"))):
        v = [pr[tag]["per_driver"].get(d, np.nan) for d in drv]
        ax.bar(x + (k - .5) * w, np.nan_to_num(v), w, label=lab)
    ax.axhline(.5, ls="--", lw=1, c="k")
    ax.text(len(drv) - .5, .52, "случайный знак", ha="right", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(drv); ax.set_ylim(0, 1)
    ax.set_ylabel("violation rate (частный эффект)")
    ax.set_title(f"Аудит приора монотонности, протокол {PLAB.get(p,p)}\nобщий VR: "
                 f"наивный={pr['naive']['overall']:.3f}, скрининг={pr['screened']['overall']:.3f}")
    ax.legend(fontsize=7, frameon=False); fig.tight_layout()
    fig.savefig(f"{out}/fig1_prior_violation.png"); plt.close(fig)


def _cost_panel(ax, R, ref, names):
    protos = list(R["protocols"]); y = np.arange(len(names))
    for k, p in enumerate(protos):
        c = R["protocols"][p]["cost"].get(ref, {})
        nm = [m for m in names if m in c]
        if not nm: continue
        yy = np.array([names.index(m) for m in nm], float)
        off = (k - (len(protos) - 1) / 2) * (.8 / max(len(protos), 1))
        d  = [c[m]["delta"] for m in nm]
        lo = [max(c[m]["delta"] - c[m]["ci_lo"], 0) for m in nm]
        hi = [max(c[m]["ci_hi"] - c[m]["delta"], 0) for m in nm]
        ax.errorbar(d, yy + off, xerr=[lo, hi], fmt="o", ms=4, capsize=2, lw=1,
                    label=PLAB.get(p, p))
    ax.axvline(0, c="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel(rf"$\Delta$ macro-F1 относительно {ref}")


def fig2_cost(R, out):
    names = _present(R, "f1")
    refs = [r for r in R.get("table_refs", ["MC"]) if r in R["protocols"][_p0(R)]["cost"]]
    if not refs:
        print("[fig2] в results.json нет cost по референсам — пропуск"); return
    fig, axes = plt.subplots(1, len(refs), figsize=(4.4 * len(refs), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, refs):
        _cost_panel(ax, R, r, [m for m in names if m != r])
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Парные разности macro-F1, 95% кластерный бутстрап\n"
                 "цена ограничений = PHYS* vs WX_ONLY и MONO* vs MC (одинаковые признаки)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(f"{out}/fig2_cost_by_protocol.png"); plt.close(fig)


def fig3_exogeneity(R, out):
    s = pd.DataFrame(R["screen"])
    if not len(s):
        print("[fig3] контекст-кандидатов нет — пропуск"); return
    s = s.sort_values("r2_on_drivers")
    fig, ax = plt.subplots(figsize=(5.2, .45 * len(s) + 1.8))
    ax.barh(s.feature, s.r2_on_drivers,
            color=["tab:green" if e else "tab:red" for e in s.exogenous])
    ax.axvline(R["config"]["max_r2"], ls="--", c="k", lw=1)
    ax.text(R["config"]["max_r2"], -.6, "порог шлюза", fontsize=7, ha="center")
    for i, (f, r) in enumerate(zip(s.feature, s.r2_on_drivers)):
        ax.text(min(r + .01, .99), i, f"{r:.4f}", va="center", fontsize=6)
    ax.set_xlim(0, 1.05); ax.set_xlabel(r"$R^2$ регрессии на квадратичный дизайн драйверов")
    ax.set_title("Скрининг экзогенности: красное = эндогенно,\n"
                 "сертификат «частичной монотонности» недействителен")
    fig.tight_layout(); fig.savefig(f"{out}/fig3_exogeneity_screen.png"); plt.close(fig)

ZTOL = 1e-9   # module-level, next to MODELS


def fig4_leakage(R, out):
    from matplotlib.patches import Patch
    protos = list(R["protocols"])
    names  = _present(R, "vr")
    tags   = (("own",   "собственная декларация (сертификат)"),
              ("naive", "против наивного приора (сопоставимо для всех)"))

    def val(p, m, t):
        d = R["protocols"][p]["vr"].get(m, {}).get(t)
        return float(d["overall"]) if d else np.nan

    vals = [val(p, m, t) for p in protos for m in names for t, _ in tags]
    top  = max([v for v in vals if np.isfinite(v)] + [0.05]) * 1.18
    stub = 0.014 * top          # visible height for an exact zero
    labY = stub + 0.012 * top

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), sharey=True)
    w = .8 / len(protos)
    for ax, (tag, ttl) in zip(axes, tags):
        x = np.arange(len(names), dtype=float)
        for k, p in enumerate(protos):
            h    = np.array([val(p, m, tag) for m in names])
            off  = (k - (len(protos) - 1) / 2) * w
            fin  = np.isfinite(h)
            zero = fin & (h <= ZTOL)
            pos  = fin & ~zero
            ax.bar(x[pos] + off, h[pos], w, color=f"C{k}", zorder=3)
            ax.bar(x[zero] + off, np.full(int(zero.sum()), stub), w,
                   color=f"C{k}", alpha=.30, edgecolor=f"C{k}", lw=.7,
                   hatch="////", zorder=3)
        for i, m in enumerate(names):
            v = np.array([val(p, m, tag) for p in protos])
            if np.all(np.isnan(v)):
                ax.text(i, labY, "n/a", ha="center", va="bottom",
                        fontsize=7, color=".45")
            elif np.nanmax(v) <= ZTOL:
                ax.text(i, labY, "0", ha="center", va="bottom",
                        fontsize=7.5, color=".25")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_title(ttl, fontsize=8)
        ax.set_axisbelow(True)

    axes[0].set_ylim(0, top)
    axes[0].set_ylabel("VR$_{e2e}$")
    hs = [Patch(facecolor=f"C{k}", label=PLAB.get(p, p))
          for k, p in enumerate(protos)]
    hs += [Patch(facecolor="0.6", alpha=.30, hatch="////", edgecolor="0.4",
                 label="VR = 0 exactly"),
           Patch(facecolor="none", edgecolor="none",
                 label="n/a = no signs declared")]
    axes[0].legend(handles=hs, fontsize=7, frameon=False, ncol=2, loc="upper left")
    fig.suptitle(f"Сертификат vs развёрнутый пайплайн   "
                 f"ctx_keep={R['ctx_keep'] or '[] (шлюз закрыт)'}", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{out}/fig4_certificate_leakage.png"); plt.close(fig)


def fig5_signs(R, out):
    s = pd.DataFrame(R["sign_stability"])
    if not len(s):
        print("[fig5] нет данных о знаках — пропуск"); return
    drv = R["drivers"]; protos = list(R["protocols"])
    cmap = {p: f"C{i}" for i, p in enumerate(protos)}
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, j in enumerate(drv):
        for _, r in s[s.driver == j].iterrows():
            dx = (protos.index(r.protocol) - (len(protos) - 1) / 2) * .13 + (r.fold - 3) * .02
            ax.errorbar(i + dx, r["mean"],
                        yerr=[[max(r["mean"] - r.ci_lo, 0)], [max(r.ci_hi - r["mean"], 0)]],
                        fmt="o" if r.identified else "x", ms=4, capsize=2,
                        color=cmap[r.protocol], lw=1)
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(range(len(drv))); ax.set_xticklabels(drv)
    ax.set_ylabel("средний частный эффект на E[y]")
    ax.set_title("Идентификация знака драйвера по фолдам\n"
                 "o = 95% CI исключает 0, x = знак не идентифицирован")
    ax.legend(handles=[plt.Line2D([], [], color=c, marker="o", ls="", label=PLAB.get(p, p))
                       for p, c in cmap.items()], fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig5_sign_identification.png"); plt.close(fig)


def fig6_protocol_ladder(R, out):
    protos = list(R["protocols"])
    names = [n for n in ["GEO_ONLY", "WX_ONLY", "MC", "MONO_sc", "MIB"]
             if n in R["protocols"][protos[0]]["f1"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(protos))
    for n in names:
        ax.plot(x, [R["protocols"][p]["f1"][n] for p in protos], "o-", ms=4, label=n)
    ax.plot(x, [R["protocols"][p]["f1"]["MAJORITY"] for p in protos], "k--", lw=1, label="MAJORITY")
    ax.set_xticks(x); ax.set_xticklabels([PLAB.get(p, p) for p in protos], rotation=15)
    ax.set_ylabel("macro-F1 (OOF-пул)")
    ax.set_title("Обобщение падает по мере устранения утечки в разбиении")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(f"{out}/fig6_protocol_ladder.png"); plt.close(fig)


def fig7_ctx_sign(R, out):
    """Чувствительность MIB к произвольно выбранному знаку контекста."""
    ref = "MONO_sc"; protos = list(R["protocols"])
    names = [m for m in ("MIB", "MIB_fp", "MIB_fm") if m in R["protocols"][protos[0]]["f1"]]
    if not names or ref not in R["protocols"][protos[0]]["cost"]:
        print("[fig7] нет cost vs MONO_sc — пропуск"); return
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    y = np.arange(len(names))
    for k, p in enumerate(protos):
        c = R["protocols"][p]["cost"][ref]
        off = (k - (len(protos) - 1) / 2) * (.8 / len(protos))
        d  = [c[m]["delta"] for m in names]
        lo = [max(c[m]["delta"] - c[m]["ci_lo"], 0) for m in names]
        hi = [max(c[m]["ci_hi"] - c[m]["delta"], 0) for m in names]
        ax.errorbar(d, y + off, xerr=[lo, hi], fmt="o", ms=4, capsize=2, lw=1,
                    label=PLAB.get(p, p))
    ax.axvline(0, c="k", lw=1); ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ macro-F1 относительно MONO\_sc")
    ax.set_title("Вклад контекстного канала зависит от выбора его знака\n"
                 "(fp: +1, fm: -1, MIB: без ограничения)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig7_ctx_sign_sensitivity.png"); plt.close(fig)


def tables(R, res_dir, out):
    t = pd.read_csv(f"{res_dir}/table_main.csv")
    refs = [r for r in R.get("table_refs", ["MC"]) if f"dF1_vs_{r}" in t.columns]
    fmt_nan = lambda x: "n/a" if pd.isna(x) else f"{x:.4f}"
    t["F1 (OOF)"] = t.f1_oof.map("{:.4f}".format)
    t[r"F1 fold $\pm$ sd"] = t.apply(
        lambda r: f"{r.f1_fold_mean:.4f} $\\pm$ {r.f1_fold_sd:.4f}", axis=1)
    t["alias_of"] = t.alias_of.fillna("")
    cols = ["protocol", "model", "alias_of", "F1 (OOF)", r"F1 fold $\pm$ sd"]
    for r_ in refs:
        c = rf"$\Delta$ vs {r_.replace('_', chr(92)+'_')} [95\% CI]"
        t[c] = t.apply(lambda z, r_=r_: "—" if z.model == r_ else
                       (rf"$\equiv$" if bool(z.get(f"ident_{r_}", False)) else
                        f"{z[f'dF1_vs_{r_}']:+.4f} "
                        f"[{z[f'ci_lo_{r_}']:+.4f}, {z[f'ci_hi_{r_}']:+.4f}]"), axis=1)
        cols.append(c)
    t[r"VR$_{own}$"] = t.vr_own.map(fmt_nan)
    t[r"VR$_{naive}$"] = t.vr_vs_naive.map(fmt_nan)
    cols += [r"VR$_{own}$", r"VR$_{naive}$"]
    tex = t[cols].to_latex(
        index=False, escape=False,
        caption=("Macro-F1 по протоколам валидации с парными кластерными 95\\% CI. "
                 r"$\Delta$ vs MC --- цена ограничений при полном наборе признаков; "
                 r"$\Delta$ vs WX\_ONLY --- цена ограничений для моделей без географии; "
                 r"$\Delta$ vs GEO\_ONLY --- прирост над моделью по координатам. "
                 r"$\equiv$ означает побайтовое совпадение предсказаний. "
                 r"VR$_{own}$ --- нарушения собственной декларации знаков (n/a = знаки "
                 r"не заявлены)."),
        label="tab:main")

    s = pd.DataFrame(R["sign_stability"])
    if len(s):
        g = s.groupby("driver").agg(folds=("fold", "size"),
                                    agree_naive=("agrees_with_naive", "sum"),
                                    identified=("identified", "sum"),
                                    n_plus=("sign_screened", lambda v: int((v > 0).sum())),
                                    mean_effect=("mean", "mean")).reset_index()
        tex += "\n\n" + g.to_latex(index=False, escape=False, float_format="%.4g",
                                   caption=("Идентификация знака драйвера. identified --- число "
                                            "фолдов, где 95\\% кластерный CI среднего частного "
                                            "эффекта исключает нуль."),
                                   label="tab:signs")
    scr = pd.DataFrame(R["screen"])
    tex += "\n\n" + (scr.to_latex(index=False, escape=False, float_format="%.6f",
                     caption="Скрининг экзогенности контекстных признаков.", label="tab:screen")
                     if len(scr) else "% контекст-кандидатов нет\n")
    open(f"{out}/tables.tex", "w").write(tex)
    print(t[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results"); ap.add_argument("--out", default="figs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = json.load(open(f"{a.res}/results.json"))
    assert "refs" in R, (f"{a.res}/results.json создан СТАРЫМ mib_study.py "
                         "(нет ключа 'refs'). Перезапусти прогон новым скриптом.")
    fig1_prior(R, a.out); fig2_cost(R, a.out); fig3_exogeneity(R, a.out)
    fig4_leakage(R, a.out); fig5_signs(R, a.out); fig6_protocol_ladder(R, a.out)
    fig7_ctx_sign(R, a.out); tables(R, a.res, a.out)
    print(f"\n[figures] -> {a.out}/fig1..fig7 + tables.tex")


if __name__ == "__main__":
    main()