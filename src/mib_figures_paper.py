#!/usr/bin/env python3
# =====================================================================
#  mib_figures_paper.py
#
#  Builds ONLY the 5 figures the paper actually uses, in English,
#  publication-styled, straight into figs/ with the exact filenames
#  the .tex expects. mib_figures.py is NOT modified.
#
#  Output
#  ------
#    figs/fig1_prior_violation.png       -> \label{fig:prior}
#    figs/fig2_cost_by_protocol.png      -> \label{fig:cost}
#    figs/fig4_certificate_leakage.png   -> \label{fig:leak}
#    figs/fig5_sign_identification.png   -> \label{fig:signs}
#    figs/fig6_protocol_ladder.png       -> \label{fig:ladder}
#    figs/tables.tex
#    figs/PAPER_FIGURES.pdf              (all 5, one per page, to review)
#
#  Usage
#  -----
#    python mib_figures_paper.py --strict
#    python mib_figures_paper.py --res results_b10
#    python mib_figures_paper.py --with-ctx       # + figs_ctx/fig3,fig7
#    python mib_figures_paper.py --style sans
#    python mib_figures_paper.py --keep-raw
# =====================================================================
from __future__ import annotations

import argparse
import re
import runpy
import shutil
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages     # noqa: E402
from matplotlib.text import Text                         # noqa: E402

CYR = re.compile(r"[А-Яа-яЁё]")

# ---------------------------------------------------------------------
# the 5 figures the paper cites, in paper order
# ---------------------------------------------------------------------
PAPER_FIGS: list[tuple[str, str, str]] = [
    ("fig6_protocol_ladder.png",     "fig:ladder",
     "headroom collapse across the validation ladder"),
    ("fig2_cost_by_protocol.png",    "fig:cost",
     "price vs certified ceiling; ablation-dominance test"),
    ("fig5_sign_identification.png", "fig:signs",
     "per-fold driver sign identification"),
    ("fig1_prior_violation.png",     "fig:prior",
     "violation rate of the unconstrained model"),
    ("fig4_certificate_leakage.png", "fig:leak",
     "certificate vs deployed pipeline"),
]

# optional appendix pair (only if --with-ctx)
CTX_FIGS: list[tuple[str, str, str]] = [
    ("fig3_exogeneity_screen.png",    "fig:exo",
     "exogeneity screen for auxiliary channels"),
    ("fig7_ctx_sign_sensitivity.png", "fig:ctx",
     "auxiliary-channel sign sensitivity"),
]

# =====================================================================
#  TRANSLATION -- substring rules, specific before generic.
# =====================================================================
SUBS: list[tuple[str, str]] = [
    # protocol names
    (r"простр\.-врем\.\s*блоки",            "spatiotemporal blocks"),
    (r"пространственно-временные\s*блоки",  "spatiotemporal blocks"),
    (r"простр\.\s*блоки",                   "spatial blocks"),
    (r"пространственные\s*блоки",           "spatial blocks"),
    (r"врем\.\s*блоки",                     "temporal blocks"),
    (r"временные\s*блоки",                  "temporal blocks"),
    (r"по\s+точкам",                        "by point"),
    # gate state
    (r"шлюз\s+закрыт",                      "gate closed"),
    (r"шлюз\s+открыт",                      "gate open"),
    (r"порог\s+шлюза",                      "gate threshold"),
    (r"\bзакрыт\b",                         "closed"),
    (r"\bоткрыт\b",                         "open"),
    (r"контекст-кандидатов\s+нет",          "no context candidates"),
    (r"—\s*пропуск",                        "— skipped"),
    # fig1
    (r"Аудит\s+приора\s+монотонности,\s*протокол\s*",
     "Monotone-prior audit — protocol: "),
    (r"общий\s+VR:\s*наивный",              "overall VR: naive"),
    (r"скрининг\s*=",                       "screened="),
    (r"violation\s+rate\s*\(частный\s+эффект\)",
     "violation rate (partial effect)"),
    (r"наивный\s*\(учебниковый\)\s*приор",  "naive (textbook) prior"),
    (r"скринингованный\s+приор",            "screened prior"),
    (r"случайный\s+знак",                   "random sign"),
    # fig2
    (r"Парные\s+разности\s+macro-F1,\s*95%\s*кластерный\s+бутстрап",
     "Paired macro-F1 differences, 95% cluster bootstrap"),
    (r"цена\s+ограничений\s*=\s*",          "constraint cost = "),
    (r"\(одинаковые\s+признаки\)",          "(feature-matched)"),
    (r"относительно",                       "vs"),
    # fig3
    (r"Скрининг\s+экзогенности:\s*красное\s*=\s*эндогенно,?",
     "Exogeneity screen — red = endogenous,"),
    (r"сертификат\s*«частичной\s+монотонности»\s*недействителен",
     "\u201cpartial-monotonicity\u201d certificate is void"),
    (r"регрессии\s+на\s+квадратичный\s+дизайн\s+драйверов",
     "of quadratic regression on the drivers"),
    # fig4
    (r"Сертификат\s+vs\s+развёрнутый\s+пайплайн",
     "Certificate vs deployed pipeline"),
    (r"собственная\s+декларация\s*\(сертификат\)",
     "model\u2019s own declaration (certificate)"),
    (r"против\s+наивного\s+приора\s*\(сопоставимо\s+для\s+всех\)",
     "against the naive prior (comparable across arms)"),
    # fig5
    (r"Идентификация\s+знака\s+драйвера\s+по\s+фолдам",
     "Per-fold identification of driver signs"),
    (r"o\s*=\s*95%\s*CI\s*исключает\s*0,\s*x\s*=\s*знак\s+не\s+идентифицирован",
     "o = 95% CI excludes 0     x = sign not identified"),
    (r"знак\s+не\s+идентифицирован",        "sign not identified"),
    (r"средний\s+частный\s+эффект\s+на\s*E\[y\]",
     "mean partial effect on E[y]"),
    # fig6
    (r"Обобщение\s+падает\s+по\s+мере\s+устранения\s+утечки\s+в\s+разбиении",
     "Generalization degrades as split leakage is removed"),
    (r"macro-F1\s*\(OOF-пул\)",             "macro-F1 (pooled OOF)"),
    (r"утечка\s+разбиения",                 "split leakage"),
    # fig7
    (r"Вклад\s+контекстного\s+канала\s+зависит\s+от\s+выбора\s+его\s+знака",
     "Auxiliary-channel value depends on the sign imposed on it"),
    (r"без\s+ограничения",                  "unconstrained"),
    # tables.tex captions
    (r"Macro-F1\s+по\s+протоколам\s+валидации\s+с\s+парными\s+"
     r"кластерными\s+95\\%\s*CI",
     "Macro-F1 across validation protocols with paired cluster 95\\% CIs"),
    (r"цена\s+ограничений\s+при\s+полном\s+наборе\s+признаков",
     "cost of the constraints under the full feature set"),
    (r"цена\s+ограничений\s+для\s+моделей\s+без\s+географии",
     "cost of the constraints for models without geography"),
    (r"прирост\s+над\s+моделью\s+по\s+координатам",
     "gain over the coordinates-only model"),
    (r"означает\s+побайтовое\s+совпадение\s+предсказаний",
     "denotes bytewise-identical predictions"),
    (r"нарушения\s+собственной\s+декларации\s+знаков",
     "violations of the model\u2019s own declared signs"),
    (r"знаки\s+не\s+заявлены",              "no signs declared"),
    (r"Идентификация\s+знака\s+драйвера\.", "Driver sign identification."),
    (r"Скрининг\s+экзогенности\s+контекстных\s+признаков\.",
     "Exogeneity screening of the context features."),
    # generic leftovers
    (r"нарушения\s+собственной\s+декларации", "own-declaration violations"),
    (r"наивный\s+приор",                    "naive prior"),
    (r"по\s+фолдам",                        "by fold"),
]

WORDS: dict[str, str] = {
    "число": "number of", "где": "where", "нуль": "zero",
    "аудит": "audit", "приора": "prior", "приор": "prior",
    "монотонности": "monotonicity", "протокол": "protocol",
    "протоколам": "protocols", "наивный": "naive", "наивного": "naive",
    "учебниковый": "textbook", "скринингованный": "screened",
    "скрининг": "screening", "экзогенности": "exogeneity",
    "общий": "overall", "средний": "mean", "среднего": "mean",
    "частный": "partial", "частного": "partial", "частичной": "partial",
    "эффект": "effect", "эффекта": "effect", "фолд": "fold",
    "фолдам": "folds", "фолдов": "folds", "знак": "sign", "знака": "sign",
    "знаки": "signs", "знаков": "signs", "драйвер": "driver",
    "драйвера": "driver", "драйверов": "drivers", "модель": "model",
    "модели": "models", "моделей": "models", "цена": "cost",
    "ограничений": "constraints", "ограничения": "constraints",
    "признаки": "features", "признаков": "features",
    "разности": "differences", "парные": "paired", "парными": "paired",
    "кластерный": "cluster", "кластерными": "cluster",
    "бутстрап": "bootstrap", "исключает": "excludes",
    "идентифицирован": "identified", "идентификация": "identification",
    "красное": "red", "эндогенно": "endogenous",
    "сертификат": "certificate", "развёрнутый": "deployed",
    "пайплайн": "pipeline", "собственная": "own", "собственной": "own",
    "декларация": "declaration", "декларации": "declaration",
    "против": "against", "сопоставимо": "comparable",
    "вклад": "contribution", "контекстного": "auxiliary",
    "контекстных": "context", "контекст": "context", "канала": "channel",
    "зависит": "depends", "выбора": "choice", "падает": "degrades",
    "устранения": "removing", "разбиении": "the split",
    "разбиения": "split", "обобщение": "generalization",
    "утечка": "leakage", "утечки": "leakage", "шлюз": "gate",
    "шлюза": "gate", "порог": "threshold", "блоки": "blocks",
    "блок": "block", "точкам": "point", "случайный": "random",
    "квадратичный": "quadratic", "дизайн": "design",
    "регрессии": "regression", "недействителен": "void",
    "нарушения": "violations", "потолок": "ceiling",
    "валидации": "validation", "географии": "geography",
    "координатам": "coordinates", "прирост": "gain",
    "означает": "denotes", "побайтовое": "bytewise",
    "совпадение": "match", "предсказаний": "predictions",
    "пропуск": "skipped", "нет": "none", "да": "yes", "без": "without",
    "все": "all", "всех": "all", "его": "its", "мере": "as",
    "для": "for", "от": "of", "на": "on", "по": "by", "и": "and",
    "в": "in", "с": "with", "у": "for",
}

_SUBS_C = [(re.compile(p, re.IGNORECASE), r) for p, r in SUBS]
_WORDS_C = [
    (re.compile(rf"(?<![A-Za-zА-Яа-яЁё]){re.escape(w)}"
                rf"(?![A-Za-zА-Яа-яЁё])", re.IGNORECASE), r)
    for w, r in sorted(WORDS.items(), key=lambda kv: -len(kv[0]))
]

LEFTOVERS: set[str] = set()


def translate(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    if "\\_" in s and "$" not in s:
        s = s.replace("\\_", "_")
    if not CYR.search(s):
        return s
    out = s
    for rx, rep in _SUBS_C:
        out = rx.sub(rep, out)
    if CYR.search(out):
        for rx, rep in _WORDS_C:
            out = rx.sub(rep, out)
    out = re.sub(r"[ \t]{2,}", "  ", out)
    if CYR.search(out):
        LEFTOVERS.add(out.replace("\n", " / ").strip())
    return out


# =====================================================================
#  shims
# =====================================================================
def install_text_shim() -> None:
    orig_init, orig_set = Text.__init__, Text.set_text

    def init(self, x=0, y=0, text="", *a, **kw):
        if "text" in kw:
            kw["text"] = translate(kw["text"])
            return orig_init(self, x, y, *a, **kw)
        return orig_init(self, x, y, translate(text), *a, **kw)

    def set_text(self, s):
        return orig_set(self, translate(s) if isinstance(s, str) else s)

    Text.__init__, Text.set_text = init, set_text


class Quiet:
    """Swallow the child script's DataFrame dumps; keep [tag] lines."""

    def __init__(self, real, verbose):
        self.real, self.verbose, self.buf = real, verbose, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if self.verbose or line.lstrip().startswith(("[", "==")):
                self.real.write(translate(line) + "\n")

    def flush(self):
        self.buf = ""
        self.real.flush()

    def __getattr__(self, n):
        return getattr(self.real, n)


def install_style(style: str, dpi: int) -> None:
    base = {
        "figure.dpi": 120, "savefig.dpi": dpi, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9.5, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "axes.titlepad": 9, "axes.grid": True,
        "grid.alpha": 0.30, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9, "lines.linewidth": 1.8,
        "lines.markersize": 5.5, "legend.frameon": False,
        "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if style == "serif":
        base |= {"font.family": "serif",
                 "font.serif": ["Times New Roman", "Nimbus Roman",
                                "DejaVu Serif"],
                 "mathtext.fontset": "stix"}
    else:
        base |= {"font.family": "sans-serif",
                 "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                 "mathtext.fontset": "dejavusans"}
    plt.rcParams.update(base)


# =====================================================================
#  build helpers
# =====================================================================
def build(script: str, res: str, staging: Path, verbose: bool) -> bool:
    staging.mkdir(parents=True, exist_ok=True)
    sys.argv = [script, "--res", res, "--out", str(staging)]
    real = sys.stdout
    sys.stdout = Quiet(real, verbose)
    try:
        runpy.run_path(script, run_name="__main__")
        return True
    except SystemExit:
        return True
    except Exception:
        sys.stdout = real
        traceback.print_exc(limit=3)
        return False
    finally:
        sys.stdout = real
        plt.close("all")


def harvest(staging: Path, dest: Path,
            wanted: list[tuple[str, str, str]]) -> list[tuple[str, Path]]:
    dest.mkdir(parents=True, exist_ok=True)
    kept = []
    for name, label, _ in wanted:
        src = staging / name
        if src.exists():
            shutil.copy2(src, dest / name)
            kept.append((label, dest / name))
        else:
            print(f"  !! missing {src} (this run does not produce it)")
    tex = staging / "tables.tex"
    if tex.exists():
        (dest / "tables.tex").write_text(
            translate(tex.read_text(encoding="utf-8")), encoding="utf-8")
    return kept


def review_pdf(items: list[tuple[str, Path, str]], dest: Path) -> None:
    if not items:
        return
    with PdfPages(dest) as pdf:
        for label, path, role in items:
            try:
                img = plt.imread(path)
            except Exception:
                continue
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(min(w / 170, 10.5),
                                      min(h / 170, 13.5) + 0.5))
            ax = fig.add_axes([0, 0, 1, 0.94])
            ax.imshow(img)
            ax.set_axis_off()
            fig.text(0.5, 0.975, f"{path.name}   \\label{{{label}}}",
                     ha="center", fontsize=10)
            fig.text(0.5, 0.951, role, ha="center", fontsize=8.5,
                     color="0.35")
            pdf.savefig(fig)
            plt.close(fig)


# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build only the 5 figures the paper cites.")
    ap.add_argument("--script", default="mib_figures.py")
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--with-ctx", action="store_true")
    ap.add_argument("--ctx-res", default="results_ctx")
    ap.add_argument("--ctx-out", default="figs_ctx")
    ap.add_argument("--style", choices=["serif", "sans"], default="serif")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    if not Path(args.script).exists():
        sys.exit(f"error: {args.script} not found -- cd to the project root")
    if not (Path(args.res) / "results.json").exists():
        sys.exit(f"error: {args.res}/results.json not found")

    install_style(args.style, args.dpi)
    install_text_shim()

    review: list[tuple[str, Path, str]] = []

    # ---- primary run -> figs/ ---------------------------------------
    raw = Path("_raw_" + args.res)
    print(f"[paper] building from {args.res}/ ...")
    if not build(args.script, args.res, raw, args.verbose):
        return 1
    kept = harvest(raw, Path(args.out), PAPER_FIGS)
    roles = {n: r for n, _, r in PAPER_FIGS}
    review += [(lbl, p, roles[p.name]) for lbl, p in kept]
    if not args.keep_raw:
        shutil.rmtree(raw, ignore_errors=True)

    # ---- optional ctx pair -> figs_ctx/ -----------------------------
    if args.with_ctx:
        if (Path(args.ctx_res) / "results.json").exists():
            raw_c = Path("_raw_" + args.ctx_res)
            print(f"[paper] building appendix pair from {args.ctx_res}/ ...")
            if build(args.script, args.ctx_res, raw_c, args.verbose):
                kc = harvest(raw_c, Path(args.ctx_out), CTX_FIGS)
                rc = {n: r for n, _, r in CTX_FIGS}
                review += [(lbl, p, rc[p.name]) for lbl, p in kc]
            if not args.keep_raw:
                shutil.rmtree(raw_c, ignore_errors=True)
        else:
            print(f"  !! {args.ctx_res}/results.json not found -- skipped")

    review_pdf(review, Path(args.out) / "PAPER_FIGURES.pdf")
    plt.close("all")

    # ---- report ------------------------------------------------------
    print("\n" + "=" * 64)
    print("DONE -- upload these, nothing else")
    print("=" * 64)
    for lbl, p, role in review:
        print(f"  {str(p):<44} \\ref{{{lbl}}}")
        print(f"      {role}")
    print(f"\n  {args.out}/tables.tex           numbers for Table II")
    print(f"  {args.out}/PAPER_FIGURES.pdf   flip through to sanity-check")
    print(f"\n  source run: {args.res}/   style: {args.style}   "
          f"dpi: {args.dpi}")
    if not args.with_ctx:
        print("\n  Note: fig3 and fig7 were NOT built. Delete the fig:exo\n"
              "  and fig:ctx figure blocks from the .tex, or re-run\n"
              "  with --with-ctx to keep them.")

    if LEFTOVERS:
        print("\n[paper] WARNING -- untranslated Cyrillic remains:",
              file=sys.stderr)
        for l in sorted(LEFTOVERS):
            print("   " + l, file=sys.stderr)
        if args.strict:
            return 2
    else:
        print("\n[paper] No Cyrillic remaining.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
