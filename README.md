# The Cost of a Physics Prior Is Bounded by the Ablation Gap
![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![repro](https://img.shields.io/badge/reproducible-seed%200-brightgreen.svg)
> **A constrained model must never be beaten by its own ablation.**
A shape constraint on features `S` cannot cost more than `S` is worth: ignoring `S` is always an admissible way to satisfy it, since a function constant in `x_j` is both non-decreasing and non-increasing in `x_j`. Hence `0 <= P(sigma) <= Delta(S)` for any risk functional, with no convexity, smoothness or realizability assumption. It reduces to one sign check on a column you already have: `F1(constrained) >= F1(ablated)`.
The consequence: a reported "cost of physics" has a dynamic range set by the free features and the validation split, not by the physics. Measured here on an ordinal wildfire-severity task (`N=26,681`, `K=3`, four monotone drivers).
## Headline result
Out-of-fold macro-F1, `results/ctx/`. `Delta = MC - GEO_ONLY` is the certified ceiling on the cost of any monotone prior; `P` is the measured price.

| protocol | `GEO_ONLY` | `MC` | `MONO_sc` | `Delta` | `P` | `P/Delta` |
|---|---|---|---|---|---|---|
| `iid` | 0.7028 | 0.8316 | 0.7843 | 0.1288 | 0.0473 | 0.37 |
| `point` | 0.7146 | 0.8311 | 0.7755 | 0.1165 | 0.0555 | 0.48 |
| `spatial` | 0.5602 | 0.6029 | 0.5850 | 0.0427 | 0.0179* | 0.42 |

\* below the self-calibrated floor `s = 0.0220`, so it carries no interpretation.
## Findings

- **Coordinates shield the prior.** `{lat, lon}` alone recover 92.9% of the full model under spatial blocking. The identical screened prior costs `0.0473` shielded and `0.3470` unshielded (`PHYS_sc` vs `WX_ONLY`) - a 7.3x swing from the same physics.
- **Prices do not transfer.** Coarsening blocks 1 deg to 10 deg moves `Delta` from `0.0942` to `0.0050` and the price from `+0.0456` to `-0.0139`: a sign flip driven by the split alone.
- **A free noise floor.** Of 318 orderings the pipeline must satisfy, 18 invert, giving `s = 0.0220`. It voids 4 of 21 price cells and 10 grid cells before any constrained model is trained.
- **Cost and compliance are independent.** The unconstrained model violates the textbook prior at rate `0.48`; enforcing it costs `0.02`.

## Screen before you run
```python
def admissible(fit, X, y, S, protocol, s_hat):
    """Two unconstrained fits. No constrained solver required."""
    delta = fit(X, y, protocol) - fit(X.drop(columns=S), y, protocol)
    if delta < s_hat:
        return f"unidentifiable: Delta={delta:.4f} < s={s_hat}"
    return f"proceed; any price on S lies in [0, {delta:.4f}]"
```
## Quickstart
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./run.sh ctx      # headline config: 2 deg blocks, gate auto/fold
./run.sh grid     # all nine configurations
./run.sh digest   # results/*/ -> digest_all.txt
```
## Repo map

| path | contents |
|---|---|
| `src/mib_study.py` | entrypoint: 11 arms x 3 protocols x 5 folds, seed 0 |
| `src/mib_models.py` | cumulative ordinal HGB ensemble, hard monotone split constraints |
| `src/mib_data.py` | loading, filtering, spatial blocking |
| `src/mib_audit.py` | falsifiers C1-C6, violation rates, exogeneity gate |
| `src/mib_digest.py` | results to manuscript tables |
| `src/mib_figures_paper.py` | figures written to `paper/figs_ctx/` |
| `results/<tag>/`, `logs/` | per-configuration artefacts and full run logs |

Arms: `MAJORITY`, `GEO_ONLY`, `WX_ONLY`, `MC`, `MONO_naive`, `MONO_sc`, `PHYS`, `PHYS_sc`, `MIB{,_fp,_fm}`. Drivers `D = {temp, rh, wind, precip}`, shield `G = {lat, lon}`, auxiliary channel is residualized insolation. All arms share one architecture (300 iterations, lr 0.06, 31 leaves, min 40 per leaf, L2 1.0, seed 0) and differ only in feature set and monotonicity spec, so class nesting holds exactly and no penalty weight is tuned.
Grid: `ctx` (headline, 2 deg), `b1`/`b2`/`b5`/`b10` (block sweep), `closed` (gate shut), `drv` (`|S|=5`), `glob` (leak probe), `k5` (`K=4`). `ctx`, `b2` and `glob` coincide bytewise.
## Scope
Not a new algorithm, and not a claim that shape constraints are useless - they buy extrapolation and auditability, which no in-distribution metric measures. Note the 92.9% shield is latitude and longitude, which carry no mechanism and cannot be evaluated outside the sampled domain. The claim is narrower: **in-distribution accuracy cost is the wrong instrument for evaluating a physical prior.** Report the triple `(price, ceiling, slack)` with the free feature set.
## Citation
```bibtex
@article{kriuk2026ablationgap,
  title  = {The Cost of a Physics Prior Is Bounded by the Ablation Gap},
  author = {Kriuk, Boris},
  year   = {2026}
}
```
Corpus and meteorological joins: `arXiv:2502.17023`. Licensed Apache-2.0.
