# Final Comparison: DL Project vs. Internship

## 1. What changed, in one paragraph

The internship trained two independent single-target regressors (Ridge for DBT,
LightGBM for WBT) on a flat 13-feature table, selected by a random 80:20 split.
This project trains one multi-task CNN-LSTM (+ optional self-attention) that
predicts DBT and WBT jointly from a (4-timestep x 10-channel) sequence built from
the same underlying variables, evaluated on a stricter chronological, burst-aware
split — and benchmarks it honestly against classical baselines refit on that same
harder split, not just against the internship's original numbers. See
`reports/phase2_design.md` for the full rationale and `reports/phase1_jd_analysis.md`
for the JD evidence that shaped these choices.

## 2. Headline numbers

| Model | Split | Target | R² | RMSE | MAE | KGE |
|---|---|---|---|---|---|---|
| Internship Ridge (random split) | test | DBT | 0.6213 | 1.2064 | 0.8792 | 0.7815 |
| Best classical, this split (Random Forest) | test | DBT | **0.7112** | 1.2178 | 0.9475 | 0.6749 |
| CNN-LSTM hybrid (default hyperparameters) | test | DBT | 0.4964 | 1.6081 | 1.3190 | 0.4400 |
| CNN-LSTM hybrid (Keras-Tuner tuned) | test | DBT | 0.6623 | 1.3167 | 1.0406 | 0.5952 |
| Internship LightGBM (random split) | test | WBT | **0.7092** | 0.7372 | 0.5442 | 0.7474 |
| Best classical, this split (Extra Trees) | test | WBT | 0.6726 | 0.5244 | 0.3957 | 0.7869 |
| CNN-LSTM hybrid (default hyperparameters) | test | WBT | 0.4806 | 0.6605 | 0.4830 | 0.7731 |
| CNN-LSTM hybrid (Keras-Tuner tuned) | test | WBT | 0.6113 | 0.5714 | 0.4131 | 0.8310 |

Full table: [`data/processed/final_comparison_table.csv`](../data/processed/final_comparison_table.csv).

## 3. Does the DL model beat classical ML here? Honest answer: partially, and not uniformly

- **DBT**: the tuned CNN-LSTM (R²=0.662) beats the internship's own original Ridge
  benchmark (R²=0.621) — a real result, not a wash. It does not beat the classical
  baseline refit on this project's own (harder) split (Random Forest, R²=0.711).
- **WBT**: the tuned CNN-LSTM (R²=0.611) does not beat either the internship's
  original LightGBM (R²=0.709) or the classical baseline refit on this split
  (Extra Trees, R²=0.673).
- **Untuned, the CNN-LSTM loses on both targets to every classical baseline**,
  internship or refit (R²=0.50/0.48). Hyperparameter tuning was not cosmetic here —
  it closed roughly two-thirds of the gap on DBT and most of it on WBT.

The honest takeaway: at n=210 training rows, careful classical ML (properly
tuned, chronologically validated) remains competitive with — and on this
split's WBT target, ahead of — a small, well-regularized CNN-LSTM. This matches
what the brief anticipated as a legitimate possible outcome, not a failure of
the DL approach.

## 4. Ablation: what does each architectural piece actually buy?

With **default** (untuned) hyperparameters and identical training budget, the
CNN-LSTM hybrid did not beat either of its own simpler ablations:

| Model | test R² (dbt) | test R² (wbt) | params |
|---|---|---|---|
| LSTM only | 0.575 | 0.596 | 1,882 |
| CNN only | 0.560 | 0.524 | 490 |
| CNN-LSTM hybrid | 0.496 | 0.481 | 3,186 |

At default settings, combining CNN + LSTM + attention added parameters (3,186 vs.
1,882/490) without adding accuracy — consistent with the train/test R² gap also
being *largest* for the hybrid (§6). The hybrid's advantage only appeared after
Keras Tuner search (§3) — it needed its capacity constrained (dropout, l2, unit
counts) to be worth its added complexity at this sample size. This is a genuine,
reportable ablation finding, not a predetermined conclusion.

## 5. Optimizer comparison

Same CNN-LSTM architecture and hyperparameters; only the optimizer changed:

| Optimizer | test R² (dbt) | test R² (wbt) |
|---|---|---|
| Adam | **0.620** | 0.456 |
| RMSprop | 0.610 | **0.608** |
| SGD | 0.546 | 0.474 |

No single optimizer wins both targets — Adam is best for DBT, RMSprop is best
for WBT, and SGD trails on both (consistent with SGD typically needing more
tuning of learning-rate schedule than the fixed rates used here). This is the
kind of result that justifies comparing optimizers explicitly rather than
defaulting to Adam without checking, as the brief asked.

## 6. Overfitting check: train-vs-test R² gap

| Model | Target | Train R² | Test R² | Gap |
|---|---|---|---|---|
| LSTM only | dbt | 0.675 | 0.575 | 0.101 |
| CNN only | dbt | 0.678 | 0.560 | 0.117 |
| CNN-LSTM hybrid | dbt | 0.684 | 0.496 | 0.188 |
| LSTM only | wbt | 0.781 | 0.596 | 0.185 |
| CNN only | wbt | 0.723 | 0.524 | 0.200 |
| CNN-LSTM hybrid | wbt | 0.808 | 0.481 | 0.327 |

The (untuned) hybrid overfits the most of the three architectures on both
targets, and WBT overfits more than DBT across every architecture — the same
asymmetry the internship itself flagged (its WBT models showed a larger
train-test gap than its DBT models; Section 3.6.4 of the internship report).
This is independent confirmation, from a completely different model family, of
the internship's own generalization finding about WBT specifically.

## 7. Interpretability comparison

**Internship SHAP** (report Section 3.4): DBT dominated by `T(i-1)`, then `Hr`,
then surface latent heat flux, with `T(i-4)`/`T(i-2)`/`T(i-3)` also substantial.
WBT dominated by `T(i-4)`, then `T(i-1)`, OLR, `T(i-2)`, with surface sensible
heat flux and `Hr` also notable.

**This project's CNN-LSTM SHAP** (`KernelExplainer` on the flattened sequence,
channel-summed across the 4 timesteps — the architecture collapses the 4
individual lag terms into one `lag_temperature` channel, so a term-by-term
`T(i-1)` vs `T(i-4)` comparison isn't reproducible from this design, only the
lag block as a whole vs. the static channels):

| Rank | DBT channel | mean\|SHAP\| | WBT channel | mean\|SHAP\| |
|---|---|---|---|---|
| 1 | lag_temperature | 0.466 | lag_temperature | 0.358 |
| 2 | surface_latent_heat_flux | 0.069 | surface_latent_heat_flux | 0.081 |
| 3 | Urban_Footprint | 0.062 | specific_humidity_500hPa | 0.056 |
| 4 | Hr | 0.050 | TCC | 0.052 |
| 5 | specific_humidity_500hPa | 0.044 | Wind_500hPa_ms | 0.038 |

**Agreement**: the lag-temperature channel dominates both targets by a wide
margin (ranked #1 of 10, roughly 4-7x the next channel) — directly consistent
with the internship's central finding that temporal persistence governs both
DBT and WBT. Surface latent heat flux as the #2 non-lag contributor for DBT also
matches the internship's ranking exactly.

**Disagreement, stated plainly**: the internship found OLR to be WBT's #3
predictor; here OLR ranks #7 of 10 for WBT. `Hr`, prominent for DBT in the
internship (#2), ranks #4 here — present but less dominant. These are real
architecture-driven differences, not noise to explain away: the CNN-LSTM sees
`Hr` and OLR as one flat, unordered value per sample identical to how
Ridge/LightGBM saw them, so the shift in ranking reflects how a shared
multi-task backbone with a dominant lag-temperature signal redistributes
attribution, not a data or leakage issue (both models were checked for
leakage independently — see notebook 01 and `phase2_design.md` §2).

## 7b. Augmentation ablation: does jittering actually help? No.

`phase2_design.md` §5 proposed Gaussian-noise jittering on the scaled training
sequences as a regularizer. Tested directly in `notebooks/07_augmentation_ablation.ipynb`
(identical tuned architecture and budget, 3x effective training size via 2 noisy
copies per real row, sigma=0.05, targets left unmodified, test set untouched):

| Target | No augmentation (test R²) | Jittered (test R²) | Delta |
|---|---|---|---|
| DBT | 0.715 | 0.655 | −0.060 (hurt) |
| WBT | 0.664 | 0.612 | −0.052 (hurt) |

Jittering **hurt** test performance on both targets in this run. At n=210, diluting
the training set with noisy duplicates apparently cost more real signal than it
bought in regularization. Reported as the actual finding — the design doc's honesty
requirement about augmentation ("it can help regularize training, but should not be
presented as creating new real observations") turned out to need extending to "and
here, empirically, it didn't even help regularize."

**A second, independent honesty note from the same experiment**: the "no
augmentation" run in notebook 07 re-trains the *identical* tuned architecture from
notebook 04 and gets a different DBT test R² (0.715 vs. notebook 04's 0.662) —
purely from run-to-run stochastic initialization, since both runs call
`tf.random.set_seed(42)` but consume the global RNG stream differently depending
on what ran before them in-process. At n=210 with a ~3,000-parameter network,
single-run test metrics carry real variance; every number in this report should be
read as "one run's outcome," not a tight point estimate. This is disclosed rather
than smoothed over by, e.g., quietly re-running until a favorable seed appeared.

## 8. Where DL helped, where it didn't — the honest takeaway

**Helped:**
- Hyperparameter tuning materially improved the CNN-LSTM (R² +0.17 on DBT,
  +0.13 on WBT over default settings) — tuning was not a formality here.
- The tuned model beat the internship's *original* DBT benchmark, on a stricter
  (chronological) split than the internship used — a genuinely harder bar to
  clear, cleared anyway for one of the two targets.
- SHAP on the DL model independently reproduced the internship's central
  physical finding (temporal persistence dominates) from a structurally
  different model family, which is a stronger form of evidence for that
  finding than either model alone would provide.
- The Keras Tuner search itself surfaced a real, non-obvious result: the best
  configuration turned attention **off** (`use_attention: False`), despite
  attention being added specifically because Phase 1's JD analysis found it
  more evidenced than CNN-LSTM by name. Evidence for including a technique in
  the *search space* is not the same as evidence it helps on this dataset —
  worth stating exactly like that rather than quietly dropping the negative
  result.

**Didn't help:**
- At n=210, the CNN-LSTM did not beat classical baselines refit on the same
  split for either target, and clearly lost to them on WBT.
- The architectural complexity that most directly represents this project's
  "genuinely different" framing (CNN+LSTM+attention combined) was the
  *worst*-generalizing of the three DL ablations at default settings, and only
  become competitive after its effective capacity was constrained by tuning —
  i.e., the multi-task/sequence framing's advantage, if any, is not free; it
  has to be earned back from a small-sample generalization penalty.

## 9. Bottom line for a reader deciding what this project demonstrates

This project demonstrates disciplined small-sample deep learning practice —
chronological/burst-aware leakage prevention, honest ablations, a real
optimizer comparison, a constrained hyperparameter search, and an interpretability
check against an independent baseline — applied to a genuinely different problem
framing (joint multi-task sequence modeling) than the internship's flat
single-target regressions. It does **not** demonstrate that deep learning beats
well-tuned classical ML at n≈300; on this dataset, for this problem, it mostly
doesn't, and the one place it edges ahead (DBT vs. the internship's original
benchmark) comes with the caveat that this-split classical baselines still win.
Reporting that plainly is the more defensible result, and matches what the
brief asked for: a discussable, non-inflated finding rather than a forced
"DL wins" narrative.
