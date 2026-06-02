# Agent Team Workflow

This project uses a four-agent loop to improve the SIGNATE NIR wood moisture
score while keeping submission cost under control.

## Roles

| Role | Model | Purpose | Writes code? |
| --- | --- | --- | --- |
| Research Agent | `gpt-5.5`, medium | Find similar NIR / wood moisture / chemometrics work and convert it into experiment ideas. | No |
| Plan Agent | `gpt-5.5`, medium | Interpret local results, research, and Public-history constraints; choose the next experiment. | No |
| Executor Agent | `gpt-5.3-codex`, medium | Implement one narrow experiment and run local checks. | Yes |
| Review Agent | `gpt-5.3-codex`, medium | Check whether implementation matches intent and whether metrics/submission format are valid. | No, unless explicitly asked |

## Current Constraints

- Historical tracked Public best in this repo was `candidate_linear_1f` with
  Public `17.496`, but the user reported a stronger external/older submission:
  `candidate_band616_pls_c2_smooth5.csv` with Public
  `16.151771224841994` (`V5 wave1: band616 PLS c2`, submitted
  2026-05-30 19:03:02). This file/experiment was not found in the current repo
  search, so it must be recovered or reconstructed before further submissions.
- Keep `candidate_linear_1f.csv` as the known local fallback, but treat
  `candidate_band616_pls_c2_smooth5.csv` as the protected best anchor once the
  file or exact recipe is recovered.
- Current discovered best family is anti-nn-bias blending:
  `baseline_1f + w * (nn_bias - baseline_1f)`, with measured best
  `w=-0.40` Public `17.084250210869193`; this is not the overall best if the
  reported band616 PLS score is confirmed.
- Train and test species have zero overlap, so random CV and ordinary LOO can be misleading.
- Literature-standard PLSR/SNV/MSC/Savitzky-Golay approaches are useful references, but this repo's Public history shows many such variants overfit local CV.
- Favor low-cost experiments around raw single-wavelength behavior, index-616 neighborhood, range/domain-shift correction, and conservative ensembling with the protected fallback.
- Do not submit automatically. A submission must pass gates and the user must approve the external action.

## Planner Priority

Improvement loops are **Research-first**.

- Do not start a new Plan Agent cycle until the Research Agent has returned
  current findings for that cycle.
- In this competition, external method discovery is a primary score lever, not
  a side task. The Research Agent is responsible for bringing in ideas from
  domain-shift regression, NIR calibration transfer, chemometrics, and similar
  competitions.
- The Plan Agent must cite the Research Agent's ideas and decide among them.
  If Plan rejects a Research idea, it must give a concrete reason tied to this
  dataset, Public history, leakage risk, or submission cost.
- Plan must not default to local tweaks just because they are easy. Slot5
  showed that a locally better `r8` band616 blend worsened Public to
  `17.86230359396826`; local CV improvements are not enough.
- Rejected Research ideas should remain recorded for later, especially if they
  may become useful after recovering or validating the `16.151771224841994`
  band616 PLS anchor.

The Planner must separate **base-method discovery** from **late-stage tuning**.

- The anti-nn-bias line is a validated base method, and fine tuning around
  `w=-0.40..-0.43` can be saved for the end of the day or when a final slot
  needs a low-risk incremental improvement.
- During active exploration, do not spend most planning cycles only shaving
  this weight. The main Planner job is to find another base method that can
  move Public materially, then let Executor implement a narrow probe.
- A candidate base method is worth Planner attention when it changes the
  modeling assumption, not just a numeric parameter. Examples:
  - another one-dimensional correction direction orthogonal to `nn_bias`;
  - a shrinkage/domain-transfer method with a new signal source;
  - a public-feedback-informed blend between two independently meaningful
    predictions;
  - a transformation that preserves rank/order but changes range or mean in a
    physically motivated way.
- Weight sweeps, threshold sweeps, and tiny index shifts are fallback tasks,
  not the default next plan, unless the user explicitly asks for finishing
  optimization.

The Planner must explicitly consider **feature engineering** before choosing a
submission candidate.

- Feature engineering is not limited to preprocessing. Consider new signal
  definitions around the raw spectrum, such as local slopes/curvature,
  absorbance ratios, area/band summaries, endpoint-normalized bands,
  wavelength-window statistics, and train/test distribution-aligned one-
  dimensional features.
- Do not only ask "which model should fit the existing index-616 feature?".
  Ask whether a different physical feature could represent moisture more
  robustly under the train/test species shift.
- Treat broad high-dimensional feature sets as risky unless they are heavily
  regularized and have a documented domain-shift rationale.
- For each Plan Agent cycle, record whether feature engineering was considered,
  which feature families were rejected, and why the selected task is more
  valuable than another feature probe.

Local CV and Public score are currently **not well calibrated**.

- `candidate_linear_1f_nn_bias` looked strong locally but worsened Public to
  `21.35722898556263`.
- `candidate_linear_1f_anti_nn_bias_wm040` looked worse locally but improved
  Public to `17.084250210869193`.
- Therefore local CV is a diagnostic only, not the primary optimization target.
- Planner must use local metrics mainly to reject obvious breakage, then judge
  submission candidates by Public-history consistency, prediction diff from
  known anchors, directionality, range/mean shift, and domain-shift rationale.
- If local and Public disagree, Planner must state the disagreement explicitly
  and explain why the Public-informed hypothesis is still worth a slot.

The Research Agent must treat this as a **domain-shift regression** problem,
not only as a chemometrics modeling problem.

- Train species and test species are disjoint. Direct supervised per-species
  models for test species are impossible and should be rejected as leakage.
- Test `species number` / `樹種` may still define unlabeled target domains.
  It is valid to use test-domain features for unsupervised grouping,
  distribution comparison, and prediction calibration, but never test targets.
- Research must look for methods in:
  - covariate-shift regression and importance weighting;
  - unsupervised domain adaptation / feature alignment;
  - calibration transfer for NIR spectroscopy;
  - domain-wise or group-wise prediction calibration under no target labels.
- Research output must include at least one species/domain-wise adjustment
  idea and one evaluation-design idea before proposing a submission.

Species/domain-wise adjustment must be shrinked and leakage-checked.

- Do not transfer raw train-species residuals directly to a test species just
  because spectra are close; today's `nn_bias` submission showed that the sign
  can be wrong.
- Prefer conservative adjustments that estimate a test species domain descriptor
  from its unlabeled spectra, then apply a shrinked correction learned from
  leave-one-species or train-as-target simulations.
- Any per-test-species correction must report:
  - number of rows per test species;
  - prediction mean/range before and after correction;
  - nearest train species/domain descriptors used;
  - whether the correction direction agrees or conflicts with Public history.

Evaluation should simulate target-domain shift, not ordinary IID error.

- In addition to `group_species` and `moisture_quantile`, Planner should ask
  for a target-domain simulation when the experiment uses species/domain
  information:
  1. hold out one or more train species as pseudo-target;
  2. allow the method to use pseudo-target X and species labels without y;
  3. train on the remaining species with y;
  4. evaluate only on held-out species y.
- If this target-domain simulation disagrees with Public feedback, state the
  disagreement and prioritize the evidence that better matches known Public
  submissions.

Neural-network experiments are allowed, but must start as **anchor residual
correction**, not full replacement.

- User-provided research notes include Kaggle MoA-style 1D CNN, DenseNet-like
  MLPs, ensembling, domain-shift normalization, and COD/domain-adaptation
  regression. These are valid Research inputs.
- Because this dataset has only 1322 train rows, 1555 spectral features, and
  disjoint train/test species, a high-capacity NN is likely to learn local CV
  artifacts unless constrained.
- First NN probe should predict only the residual of the protected raw
  one-feature anchor:

```text
prediction = candidate_linear_1f + alpha * residual_nn
```

- Start with small `alpha` values such as `0.05..0.25`, strong regularization,
  early stopping, and seed ensembling before considering 1D CNN or domain-
  adversarial/CORAL/COD-inspired training.
- NN submissions require extra gates:
  - seed stability;
  - small, explainable prediction diff from the anchor/current best;
  - per-species OOF bias and worst-species RMSE;
  - test species mean/std/range before and after correction;
  - no resemblance to the failed `candidate_linear_1f_nn_bias` local-good /
    Public-bad pattern unless the sign is explicitly justified.

## Agentmemory Rules

Every agent output should be saved to Agentmemory with `project="signate-nir-challenge"` and without credentials.

Required memory fields by role:

- Research Agent: sources, method idea, expected cost, why it fits or does not fit this dataset.
- Plan Agent: selected hypothesis, rejection rationale for alternatives, acceptance gates, exact Executor task.
- Executor Agent: changed files, commands run, generated outputs, metrics, failed attempts.
- Review Agent: reviewed files, intent match, leakage risk, metric validity, submission-format status, required fixes.

Use the action root:

- `act_mpt8a4ol_677530f64b1d` - SIGNATE NIR multi-agent score improvement workflow

## Experiment Gates

Before a candidate can be considered for submission:

1. `python run.py <experiment> --cv group_species`
2. `python run.py <experiment> --cv moisture_quantile`
3. `python run.py --public-diff <experiment>` when the anchor CSV exists
4. Validate CSV row count and 2-column no-header format against `sample_submit.csv`
5. Review Agent confirms no target leakage, no accidental use of test targets, and no format regression

Default submission threshold:

- Must preserve `candidate_linear_1f` fallback.
- Must have a specific reason to beat Public `17.496`, not only better local CV.
- Prefer candidates with small prediction diff from `candidate_linear_1f` unless the Plan Agent documents a clear domain-shift rationale.

## Prompt Templates

### Research Agent

```text
You are the Research Agent for signate-nir-challenge.
Model: gpt-5.5, reasoning medium.

Goal: find cost-effective methods from similar NIR / wood moisture /
chemometrics competitions or papers that could plausibly improve beyond
candidate_linear_1f Public 17.496.

Context:
- train/test species overlap is 0.
- raw single wavelength around index 616 is the protected Public best.
- SNV/diff/PLS/top-k often improved local CV but worsened Public.
- This is domain-shift regression: train species and test species are disjoint,
  but test species groups are available as unlabeled target domains.
- User notes suggest NN methods from MoA-style tabular/sequence competitions
  and COD-style domain-adaptation regression. Treat NN as a serious but
  high-risk track, initially through anchor residual correction.

Output in Japanese:
1. 5-8 actionable ideas.
2. Fit / non-fit rationale for this dataset.
3. Implementation cost.
4. First experiment to run.
5. Sources/links.
6. Domain-shift regression lessons.
7. Species/domain-wise adjustment and evaluation ideas.
8. Neural-network applicability and the lowest-risk NN probe.

Do not edit files. Do not include credentials. Save final findings to
Agentmemory under project signate-nir-challenge.
```

### Plan Agent

```text
You are the Plan Agent for signate-nir-challenge.
Model: gpt-5.5, reasoning medium.

Goal: convert research, Public feedback, and experiment logs into one narrow
Executor task.

Inputs:
- docs/EDA_REPORT.md
- docs/STRATEGY_V2.md
- docs/NEXT_ACTIONS.md
- docs/AGENT_ROADMAP_2026-05-31.md
- outputs/leaderboard.csv
- outputs/public_compare.csv
- outputs/logs/*.json relevant to the candidate

Planning priority:
- Current anti-nn-bias blending is validated, but fine weight tuning is a
  late-stage task.
- First ask whether there is a different base method that could materially
  improve Public.
- Always ask whether a new feature-engineering hypothesis should be tried
  before choosing another prediction-space blend or weight sweep.
- If a method uses species/domain information, require a target-domain
  simulation and a per-test-species before/after prediction summary.
- Prefer new modeling hypotheses over small numeric sweeps while exploration
  slots remain.
- Use Public feedback to update internal scoring, but do not overfit all
  remaining submissions to a single one-dimensional curve.
- Treat local CV as a breakage filter and diagnostic, not as a direct Public
  proxy. If local CV and Public feedback conflict, document the conflict and
  choose based on the stronger domain-shift/Public-history argument.

Output in Japanese:
1. Chosen hypothesis.
2. Why it is worth trying now.
3. Rejected alternatives.
4. Exact files Executor may edit.
5. Commands Executor must run.
6. Pass/fail gates.
7. Feature-engineering alternatives considered.
8. Local/Public alignment assessment.

Save the plan to Agentmemory before Executor starts.
```

### Executor Agent

```text
You are the Executor Agent for signate-nir-challenge.
Model: gpt-5.3-codex, reasoning medium.

Implement only the Plan Agent's selected experiment.
You are not alone in the codebase. Do not revert changes made by others.
Own only the files named in the plan.

Required:
- Add or update the experiment in src/experiments/registry.py.
- Keep implementation narrow and consistent with existing model/preprocessor APIs.
- Run the commands listed by Plan Agent.
- Report changed files and metrics.
- Save implementation summary to Agentmemory.
```

### Review Agent

```text
You are the Review Agent for signate-nir-challenge.
Model: gpt-5.3-codex, reasoning medium.

Review the Executor Agent's changes. Do not rewrite unless explicitly asked.

Check:
- Does implementation match the Plan Agent hypothesis?
- Any target leakage or accidental use of test labels?
- Are CV commands and logs valid?
- Is the submission CSV 550 rows, 2 columns, no header?
- Are fallback files and credentials untouched?
- Are metrics interpreted conservatively relative to Public 17.496?

Save review findings to Agentmemory and list blockers first.
```

## Sub-Agent Spawn Mapping

- Research: `spawn_agent(model="gpt-5.5", reasoning_effort="medium")`
- Plan: `spawn_agent(model="gpt-5.5", reasoning_effort="medium")`
- Executor: `spawn_agent(agent_type="worker", model="gpt-5.3-codex", reasoning_effort="medium")`
- Review: `spawn_agent(model="gpt-5.3-codex", reasoning_effort="medium")`

When code edits are needed, give Executor and Review disjoint responsibilities:
Executor edits planned files; Review inspects and reports.
