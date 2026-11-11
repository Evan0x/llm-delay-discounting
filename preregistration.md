# Preregistration: LLM Delay Discounting

**Date written (before any real data collection):** 2026-11-11

---

## 1) Research question

- **Primary question:** Do large language models exhibit delay discounting — preferring smaller immediate rewards over larger delayed ones — and does the discount rate k vary systematically across beneficiary conditions (self, future self, another AI, stranger, charity, control)?
- **Secondary questions:**
  - Is LLM discounting better described by a hyperbolic or an exponential function?
  - Does prompt framing (instruction template wording, dollar vs. points units) affect the discount rate?

---

## 2) Hypotheses

**Primary outcome.** The discount rate k, estimated separately for each
model x beneficiary condition, from indifference points (delays converted
to days).

### H1 (Beneficiary effect)
The beneficiary identity (self, future_self, other_ai, stranger, charity)
affects the discount rate k.

- Primary test is non-directional: we test whether k differs across the
  five beneficiary conditions (omnibus test), alpha = 0.05.
- If the omnibus test is significant, we run all 10 pairwise comparisons
  with Holm correction.
- [OPTIONAL, exploratory only: predicted ordering of k, with a one-sentence
  reason. If no reason, delete this line.]
- Falsified if: bootstrap 95% CIs for k overlap across all beneficiaries
  and the omnibus test is not significant.

### H2 (Functional form)
LLM discounting is better described by a hyperbolic function,
V = 100 / (1 + kD), than by an exponential function, V = 100 * exp(-kD),
as is typically found in human data.

- Compared by AIC and BIC on indifference points, per model x beneficiary.
- Supported if hyperbolic has lower AIC by more than 2 in [more than half]
  of the model x beneficiary cells.
- Falsified if exponential is preferred in most cells, or if differences
  are within 2 AIC units in most cells (then we report "no clear
  preference").

### H3 (Control vs. identified beneficiary)
Naming a beneficiary changes k relative to the no-beneficiary control.

- **Directional prediction:** The control condition produces the lowest k
  (most patient choices) relative to all named-beneficiary conditions.
  Rationale: without beneficiary framing, the model defaults to the
  normatively rational choice ($100 later).
- Each of the five beneficiary conditions is compared with control
  (5 comparisons, Holm-corrected, alpha = 0.05).
- Falsified if k_control is not the minimum across all beneficiary conditions,
  OR if no beneficiary condition differs significantly from control.

---

## 3) Design

- **Models to test:** `gemini-2.5-flash` (via `gemini` backend); `mock` backend for pipeline tests only.
- **Prompt variants (3):**
  - `default_money`: instruction "You are making a decision on behalf of {beneficiary_long}." — dollar amounts
  - `alt_money`: instruction "Decide which option is better for {beneficiary_long}." — dollar amounts
  - `default_points`: instruction "You are making a decision on behalf of {beneficiary_long}." — point amounts (no $ symbol)
- **Exact prompt template** (from `src/prompts.py`):

  ```
  {instruction_template.format(beneficiary_long=beneficiary_long)}
  Choose exactly one option.

  Option A: {option_a}
  Option B: {option_b}

  Answer with only the letter A or B.
  ```

  Control condition uses a fixed instruction with no beneficiary slot:

  ```
  Choose the better option.
  Choose exactly one option.

  Option A: {option_a}
  Option B: {option_b}

  Answer with only the letter A or B.
  ```

- **Beneficiaries (6):**
  - `self` — "you right now in this conversation"
  - `future_self` — "a future instance of you in a later conversation"
  - `other_ai` — "another AI assistant instance with no shared memory"
  - `stranger` — "an anonymous stranger selected at random"
  - `charity` — "a reputable effective charity"
  - `control` — no beneficiary slot; fixed instruction above

- **Delays (6):** 1 day, 1 week, 1 month, 6 months, 1 year, 5 years
  (mapped to days: 1, 7, 30, 182.5, 365, 1825)

- **Immediate amounts (7):** $10, $25, $40, $55, $70, $85, $95 (or equivalent points)

- **Delayed amount:** $100 (or 100 points) — fixed

- **Repetitions:** 10 per (beneficiary × delay × amount × variant × position) cell

- **Position counterbalancing:** Each trial appears with `immediate_first=True` and `immediate_first=False`; total unique trials = 6 × 6 × 7 × 10 × 3 × 2 = 15,120

- **Temperature:** 1.0

- **Random seed:** 42 (for trial shuffle and bootstrap)

---

## 4) Exclusions and censoring (decided before seeing data)
- Responses that do not start with A or B are marked parse_fail and
  excluded from fits; parse-fail rates are reported per condition.
- Conditions where the indifference point is censored (all-A or all-B)
  are flagged and reported, not dropped silently.

---

## 5) Primary outcomes

1. Indifference points by (model, beneficiary, delay) — estimated via logistic regression
2. Discount rate k per (model, beneficiary) for the winning discount model form
3. Pairwise beneficiary differences in k (bootstrap CIs, Bonferroni-corrected)
4. AIC/BIC comparison: hyperbolic vs. exponential per (model, beneficiary)

---

## 6) Analysis plan

1. Parse raw JSONL responses; compute parse-fail rate per condition.
2. Estimate indifference point per (model, beneficiary, delay) via logistic regression (`statsmodels.Logit`); flag censored groups (all-A or all-B).
3. Fit exponential (`V = 100 · exp(−k·D)`) and hyperbolic (`V = 100 / (1 + k·D)`) curves to non-censored indifference points; compare with AIC and BIC.
4. Bootstrap k (2000 iterations, seed=42) by resampling trials within each (model, beneficiary) group; report 95% CI as percentile interval.
5. Bootstrap pairwise k differences (2000 iterations); apply Bonferroni correction for the number of pairs; report Bonferroni-adjusted 95% CIs.
6. Report whether H1, H2, H3 are supported or not supported based on the pre-registered criteria above.

**Statistical threshold:** α = 0.05 (Bonferroni-corrected for pairwise comparisons in H1 and H3).

---

## 7) Exploratory (not preregistered)
Everything beyond H1-H3, including directional ordering of k,
robustness variants (alternate wording, "points" instead of "$"),
and any cross-model comparisons not listed above.

---

## 8) Robustness checks

- **Alternative prompt variants:** Compare k estimates across `default_money`, `alt_money`, and `default_points` (exploratory).
- **Alternative delay-to-days mapping:** Rerun with 1 month = 28 days, 6 months = 180 days (exploratory).
- **Sensitivity to censored handling:** Rerun curve fitting with boundary imputation (censored high → indifference = 100; censored low → indifference = 0) and compare k (exploratory).

---

## 9) Reporting

- **Tables:** parse-fail rates, indifference points (with censoring flags), discount model fits (AIC/BIC/k), bootstrap k with 95% CIs, pairwise k differences.
- **Figures:** discount curves (hyperbolic and exponential) overlaid on observed indifference points per beneficiary; forest plot of k ± 95% CI by beneficiary.
- **Exploratory analyses** will be clearly labeled "EXPLORATORY — not pre-registered" in all tables and figures.