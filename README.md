# Survival Retention Engine: Growth Strategy and Runway Modeling

## Strategic Intent: From Reactive Churn Reporting to Proactive Retention Strategy

How do you move beyond asking whether a customer will leave and begin estimating when risk is likely to emerge, which personas have the shortest runway, and which interventions could defend revenue?

This Python project implements a survival-analysis and strategy-simulation engine for retention and lifecycle analytics. It combines persona segmentation, Cox Proportional Hazards modeling, high-risk scenario simulation, Kaplan-Meier survival curves, and automated executive / technical reporting.

The objective is not only to predict churn-like events. The objective is to quantify customer runway and translate risk movement into a strategy narrative.

This project demonstrates a practical growth-analytics principle:

> Retention strategy is more useful when it explains timing, segment behavior, intervention impact, and model evidence together.

---

## Executive Dashboard

[![Growth Engineering Dashboard](./docs/executive_dashboard_preview.png)](./docs/executive_dashboard_preview.png)

[Open dashboard full size](./docs/executive_dashboard_preview.png)

The dashboard translates survival-modeling concepts into an executive view of retention runway, persona lifecycle differences, defended LTV, and model / data integrity controls.

### Dashboard Interpretation

#### Strategic ROI: Pushing the Curve to the Right

The survival curve shows the probability of continued retention over time. A successful intervention shifts the curve to the right, extending expected customer lifetime and reducing event risk during the observed window.

#### Persona Retention Runways

The persona curves show that different customer groups have different lifecycle patterns. Segment-level survival curves help identify where retention strategy should be targeted rather than treating all customers as one average population.

#### Defended LTV

The defended LTV readout converts risk movement into an executive value narrative. It is intended to show how improved retention timing can translate into protected revenue or avoided attrition impact.

#### Audit Defense Ledger

The dashboard includes an audit-defense narrative layer to reinforce that growth modeling depends on reliable input data, interpretable features, and documented remediation logic. In the source engine, the technical audit support is represented through confidence interval toggles, proportional hazards testing, reproducible clustering, dependency synchronization, and PDF / PPTX evidence outputs.

---

## Modeling Framework

### 1. Persona Segmentation

The engine groups subjects into behavioral personas using K-Means clustering.

Implementation function:

```python
apply_segmentation(df, features, n_clusters=3)
```

Key characteristics:

- standardizes selected features using `StandardScaler`
- applies K-Means clustering with a reproducible random state
- assigns each row to a `Segment_ID`
- creates readable persona labels

This supports the idea that retention strategy should be persona-aware rather than population-average only.

### 2. Cox Proportional Hazards Modeling

The survival model is built with:

```python
CoxPHFitter
```

The default demo uses:

```python
PENALIZER = 0.1
```

The penalizer provides L2 regularization to reduce overfitting risk.

The model is used to estimate partial hazard scores and compare risk across the population.

### 3. High-Risk Scenario Dashboard

The engine identifies the high-risk segment using the 75th percentile of model-predicted partial hazard scores.

Implementation function:

```python
run_scenario_dashboard(model, df, duration_col, event_col)
```

The scenario dashboard compares strategy shifts against the high-risk baseline and calculates:

```text
ROI (Risk Reduction)
```

Conceptually:

```text
baseline high-risk partial hazard
vs.
simulated high-risk partial hazard after strategy shift
```

Default demonstration scenarios include:

```text
Strategic Pivot A
Strategic Pivot B
Strategic Pivot C
```

These are illustrative scenario labels designed to demonstrate how the framework can compare strategy alternatives.

### 4. Dependency Synchronization

The engine includes a synchronization function for engineered feature dependencies.

Implementation function:

```python
sync_dependencies(df_sim)
```

Supported conventions:

| Pattern | Meaning |
|---|---|
| `_x_` | Interaction term, such as `age_x_prio` |
| `_sq` | Squared term, such as `tenure_sq` |

When a strategy simulation modifies an underlying feature, the engine recalculates dependent interaction and squared terms. This prevents mathematical inconsistency during what-if simulations.

### 5. Kaplan-Meier Survival Visuals

The engine generates survival curves for:

- persona-level retention runways
- model-derived risk tiers

Implementation function:

```python
generate_forensic_plots(df, model, duration_col, event_col)
```

The visual outputs support both executive interpretation and technical review.

### 6. Automated Evidence Exports

The engine exports two stakeholder-facing artifacts:

| Output | Purpose |
|---|---|
| `Strategic_Risk_Deck.pptx` | Executive strategy deck with persona survival visuals |
| `Technical_Forensic_Audit.pdf` | Technical audit report with scenario table and risk-tier benchmark visual |

Implementation function:

```python
export_assets(dashboard_df, figs)
```

---

## Technical Architecture

### Core Source File

Primary source file:

[`src/strategic_survival_engine.py`](./src/strategic_survival_engine.py)

### Main Python Libraries

The implementation uses:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `lifelines`
- `python-pptx`
- `reportlab`

### Main Engine Components

| Component | Purpose |
|---|---|
| `apply_segmentation` | Builds K-Means persona groups |
| `sync_dependencies` | Recalculates interaction and squared terms during simulation |
| `run_scenario_dashboard` | Compares strategic scenarios against high-risk baseline |
| `generate_forensic_plots` | Produces persona and risk-tier survival visuals |
| `export_assets` | Creates PPTX and PDF evidence artifacts |
| `cleanup_temp_files` | Removes temporary chart files after export |

### Rigor Toggles

The engine exposes toggles that let the user balance technical rigor and executive simplicity.

```python
USE_CI = True
CHECK_PH = False
```

| Toggle | Purpose |
|---|---|
| `USE_CI` | Shows confidence intervals in survival visuals |
| `CHECK_PH` | Runs proportional hazards assumption checks when enabled |

Additional controls:

```python
PENALIZER = 0.1
RANDOM_STATE = 42
```

| Control | Purpose |
|---|---|
| `PENALIZER` | L2 regularization for CoxPH model |
| `RANDOM_STATE` | Reproducibility for clustering and simulation behavior |

---

## Data Requirements

The engine is designed for survival-style datasets with:

| Requirement | Description |
|---|---|
| Duration column | Numeric time-to-event duration, such as days, weeks, or months |
| Event column | Binary event indicator, where `1` means event occurred and `0` means censored / still active |
| Features | Numeric or Boolean model features |
| Optional engineered features | Interaction terms using `_x_` and squared terms using `_sq` naming conventions |

The included demo uses the `lifelines` Rossi dataset to demonstrate the workflow. The demo adds an interaction term:

```python
age_x_prio = age * prio
```

and then fits a CoxPH model using the available survival data.

---

## Repository Contents

```text
survival-retention-engine/
│
├── README.md
│
├── docs/
│   └── executive_dashboard_preview.png
│
└── src/
    └── strategic_survival_engine.py
```

### Core Artifacts

| Artifact | Purpose |
|---|---|
| [`docs/executive_dashboard_preview.png`](./docs/executive_dashboard_preview.png) | Executive dashboard preview |
| [`src/strategic_survival_engine.py`](./src/strategic_survival_engine.py) | Survival modeling, persona segmentation, scenario simulation, and reporting engine |

---

## How to Run

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn lifelines python-pptx reportlab
```

### 2. Run the Demo Engine

```bash
python src/strategic_survival_engine.py
```

The demo pipeline will:

1. load the Rossi sample dataset from `lifelines`
2. create an interaction feature
3. apply K-Means persona segmentation
4. fit a Cox Proportional Hazards model
5. run high-risk strategy simulations
6. generate survival plots
7. export executive and technical reporting assets

### 3. Review Generated Outputs

The default script produces:

```text
Strategic_Risk_Deck.pptx
Technical_Forensic_Audit.pdf
```

The console also prints a completion message when the pipeline finishes.

---

## Example Workflow

```text
Input survival dataset
→ Persona segmentation
→ CoxPH model fit
→ High-risk segment identification
→ Scenario simulation
→ Dependency synchronization
→ Risk-reduction calculation
→ Kaplan-Meier visuals
→ Executive PPTX
→ Technical PDF
```

---

## Executive Interpretation

### Predicting the Runway

The engine reframes churn-like risk from a binary question into a time-based strategy question:

```text
Not only: will the customer leave?
But also: when does risk emerge?
```

This allows intervention strategy to be timed more precisely.

### Persona Lifecycle Strategy

K-Means segmentation creates behavioral personas so the model can show which groups have the shortest survival runway and which groups retain longer.

This supports targeted strategy rather than one-size-fits-all retention treatment.

### Strategy Simulation

The high-risk dashboard tests how strategic changes affect the partial hazard profile of the riskiest segment.

This gives stakeholders a way to evaluate whether an intervention reduces modeled risk before committing operational resources.

### Growth Narrative

The dashboard translates survival modeling into a business-facing story:

```text
extend the retention curve
reduce high-risk hazard
defend LTV
prioritize intervention timing
```

### Model Evidence

The technical outputs support review through survival curves, risk-tier benchmarking, scenario tables, and optional proportional hazards assumption checks.

---

## Supported Boundaries

This project is a survival-analysis and retention-strategy demonstration, not a production churn system.

Important boundaries:

- The included demo uses the Rossi dataset from `lifelines`, not a proprietary customer dataset.
- Dashboard dollar values and defended LTV language are illustrative portfolio narrative metrics.
- Scenario labels are demonstration labels and should be replaced with business-specific strategy names in production analysis.
- The CoxPH model should be validated, monitored, and recalibrated before any production use.
- Proportional hazards assumptions should be evaluated during model development and validation.
- Feature engineering, censoring definitions, intervention design, and event definitions must be governed by the applicable business context.

---

## Data Privacy and Interpretation Boundaries

All data and visual outputs in this repository are generated from synthetic, demonstration, or publicly available sample data.

This framework demonstrates methodology for growth analytics and retention strategy, but it does not expose real customer data, proprietary strategy rules, confidential revenue models, or production retention pipelines.

Important interpretation boundaries:

- Survival outputs are modeling demonstrations, not production retention forecasts.
- Defended LTV is an executive interpretation layer, not a booked financial result.
- Risk reduction is based on model-predicted partial hazard movement in the demo framework.
- The exported PPTX and PDF are stakeholder evidence artifacts, not formal model validation packages.

---

## Portfolio Philosophy

**No Cold Handoffs** — engineering zero-defect, audit-ready results so stakeholders internalize the underlying “why.”

This project is designed to ensure that survival modeling does not remain a technical black box. The goal is to connect model behavior, persona segmentation, intervention strategy, and executive growth narrative in a way that business, analytics, and technical stakeholders can understand and challenge.
