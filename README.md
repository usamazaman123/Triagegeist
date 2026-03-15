# Triagegeist: AI-Assisted Emergency Triage with Equity Analysis

**Competition:** [Triagegeist — Kaggle](https://www.kaggle.com/competitions/triagegeist)  
**Sponsor:** Laitinen-Fredriksson Foundation, Helsinki, Finland  
**Prize Pool:** $10,000  

---

## What This Does

Predicts Emergency Severity Index (ESI) acuity levels (1–5) from patient data collected at ED intake — vitals, demographics, comorbidity history, and chief complaint. The model is designed to work like a passive second opinion: given what the nurse observes at the bedside, what does the physiology suggest about urgency?

A secondary analysis examines whether triage errors (undertriage specifically) distribute non-randomly across patient demographics and nursing staff — because in real EDs, they don't.

---

## Approach

### Two Models, Two Questions

**Model A — Rule Validator**  
Trained on all features including calculated scores (NEWS2, shock index, MAP). Reaches near-perfect performance. Not a clinical claim — it confirms our NEWS2-to-ESI label derivation is internally consistent.

**Model B — Bedside Decision Support** ← *this is the submission model*  
Trained on raw bedside-observable features only. No derived aggregate scores. F1-macro = 0.9861, QWK = 0.9982 on out-of-fold validation.

The split is deliberate. Model A validates the methodology. Model B tests whether AI can support a triage nurse using only what's available at first contact.

### Label Derivation

The competition dataset has no pre-labeled acuity column. Ground truth was derived using NEWS2 thresholds combined with ESI v4 absolute vital sign cutoffs:

| ESI Level | Criteria |
|-----------|----------|
| 1 — Immediate | GCS ≤ 8, SpO2 < 85%, SBP < 70, NEWS2 ≥ 9, or shock index > 1.4 |
| 2 — High Risk | NEWS2 ≥ 7, SpO2 < 92%, SBP < 90, GCS ≤ 12, SI > 1.1, HR > 130 |
| 3 — Urgent | NEWS2 ≥ 5, SpO2 < 95%, SBP < 100, GCS ≤ 14, HR > 110 |
| 4 — Less Urgent | NEWS2 ≥ 3 or HR > 100 |
| 5 — Non-Urgent | All other presentations |

Reference: Royal College of Physicians NEWS2 guidelines + ESI v4 algorithm.

### Algorithm

LightGBM (gradient-boosted trees). Chosen for native handling of mixed feature types, fast training, and clean SHAP integration. Validated with stratified 5-fold cross-validation.

```
learning_rate:    0.05
num_leaves:       63
feature_fraction: 0.8
bagging_fraction: 0.8
early_stopping:   50 rounds
random_state:     42
```

---

## Results

| Model | F1-Macro | QWK |
|-------|----------|-----|
| Model A — Full features | 0.9992 | 0.9999 |
| Model B — Bedside only | **0.9861** | **0.9982** |

All errors in Model B are adjacent-class disagreements (e.g. ESI-2 predicted as ESI-3). No ESI-1 patient was predicted as ESI-4 or ESI-5.

**Test set submission distribution:**

| ESI Level | Count | % |
|-----------|-------|---|
| 1 — Immediate | 3,442 | 17.2% |
| 2 — High Risk | 1,906 | 9.5% |
| 3 — Urgent | 3,415 | 17.1% |
| 4 — Less Urgent | 1,755 | 8.8% |
| 5 — Non-Urgent | 9,482 | 47.4% |

ESI-5 majority at ~47% is consistent with real ED volume patterns.

---

## Equity Analysis

Undertriage was defined as the model predicting a higher ESI number (lower urgency) than the NEWS2-derived label warrants.

Four demographic dimensions were analyzed:

- **Sex** — differential undertriage rates between male and female patients, consistent with documented patterns in female chest pain presentations
- **Insurance type** — higher undertriage among uninsured and Medicaid patients vs. privately insured
- **Language** — elevated undertriage rates for non-English speakers, reflecting known communication barriers at intake
- **Age group** — higher undertriage in elderly patients (≥65), likely from atypical presentation patterns

**Nurse-level analysis** across staff with ≥30 patient encounters found wide variability in individual undertriage rates — a signal directly useful for institutional quality monitoring.

---

## Repository Structure

```
triagegeist/
├── README.md                   ← you are here
├── triagegeist_notebook.ipynb  ← full competition notebook
├── requirements.txt            ← Python dependencies
└── outputs/
    ├── submission.csv
    ├── model_performance.png
    ├── model_comparison.png
    ├── shap_esi1_bar.png
    ├── shap_beeswarm_esi1.png
    ├── shap_global.png
    └── equity_analysis.png
```

---

## How to Reproduce

### Requirements

```
python >= 3.10
lightgbm >= 4.0
shap >= 0.45
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
```

Install with:

```bash
pip install -r requirements.txt
```

### Run on Kaggle

1. Open `triagegeist_notebook.ipynb` in a Kaggle notebook
2. Add the Triagegeist competition dataset as input
3. Enable GPU accelerator (T4 x2) and Internet ON
4. Run all cells — end-to-end runtime is approximately 10–15 minutes
5. Output: `submission.csv` + all analysis plots

### Run locally

```bash
git clone https://github.com/YOUR_USERNAME/triagegeist
cd triagegeist
pip install -r requirements.txt

# Update data path in notebook cell 1:
# PATH = './data/'  (place competition CSVs here)

jupyter notebook triagegeist_notebook.ipynb
```

---

## Data

All data is from the Triagegeist competition input files:

| File | Records | Description |
|------|---------|-------------|
| `train.csv` | 20,000 | Patient encounters, features only (no acuity label) |
| `test.csv` | 20,000 | Test encounters for submission |
| `chief_complaints.csv` | 100,000 | Free-text and system-level complaint data |
| `patient_history.csv` | 100,000 | Binary comorbidity flags (25 conditions) |
| `sample_submission.csv` | 20,000 | Required submission format |

No external datasets were used. Data is used for competition purposes only per the competition rules.

---

## Limitations

**Labels are derived, not observed.** Ground truth ESI levels were computed from NEWS2 thresholds and ESI v4 cutoffs — not from actual clinician assignments. Model performance reflects agreement with a rule-based baseline, not with real triage judgments. Validation against prospective nurse-assigned scores would be needed before any clinical use.

**Dataset is likely synthetic.** Class separability is unusually clean. Real ED data is noisier, has more missing values, and contains genuinely ambiguous presentations. Real-world performance would be lower.

**Equity analysis is observational.** Demographic undertriage rate differences reflect model output patterns relative to derived labels. Causal claims about structural bias require linkage to actual patient outcomes.

---

## Potential Clinical Applications

**Short term:** Passive alert layer in ED triage software. When model-predicted acuity diverges from the nurse's assessment by ≥2 levels, prompt a second review. Not an override — a flag.

**Medium term:** Monthly institutional quality monitoring using the nurse-level inter-rater analysis. Track reliability trends, identify staff for targeted feedback.

**Long term:** Demographic undertriage audit framework, transferable to any ED with structured intake data. Methodology is the contribution, not just the model.

---

## License

Code: MIT License  
Data: Non-commercial use only per Triagegeist competition rules

---

## Acknowledgments

Laitinen-Fredriksson Foundation for posing the problem and providing the dataset.  
Kaggle platform for hosting.  
LightGBM and SHAP open-source communities.
