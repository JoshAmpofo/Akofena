# Akofena

**An AI decision-support tool for sickle-cell disease (SCD) screening and crisis prediction in community health settings — Ghana, CHPS-zone.**

> *Akofena* is the Akan symbol of courage and valour. The project takes its name as a reminder of who it serves: frontline community health workers extending scarce clinical expertise into the communities that need it most.

> **Guiding constraint:** Every output is *decision support for a health worker* — a flag, a risk tier, a confidence level — **never an autonomous diagnosis**. The system is designed to make a frontline worker faster and safer, not to replace a clinician.

**Status:** Design / data-selection phase. The methodology, data requirements, and technical architecture are specified; model code has not yet been written.

---

## The problem

Sickle-cell disease is among the most common genetic disorders in Ghana, yet two gaps drive much of its avoidable harm:

1. **Screening reach.** Confirmatory diagnosis depends on labs and microscopy that are scarce in community (CHPS-zone) settings. Many cases are identified late, or not at all.
2. **Crisis warning.** *Vaso-occlusive crises* (VOC) — the painful, dangerous episodes that drive most SCD hospitalisations — strike with little warning. Cold, dehydration, infection, and seasonal conditions (e.g. the *harmattan*) are documented triggers, but a frontline worker has no tool to translate them into actionable risk.

Akofena turns a low-cost smartphone and routine clinic data into two complementary early-warning signals to close these gaps, in low-connectivity settings, without replacing clinical judgement.

---

## Two problems, not one

These look like one product but are **two fundamentally different machine-learning problems**. Treating them separately — with separate data, models, and safety checks — is a deliberate design decision.

| | **Task 1 — Smear screening** | **Task 2 — Crisis prediction** |
|---|---|---|
| **Input** | A blood-smear image (pixels) | A patient's clinical history + local environment |
| **AI family** | Computer vision | Risk / time-to-event modelling |
| **Output** | Sickle cells present? + % sickled | Short-horizon crisis-risk tier |
| **Hardest part** | Image quality varies across phones | Learning *when* a crisis is coming |
| **Costliest error** | Missing a positive smear | Missing an impending crisis |

---

## Proposed methodology (summary)

### Task 1 — Smear screening (computer vision)
Teach a model to recognise the elongated, crescent shape that defines a sickled cell — the same morphological cue a hematologist uses.

- **An interpretable shape-based baseline first** — classical morphology measures (elongation, roundness, regularity) feeding a simple classifier. Because a sickle cell is *defined* by its shape, this is a meaningful, explainable reference and a guard against a deep model that looks accurate for the wrong reasons.
- **A transfer-learning vision model as the primary method** — adapt a network pretrained on general images and specialise it to red-blood-cell morphology (small medical datasets cannot support training from scratch). Backbones are chosen with **on-device inference** in mind.
- **Cell-level quantification, not just yes/no** — because the annotated smear data marks individual cells, the model can report a **percent-sickled** estimate, closer to how a lab actually reasons.

### Task 2 — Crisis-risk prediction (clinical + environmental modelling)
Combine what we know about the patient with what's happening in their environment to estimate how soon the next crisis is likely.

- **Tree-ensemble and time-to-event models, deliberately — not deep learning.** Routine clinical data is tabular, sparse, and full of gaps; gradient-boosted trees and survival models outperform neural nets here while being cheaper and far easier to explain.
- **Environment as the novel signal** — joining each patient encounter to local conditions (temperature, humidity, harmattan season) lets the model learn triggers a record-only model would miss.
- **Calibrated risk over a window**, not a bare label — a *low / medium / high* tier plus the few factors driving it, designed to give a worker actionable lead time.

### The trust layer — what makes it deployable, not just demoable
First-class components, not afterthoughts:

- **Input-quality gate** — reject blurry smears / incomplete forms *before* the model runs.
- **The right to abstain** — "I'm not sure — refer to a clinician." Screening is tuned to err toward referral.
- **Honest probabilities (calibration)** — a stated "70% risk" is checked to genuinely mean ~70%.
- **An explanation with every output** — the cells or factors that drove it (Grad-CAM for Task 1, SHAP for Task 2).
- **Fairness auditing** across skin tone, device, site, age, and genotype.
- **Validation by patient, not by sample**, with an external test site — because in-distribution numbers lie.

---

## Data

- **`data/sickle_cell_clinical_note.csv`** — longitudinal clinical records supporting Task 2.
- **`Version 2, erythrocytesIDB/`** — the *erythrocytesIDB* blood-smear dataset, with **cell-level annotations** (individual cells marked normal vs. sickled) — precisely what enables the percent-sickled approach in Task 1.

Public smear data comes from microscope cameras; Ghana-specific smartphone images will be collected later for domain adaptation and external validation. A **domain gap** between the two is expected and planned for.

---

## Documentation

| Document | What it covers |
|---|---|
| [`PROPOSAL.md`](PROPOSAL.md) | Methodology & study-design proposal — problem decomposition, methods per task, evaluation plan, build order, and open decisions. |
| [`METHODOLOGY.md`](METHODOLOGY.md) | Plain-language methodology and architecture narrative (the basis for this README's summary). |
| [`DATA_READINESS_ASSESSMENT.md`](DATA_READINESS_ASSESSMENT.md) | Governance/ethics → image-label → cohort inventory checklists to resolve before any model code is written. |
| [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md) | Concrete, reproducible ML stack and framework specification. |

---

## Repository layout

```
Akofena/
├── data/                          # clinical records (Task 2)
├── Version 2, erythrocytesIDB/    # cell-annotated blood-smear dataset (Task 1)
├── notebooks/                     # exploratory analysis
├── PROPOSAL.md                    # methodology & study design
├── METHODOLOGY.md                 # plain-language methodology + architecture
├── DATA_READINESS_ASSESSMENT.md   # governance / data-inventory checklists
└── TECHNICAL_ARCHITECTURE.md      # ML stack specification
```

---

## Responsible-use note

Akofena is **decision support for human clinicians**. It is designed to extend the reach of scarce clinical expertise — never to replace it. The work is framed as Software as a Medical Device (SaMD); ethics/IRB approval, data-use agreements, and de-identification are prerequisites, not afterthoughts (see `DATA_READINESS_ASSESSMENT.md`).
