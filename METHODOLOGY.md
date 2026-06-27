# Akofena — Methodology & Technical Architecture

**An AI decision-support tool for sickle-cell disease screening and crisis prediction in community health settings (Ghana, CHPS-zone).**

> **Guiding constraint:** Every output is *decision support for a health worker* — a flag, a risk tier, a confidence level — never an autonomous diagnosis. The system is designed to make a frontline worker faster and safer, not to replace a clinician.

---

## 1. Overview

Akofena turns a low-cost smartphone and routine clinic data into two complementary early-warning signals for sickle-cell disease (SCD):

1. **Smear screening** — from a photo of a blood smear, identify whether sickled red blood cells are present and estimate *how many*, so a community worker can decide who to refer for confirmatory testing.
2. **Crisis-risk prediction** — from a patient's routine clinical record and local environmental conditions, estimate the short-term risk of a *vaso-occlusive crisis* (the painful, dangerous episodes that drive most SCD hospitalisations) — early enough to act.

Both run behind one simple interface for the frontline worker and are engineered to function in low-connectivity settings.

---

## 2. Why this is two problems, not one

These look like one product but are two fundamentally different machine-learning problems. Treating them separately — with separate data, models, and safety checks — is a deliberate design decision, not an accident of scope.

| | **Task 1 — Smear screening** | **Task 2 — Crisis prediction** |
|---|---|---|
| **Input** | A blood-smear image (pixels) | A patient's clinical history + environment |
| **AI approach family** | Computer vision | Risk / time-to-event modelling |
| **Output** | Sickle cells present? + % sickled | Short-horizon crisis-risk tier |
| **Hardest part** | Image quality varies across phones | Learning *when* a crisis is coming |
| **Costliest error** | Missing a positive smear | Missing an impending crisis |

Each is validated and risk-assessed on its own terms, then surfaced through a shared decision-support layer.

---

## 3. Task 1 — Smear screening (computer vision)

**The idea in one line:** teach a model to recognise the elongated, crescent shape that defines a sickled cell, the same morphological cue a hematologist uses.

**Approach, in increasing sophistication:**

- **An interpretable shape-based baseline first.** Because a sickle cell is *defined* by its shape, we begin with classical morphology measures (how elongated, how round, how regular each cell is). This gives an explainable, hard-to-fool reference point — and a guard against a deep model that looks accurate for the wrong reasons.
- **A transfer-learning vision model as the primary method.** Rather than training a large network from scratch (which small medical datasets cannot support), we adapt a model already pretrained on millions of general images and specialise it to red-blood-cell morphology. Backbones are chosen with **on-device inference in mind**, so the model can eventually run on a mid-range phone without a server.
- **Cell-level quantification, not just a yes/no.** Because our annotated smear data marks individual cells (normal vs. elongated), the model can report a **percent-sickled estimate** — far more clinically useful than a whole-image flag, and closer to how a lab actually reasons. Localising and counting cells for a visual overlay is a natural extension of the same approach.

---

## 4. Task 2 — Crisis-risk prediction (clinical + environmental modelling)

**The idea in one line:** combine what we know about the patient with what's happening in their environment to estimate how soon the next crisis is likely.

**Approach:**

- **Tree-ensemble and time-to-event models, deliberately — not deep learning.** Routine clinical data is tabular, sparse, and full of gaps. On this kind of data, gradient-boosted tree ensembles and survival (time-to-event) models consistently *outperform* neural networks while being cheaper to train and far easier to explain. Choosing the simpler, stronger tool here is itself a mark of methodological maturity.
- **Environment as the novel signal.** Cold exposure, dehydration, and seasonal conditions (e.g. the harmattan) are documented crisis triggers. By joining each patient encounter to local environmental conditions, the model learns triggers that a record-only model would miss — this is a core part of the project's contribution.
- **The clinician's own words as an auxiliary signal.** Free-text clinical notes carry information that structured fields miss; we treat them as a complementary input, not a replacement for structured data.
- **Framed as "risk over the coming window," not a label.** The output is a calibrated probability and a risk tier — *low / medium / high*, plus the few factors driving it — designed to give a worker actionable lead time.

---

## 5. The trust layer — what makes it deployable, not just demoable

A model that confidently scores a blurry photo, or gives a precise-looking risk it can't back up, is *actively dangerous* in a clinical setting. The following safeguards are first-class components of the architecture, not afterthoughts:

- **Input-quality gate.** Poor-quality smear images and incomplete clinical forms are caught *before* the model runs — the system asks for a retake or the missing field rather than guessing.
- **The right to abstain.** When the model is uncertain, it is allowed to say *"I'm not sure — refer to a clinician."* For screening, a missed case is the expensive error, so the system is tuned to err toward referral.
- **Honest probabilities (calibration).** A stated "70% risk" is checked to genuinely mean roughly 70% — so a health worker can trust the number, not just the rank.
- **An explanation with every output.** Each result comes with *why*: the cells or the factors that drove it. Clinicians and regulators do not trust black boxes.
- **Fairness auditing.** Performance is checked across skin tone, device, site, age, and genotype — so the tool works for everyone it serves, not just the majority group in the training data.
- **Drift monitoring.** Once deployed, the system watches for the input distribution shifting away from what it was trained on — the silent failure mode of real-world ML.

---

## 6. System architecture

```
        Frontline health worker  ·  smartphone, often offline
                  │                                 │
          [ blood-smear photo ]            [ clinical + environment form ]
                  ▼                                 ▼
        ┌───────────────────┐             ┌───────────────────────┐
        │  TASK 1            │             │  TASK 2                │
        │  quality gate →    │             │  completeness check →  │
        │  vision model →    │             │  risk model →          │
        │  % sickled + conf. │             │  calibrated risk tier  │
        └─────────┬─────────┘             └───────────┬───────────┘
                  └───────────────┬───────────────────┘
                                  ▼
                  Unified decision-support output
                  • Screening flag (refer / not)
                  • Risk tier + confidence
                  • Plain-language "why"
                  • "Confirm with a clinician" — never a diagnosis
```

**Deployment posture — built for CHPS reality:**

- **Offline-capable.** The screening model is designed to run *on the device*, so a worker in a low-connectivity zone is never blocked by the network.
- **Frugal by design.** Model training is a one-time cost on a single modest GPU; everyday use runs on a phone or a low-power machine. This keeps recurring cost low — a sustainability requirement for a low-resource health system, not just a nice-to-have.
- **A thin client over a model service.** The worker's app stays simple; the intelligence sits behind a clean interface that can scale from a single clinic to many.

---

## 7. How we'll know it works (validation philosophy)

Accuracy on the data you trained on is misleading. Our evaluation is built around honesty:

- **Split by patient, not by sample** — so the model is never tested on a patient it has already seen, the most common way clinical ML fools itself.
- **Test on a genuinely different setting** — a different clinic, phone, or population — because in-distribution numbers lie.
- **Measure calibration, not just accuracy** — does "70%" really mean 70%?
- **Audit for fairness across subgroups** before any deployment claim.
- **Keep a human in the loop** — field evaluation with health workers using the tool, before any wider rollout.

---

## 8. Feasibility

This architecture is grounded in data already assembled, not a hypothetical. We hold **cell-level annotated blood smears** (individual cells marked as normal or sickled), which is precisely what enables the percent-sickled approach in Task 1. We also hold **longitudinal patient records** — many encounters per patient over time, carrying crisis indicators, clinical measurements, environmental context, and clinical notes — which is exactly the structure crisis prediction in Task 2 requires. The remaining work is selecting and refining within these assets and validating against the real deployment setting — which is the stage the project is at now.

---

*Akofena is decision support for human clinicians. It is designed to extend the reach of scarce clinical expertise into the communities that need it most — never to replace it.*
