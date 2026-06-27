# AI-Powered Sickle Cell Screening & Crisis-Prediction Tool

## Methodology & Study-Design Proposal

**Context:** Ghana National SCD Strategy — community-level (CHPS zone) screening support
**Status:** Design phase. No model code yet. This document defines methods, data requirements, and study design.
**Guiding constraint:** The tool *complements, never replaces* clinical care. All outputs are decision-support, not diagnosis.

---

## 1. Problem decomposition

This is **two distinct ML problems** that share a product surface but no modeling machinery. They must be designed, validated, and risk-assessed separately.

| | **Task 1 — Cell Classification** | **Task 2 — Crisis (VOC) Prediction** |
|---|---|---|
| Input | Smartphone blood-smear image (pixels) | Clinical + environmental tabular / longitudinal data |
| ML family | Computer vision (CNN, transfer learning) | Survival analysis / gradient-boosted trees |
| Output | Sickle vs. normal (+ % sickled / count) | Short-horizon risk of vaso-occlusive crisis |
| Label source | Microscopy ground truth | Documented crisis events over time |
| Primary failure mode | Misclassifies a cell on a poor-quality image | Misses an impending crisis (false negative) |
| Hardest practical part | Image quality variance from phones | Obtaining longitudinal labeled outcome data |
| Readiness | Prototype-able now with public data | **Gated on data that may not yet exist** |

**Design decision:** Build as two pipelines reporting into one app. Task 1 is tractable immediately. Task 2 is scientifically harder and data-gated — treat it accordingly.

---

## 2. Task 1 — Sickle vs. normal RBC classification

### 2.1 Methods (in recommended trial order)

1. **Classical baseline (interpretable sanity check).** Segment cells → extract shape descriptors (circularity, eccentricity, aspect ratio, solidity, perimeter:area) → SVM / Random Forest. Sickle cells are *defined* by morphology, so this is a meaningful, explainable baseline and a guard against CNN over-optimism.
2. **Transfer learning CNN (primary method).** Fine-tune an ImageNet-pretrained backbone. Do **not** train from scratch — datasets will be small.
   - Backbones, prioritized for eventual **on-phone inference** in low-connectivity zones: **MobileNetV3**, **EfficientNet-B0**, then **ResNet-18** as a heavier reference.
3. **Object detection (if quantification/localization is the chosen output).** YOLOv8 / RT-DETR to locate and count sickle cells in one pass.

### 2.2 The output-format decision (currently OPEN)

Three framings, with different data-labeling burdens. **This must be decided by clinical value AND available label granularity:**

| Framing | Output | Pros | Cons / data need |
|---|---|---|---|
| **(A) Whole-image flag** | "Sickle cells present? Y/N" | Fastest to prototype; image-level labels only | Weakest clinical signal; no quantification |
| **(B) Single-cell** | Segment → classify each → **% sickled** | Matches how hematologists reason; interpretable | Needs cell-level segmentation + labels |
| **(C) Detection** | Locate + **count** sickle cells | Quantitative; visual overlay for trust | Needs bounding-box annotations |

**Decision criterion:** If you can obtain cell-level or box-level annotations → prefer (B) or (C) (clinically richer). If only image-level labels exist → start with (A) and treat (B)/(C) as a later upgrade. **Recommendation:** target (B) as the clinical goal, but ship (A) first as a working baseline.

### 2.3 Task 1 pipeline

```
Smartphone image
  → [1] Image-QC gate      reject blurry / over- or under-stained images BEFORE inference
  → [2] Preprocessing       color/stain normalization, illumination correction
  → [3] Segmentation        isolate individual RBCs  (Cellpose / StarDist / watershed / detector)   [framing B/C]
  → [4] Classification CNN   sickle / normal
  → [5] Aggregation         % sickled, count, confidence
  → [6] Uncertainty gate    low-confidence → "refer to clinician / retake image"
```

Steps **[1]** and **[6]** are non-negotiable for a deployable tool. A model that confidently scores a blurry phone photo is actively dangerous. The QC gate is a hard requirement, not a nicety.

### 2.4 Candidate datasets to start

- **erythrocytesIDB** (sickle-cell smear dataset, cell-level)
- Kaggle sickle-cell / blood-smear datasets (for prototyping framing A)
- **Ghana-specific smartphone images** — collected later for domain adaptation & external validation. Public datasets are microscope-camera images; phone images differ in optics, lighting, compression. Plan for a **domain gap.**

---

## 3. Task 2 — Vaso-occlusive crisis (VOC) prediction

### 3.1 Methods (in recommended trial order)

1. **Gradient-boosted trees (XGBoost / LightGBM)** — binary "crisis within next *N* days (e.g., 30)?" **Strongest starting baseline:** dominates on tabular clinical data, handles missing values natively (critical for sparse rural records), interpretable via **SHAP**.
2. **Survival / time-to-event models** — Cox proportional hazards (clinician-trusted, interpretable) or **Random Survival Forest** / survival GBM. Better matches the real question ("*when* is the next crisis?") rather than a fixed window.
3. **Sequence models (LSTM / Temporal CNN / Transformer)** — *only if* dense per-patient longitudinal series exist. With sparse CHPS-zone visit data these overfit. **Do not start here.**

### 3.2 Candidate features (domain-expert input required)

- **Clinical:** genotype (HbSS vs HbSC), HbF level, baseline hemoglobin, prior crisis frequency, pain history, WBC count, infection/inflammation markers, hydration status, hydroxyurea adherence.
- **Environmental** *(these are the project's novel signal — cold, dehydration, and infection are documented VOC triggers):* temperature, humidity, **harmattan season**, altitude, air quality.
- **Demographic:** age, sex.

### 3.3 The honest constraint — data readiness

Task 2 requires **longitudinal labeled outcome data**: per-patient records over time *with crisis events and dates*. This likely does **not** exist in clean form in target CHPS zones — which is the very reason the problem exists.

**Data-readiness assessment (do this BEFORE any Task 2 modeling):**

| Scenario | Implication for Task 2 |
|---|---|
| Retrospective cohort available (e.g., **Korle-Bu**, **KATH** teaching-hospital records) | Trainable now. Proceed to modeling. |
| Only cross-sectional / single-timepoint data | Task 2 = **prospective data-collection project first**, modeling second. |
| No data | Design Task 2 as a **data-collection protocol + study design**; build Task 1 in parallel. |

Stating this constraint explicitly in a proposal is a **strength**, not a weakness — it shows the work is grounded in data reality.

### 3.4 Task 2 pipeline

```
Clinical + environmental data form (per visit)
  → [1] Feature assembly      join patient record + environmental data (by date/location)
  → [2] Missingness handling  native (GBDT) or principled imputation; record missingness as signal
  → [3] Risk model            XGBoost → survival model
  → [4] Calibration           map raw score → trustworthy probability
  → [5] SHAP explanation      per-patient risk drivers for the health worker
  → [6] Risk tier             low / medium / high + "see clinician"
```

---

## 4. Combined system

```
  Frontline health worker (smartphone, possibly offline)
        │                                   │
  [Blood-smear photo]              [Clinical + env. data form]
        ▼                                   ▼
  TASK 1: Cell CNN                  TASK 2: Risk model
  QC → segment → classify           XGBoost / survival + SHAP
  → % sickled + confidence          → risk tier + drivers
        └──────────────┬────────────────────┘
                       ▼
        Decision-support output
        • Screening flag (refer / not)
        • Risk tier (low / med / high) + confidence
        • "Confirm with clinician"  — NEVER an autonomous diagnosis
```

**FastAPI backend + Flask UI** pattern is a good architectural fit: model service + thin client, extendable to on-device inference later.

---

## 5. Cross-cutting requirements (matter more than model choice)

- **External validation.** Train on one site/phone/population; test on a *different* one. In-distribution accuracy lies — especially across phone models and clinics.
- **Calibration.** A "70% risk" must mean ~70% of such patients actually crise. Report reliability curves, not just AUC/accuracy.
- **Uncertainty + abstention.** The model must be allowed to say "I don't know — refer." Especially for screening, where missing a case is the costly error.
- **Operating-point choice.** Screening favors **high recall** (catch every potential case, tolerate false alarms) over high precision. Set and justify the threshold explicitly.
- **Bias auditing.** Stratify performance by skin tone (affects smear background), phone model, site, age, genotype.
- **Regulatory & ethics.** Frame as **Software as a Medical Device (SaMD)**. Engage Ghana FDA + an IRB/ethics committee early. Patient data → informed consent + privacy/security plan.

---

## 6. Evaluation plan

| | Task 1 | Task 2 |
|---|---|---|
| Primary metric | Recall/sensitivity at fixed specificity; per-class F1 | Recall for crisis; time-dependent AUC / C-index (survival) |
| Calibration | Reliability curve, Brier score | Reliability curve, Brier score |
| Robustness | Performance under blur/lighting/phone shift | Performance under missing features |
| Clinical | Agreement with hematologist on held-out smears | Lead time before crisis; net clinical benefit |
| Fairness | Stratified by skin tone, phone, site | Stratified by genotype, age, site |

**Validation protocol:** patient-level (not image-level) splits to prevent leakage; an explicit **external test site**; prospective evaluation with health-workers in the loop before any deployment claim.

---

## 7. Recommended build order

1. **Data-readiness assessment** — Task 2 cohort inventory (what exists at Korle-Bu / KATH / target zones?); Task 1 label-granularity inventory (image- vs cell- vs box-level). *These resolve the two open decisions in §2.2 and §3.3.*
2. **Task 1 baselines** — classical shape+SVM → MobileNetV3/EfficientNet-B0 transfer learning. Honest held-out number.
3. **Task 1 QC gate + uncertainty + calibration.**
4. **Task 2 baseline** — XGBoost "crisis in 30 days" + SHAP (only if cohort exists; else write the collection protocol).
5. **Thin-client app** (FastAPI + Flask, per repo pattern), then on-device path.
6. **External + prospective validation**, ethics approval, field pilot.

---

## 8. Open decisions to resolve next

1. **Task 2 data source** — inventory retrospective cohorts before committing to modeling vs. collection. *(Currently: unknown — investigate.)*
2. **Task 1 output format** — image-flag (A) vs per-cell %-sickled (B) vs detection/count (C), driven by available label granularity. *(Currently: undecided — recommend targeting B, shipping A first.)*
3. **Deployment mode** — cloud-backed thin client vs. fully offline on-device (CHPS connectivity reality decides this).
