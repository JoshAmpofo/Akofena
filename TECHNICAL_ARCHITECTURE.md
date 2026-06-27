# Technical Architecture & ML Framework Specification

**Companion to:** `PROPOSAL.md`, `DATA_READINESS_ASSESSMENT.md`
**Audience:** Funding reviewers + implementing team
**Assumption for this document:** Required data (per `DATA_READINESS_ASSESSMENT.md`) is available.
**Goal:** A concrete, reproducible, defensible stack that produces a *robust* solution deployable in low-resource CHPS settings.

---

## 1. Design principles (what "robust" means here, and how the stack enforces it)

| Principle | Why it matters for a funder | How the stack enforces it |
|---|---|---|
| **Reproducibility** | Funded work must be auditable & extensible | Pinned deps (`uv`/`conda`), experiment tracking, seeded splits |
| **Frugal deployment** | CHPS zones have weak connectivity & low-end phones | Mobile-first backbones, ONNX/TFLite export, offline-capable client |
| **Honest validation** | Avoids the "great paper, useless in clinic" failure | Patient-level splits, external test site, calibration |
| **Interpretability** | Clinicians & regulators won't trust a black box | SHAP (Task 2), Grad-CAM (Task 1), abstention |
| **Mainstream tooling** | Lowers maintenance risk, eases hiring | PyTorch / scikit-learn / XGBoost — no exotic dependencies |

These principles are the rubric a technical reviewer will apply. Each row below maps to a concrete library choice.

---

## 2. Recommended framework stack

### 2.1 Core (shared)

| Layer | Choice | Rationale |
|---|---|---|
| Language | **Python 3.11** | Matches existing workspace; ecosystem standard |
| Env / deps | **uv** (+ lockfile) | Already used in repo; fast, reproducible |
| Experiment tracking | **MLflow** (or Weights & Biases) | Every run logged: params, metrics, artifacts, model version |
| Config | **Hydra** / YAML | Config-driven runs; no hard-coded hyperparameters |
| Data versioning | **DVC** | Datasets + models versioned alongside code (audit trail) |
| Serving | **FastAPI + Uvicorn** | Matches existing repo pattern; async, OpenAPI docs |
| Containerization | **Docker** (multi-stage) | Reproducible deploy; matches repo's Cloud Run pattern |

> **Funder-credibility note:** MLflow + DVC + pinned envs is what turns "a model" into "a reproducible scientific deliverable." Call this out explicitly in the budget/methods — reviewers reward it.

### 2.2 Task 1 — Computer Vision

| Component | Choice | Rationale |
|---|---|---|
| DL framework | **PyTorch** | Dominant research framework; best ecosystem for vision |
| Training wrapper | **PyTorch Lightning** | Removes boilerplate; built-in mixed precision, checkpointing, reproducibility |
| Models / backbones | **timm** (PyTorch Image Models) | One-line access to MobileNetV3, EfficientNet-B0, ResNet — all ImageNet-pretrained |
| Augmentation | **Albumentations** | Fast; medical-imaging-friendly transforms (stain jitter, blur, compression) |
| Segmentation (framing B) | **Cellpose** or **StarDist** | Pretrained cell segmentation; avoids training masks from scratch |
| Detection (framing C) | **Ultralytics YOLOv8** | Mature, mobile-exportable, locate+count in one pass |
| Explainability | **Grad-CAM** (`pytorch-grad-cam`) | Visual "why" overlay — builds clinician trust |
| Mobile export | **ONNX** → **ONNX Runtime** / **TFLite** | On-device inference path for offline CHPS use |

**Classical baseline (build first):** **scikit-image** (shape descriptors) + **scikit-learn** (SVM/RF). Interpretable sanity check per `PROPOSAL.md §2.1`.

### 2.3 Task 2 — Clinical / Tabular Risk

| Component | Choice | Rationale |
|---|---|---|
| Primary model | **XGBoost** (or **LightGBM**) | Best-in-class on tabular clinical data; native missing-value handling |
| Survival models | **scikit-survival** (RSF, Cox), **lifelines** (Cox, KM curves) | Proper time-to-event; clinician-familiar hazard outputs |
| Explainability | **SHAP** | Per-patient risk drivers; the standard for clinical ML transparency |
| Calibration | **sklearn `CalibratedClassifierCV`**, `calibration_curve` | Probabilities that mean what they say (`PROPOSAL.md §5`) |
| Imbalance handling | class weights / **`imbalanced-learn`** | Crises are rare; avoid majority-class collapse |
| Feature store (env. join) | **pandas** + **xarray** (gridded weather) | Join patient visits to environmental data by date+location |

> **Why XGBoost over a neural net here:** With sparse, missing-heavy, modest-sized rural clinical data, gradient-boosted trees beat deep nets on accuracy *and* interpretability *and* training cost. Choosing the simpler-but-stronger tool is itself a signal of methodological maturity to reviewers.

### 2.4 Application / delivery

| Component | Choice | Rationale |
|---|---|---|
| Backend API | **FastAPI** | Two model services (vision + risk) behind one API |
| Frontend | **Flask** thin client / **Streamlit** demo / **Gradio** for rapid funder demo | Repo already uses Flask; Gradio gives a fast shareable demo |
| Offline mode | **ONNX Runtime Mobile** / **PWA** | Inference without connectivity |
| Monitoring | **Evidently** (data/▲prediction drift) | Post-deployment safety: detect when input distribution shifts |

`★ Gradio is worth highlighting for the proposal:` a Hugging Face Spaces demo gives reviewers a *clickable* prototype, which dramatically strengthens a funding application over screenshots.

---

## 3. End-to-end pipeline architecture

```
                        ┌──────────────────────────────────────────┐
                        │  DATA LAYER (DVC-versioned)               │
                        │  images/ • clinical.csv • environmental/  │
                        └───────────────┬──────────────────────────┘
                                        │
        ┌───────────────────────────────┴────────────────────────────┐
        ▼                                                             ▼
┌─────────────────────────────┐                     ┌──────────────────────────────┐
│ TASK 1 TRAINING (PyTorch     │                     │ TASK 2 TRAINING (XGBoost/      │
│ Lightning + timm)            │                     │ scikit-survival)               │
│  • Albumentations aug        │                     │  • feature assembly (pandas)   │
│  • transfer learn MobileNet  │                     │  • missingness handling        │
│  • patient-level CV split    │                     │  • patient-level CV split      │
│  • Grad-CAM checks           │                     │  • SHAP + calibration          │
│         ↓ MLflow log         │                     │         ↓ MLflow log           │
└──────────────┬───────────────┘                     └───────────────┬───────────────┘
               │  export ONNX/TFLite                                 │  serialize (joblib)
               ▼                                                     ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  SERVING LAYER  (FastAPI, Dockerized → Cloud Run / on-prem)        │
        │   /predict/cell     QC gate → model → uncertainty gate            │
        │   /predict/risk     features → model → calibration → SHAP         │
        │   /health                                                        │
        └───────────────────────────────┬─────────────────────────────────┘
                                         ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  CLIENT  (Flask/PWA, offline-capable, ONNX Runtime Mobile)        │
        │   • capture smear photo + clinical form                          │
        │   • decision-support output: flag • risk tier • confidence       │
        │   • "Confirm with clinician" — never autonomous diagnosis        │
        └─────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  MONITORING (Evidently): input-drift + performance dashboards     │
        └─────────────────────────────────────────────────────────────────┘
```

---

## 4. Robustness engineering (the parts funders should see line-itemed)

These are *not* optional extras — they are what make the difference between a demo and a deployable clinical tool. Budget for them explicitly.

1. **Validation protocol**
   - Patient-level (not image/visit-level) splits — prevents leakage.
   - A held-out **external site** (different clinic / phone / population).
   - Nested CV for honest hyperparameter selection.
2. **Calibration** — reliability curves + Brier score for both tasks; recalibrate per deployment site if needed.
3. **Uncertainty & abstention** — Task 1: softmax-confidence + MC-dropout / deep-ensemble; Task 2: prediction intervals. Low-confidence → "refer to clinician."
4. **Input-quality gates** — Task 1 QC gate (blur/stain rejection) *before* inference; Task 2 completeness check on the clinical form.
5. **Fairness audit** — stratified metrics by skin tone, phone model, site, age, genotype (`PROPOSAL.md §5`).
6. **Drift monitoring** — Evidently dashboards flag when live inputs diverge from training distribution (the silent killer of deployed models).
7. **Reproducibility bundle** — seeded runs, MLflow registry, DVC data hashes, Docker image digests. A reviewer can re-run any result.

---

## 5. Compute & infrastructure (for the budget section)

| Phase | Need | Frugal option |
|---|---|---|
| Task 1 training | 1 GPU (e.g., single T4/A10), hours-to-days | Colab Pro / Kaggle / single cloud GPU; or Hugging Face Jobs |
| Task 2 training | CPU only | Laptop / small cloud VM |
| Serving | CPU inference, 2 GB RAM | Cloud Run (scale-to-zero) or on-prem mini-PC at hub clinic |
| On-device | INT8-quantized ONNX/TFLite | Mid-range Android; no GPU required |
| Storage | DVC remote (GCS/S3) | Modest object storage |

> **Cost message for the proposal:** training is a *one-time, single-GPU* expense; inference is CPU/on-device. This is deliberately a low-recurring-cost architecture — a strength for sustainability in an LMIC health system.

---

## 6. Phased delivery plan (maps to funding milestones)

| Phase | Deliverable | Stack exercised | Funder milestone |
|---|---|---|---|
| **0. Setup** | Repro env, DVC data, MLflow | uv, DVC, MLflow | Infrastructure ready |
| **1. Task 1 baseline** | Classical + CNN classifier, honest held-out metric | timm, Lightning, scikit-image | First validated model |
| **2. Task 1 hardening** | QC gate, uncertainty, Grad-CAM, ONNX export | pytorch-grad-cam, ONNX | Deployable classifier |
| **3. Task 2 baseline** | XGBoost risk + SHAP + calibration | XGBoost, SHAP, sklearn | Validated risk model |
| **4. Task 2 survival** | Time-to-event model | scikit-survival | Lead-time prediction |
| **5. Integration** | FastAPI + client + offline path | FastAPI, Flask/PWA | Working prototype |
| **6. Validation** | External-site + prospective eval, fairness audit, drift dashboard | Evidently | Field-ready evidence |

Each phase is a fundable, independently-valuable milestone — de-risking the grant.

---

## 7. One-paragraph summary (for the proposal narrative)

> We will build two complementary models on a mainstream, reproducible Python stack. A **PyTorch + timm** transfer-learning classifier (MobileNetV3/EfficientNet-B0, chosen for on-device inference) will screen smartphone blood-smear images, exported via **ONNX** for offline use in CHPS zones. An **XGBoost / survival-analysis** model (with **SHAP** explanations and probability **calibration**) will estimate short-term vaso-occlusive-crisis risk from routine clinical and environmental data. Both are served behind a **FastAPI** backend with a lightweight, offline-capable client, and are engineered for robustness through patient-level external validation, calibration, uncertainty-based abstention, fairness auditing, and post-deployment **drift monitoring** — with full reproducibility via **MLflow** and **DVC**. Outputs are strictly decision-support, never autonomous diagnosis.
