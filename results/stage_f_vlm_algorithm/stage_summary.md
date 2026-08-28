# Stage F — VLM Algorithm Deep Dive: Summary

**Date**: 2026-08-28  
**Status**: VLM_ALGORITHM_UNDERSTOOD_AND_VALIDATED  
**Scope**: Algorithm analysis + small controlled experiments, NOT new deployment/training/large model work

---

## Completed Deliverables

### 1. Algorithm Call Graph
**File**: `results/stage_f_vlm_algorithm/algorithm_callgraph.json`

Traced the actual NanoOWL (OWL-ViT) algorithm pipeline from source (`third_party/nanoowl/nanoowl/owl_predictor.py`, verified on-device 2026-08-28):
- **Image branch**: 7 steps (vision_model → CLS*patch fusion → LayerNorm → box_head/class_head/logit_shift/logit_scale)
- **Text branch**: 3 steps (tokenizer → text_model → text_projection)
- Verified tensor shapes: image_class_embeds [B,576,512], text_embeds [N,512], pred_boxes [B,576,4], logit_shift/scale [B,576,1]
- Confirmed: 576 patches (24×24 grid from 768×768 image / 32×32 patches), embedding_dim=512

### 2. Matching Equation
**File**: `results/stage_f_vlm_algorithm/matching_equation.json`

Derived actual cross-modal scoring equation from `OwlPredictor.decode()` source (lines 254-284):
- **L2 normalization**: both image_class_embeds and text_embeds normalized to unit hypersphere (epsilon=1e-6)
- **Cosine similarity**: `torch.einsum("...pd,...qd->...pq", ...)` computes dot product of normalized vectors → cos(θ)
- **Per-patch shift + scale**: `logit = (cos_sim + logit_shift) * logit_scale`
- **Sigmoid**: `score = σ(logit)` per (batch, patch, text_query), multi-label not softmax
- **Final equation**: score_{b,p,n} = σ( τ_{b,p} · (cos(v_{b,p}, t_n) + s_{b,p}) )

Verified: `text_projection` has NO bias (Linear(512→512, bias=False), confirmed from live model)

### 3. Text Embedding Geometry
**Script**: `scripts/vlm_analysis/analyze_text_embeddings.py`  
**Results**: `results/stage_f_vlm_algorithm/text_embedding_similarity.csv`, `text_embedding_summary.json`

5 categories × 4 prompt templates = 20 prompts. Extracted real text embeddings via NanoOWL, computed cosine similarity matrix:
- **Within-category mean similarity**: 0.734 (not 0.99+ → moderate prompt sensitivity exists)
- **Cross-category mean similarity**: 0.163 (good discrimination)
- Per-category: "person" and "car" have high intra-similarity (0.78-0.79), "hair drier" and "chair" more sensitive (0.66-0.68)

### 4. Prompt Sensitivity Experiment
**Script**: `scripts/vlm_analysis/evaluate_prompt_variants.py`  
**Results**: `results/stage_f_vlm_algorithm/prompt_sensitivity.csv`, `prompt_sensitivity_summary.json`

Fixed 50 COCO instances (E4 manifest), varied ONLY text prompt across 4 variants (base, "a <cat>", "photo of a <cat>", synonym). Measured detection_success, score, box_iou, mask_iou:
- **Templates hurt recall**: "a <cat>" lost net 4 detections, "photo of..." lost net 3 vs base
- **Localization unchanged among successes**: mean box_iou ~0.50-0.51 across templates
- **Synonyms dangerous**: "human"/"automobile"/"seat" showed catastrophic box_iou collapse (0.170 vs 0.513 base) despite similar detection_success_rate
- **Conclusion**: Impact is mainly on CLASSIFICATION CONFIDENCE (threshold crossing), not localization — except synonyms which also mislocalize.

### 5. Prompt Ensemble
**Script**: `scripts/vlm_analysis/evaluate_prompt_ensemble.py`  
**Results**: `results/stage_f_vlm_algorithm/prompt_ensemble.csv`, `prompt_ensemble_summary.json`

Mean+renormalize of 3 templates (`<cat>`, `a <cat>`, `photo of a <cat>`) vs single-prompt base:
- **Single**: 0.84 detection_success, 0.513 box_iou, 0.457 mask_iou
- **Ensemble**: 0.82 detection_success, 0.504 box_iou, 0.449 mask_iou
- **Delta**: -0.02 / -0.009 / -0.009 (all worse)
- **Decision**: **PROMPT_ENSEMBLE_NO_CLEAR_GAIN**

### 6. Grounding Error Propagation + Correlation
**Script**: `scripts/vlm_analysis/analyze_grounding_propagation.py`  
**Results**: `results/stage_f_vlm_algorithm/grounding_propagation.csv`, `grounding_correlation.json`

Re-analyzed E4's existing 50-instance data (no new GPU inference). Computed degradation, Pearson/Spearman correlation, bucketed by box_iou:
- **box_iou vs mask_iou**: Pearson r=0.9306, Spearman ρ=0.8566 (both p<0.0001)
- **box_iou vs degradation**: Pearson r=-0.8712, Spearman ρ=-0.7986 (negative as expected)
- **Bucketed**: box_iou [0,0.25) → mean mask_iou 0.0047 (near-zero), box_iou [0.75,1.0] → 0.7661
- **Conclusion**: BBox localization quality is PRIMARY driver of E2E mask quality.

### 7. Failure Taxonomy (Refined)
**File**: `results/stage_f_vlm_algorithm/failure_taxonomy.json`

Documented E4's pre-existing thresholds (NOT tuned in Stage F) with algorithmic rationale:
- **A_detection_failure** (8/50): score ≤ 0.10 at all patches → classification/semantic-grounding failure
- **C_segmentation_failure** (3/50): mask_iou_gtbox < 0.5 → NanoSAM's own limitation (independent of NanoOWL)
- **B_localization_propagation** (17/50): box_iou < 0.5 OR degradation > 0.15 → bbox imprecision hurt mask
- **OK** (22/50): both stages performed acceptably

### 8. VLM Algorithm Deep Dive Document
**File**: `docs/vlm_algorithm_deep_dive.md` (16 sections, 20 interview Q&A)

Comprehensive technical reference covering:
1. What is Vision-Language Modeling
2. CLIP-style contrastive alignment (why normalize, why temperature)
3. CLIP → OWL-ViT (global embedding → per-patch, detection heads)
4. OWL-ViT image branch (7-step pipeline, verified tensor shapes/config)
5. OWL-ViT text branch (3-step pipeline, cacheability evidence from E3)
6. Cross-modal matching (actual equation derived from source)
7. Open-vocabulary mechanism (why it works, its limits)
8. Bounding box regression (box_head architecture, box_bias anchor)
9. Grounding error propagation to NanoSAM (quantitative findings)
10. Text embedding geometry (Stage F experiment findings)
11. Prompt sensitivity (Stage F experiment findings)
12. Prompt ensemble (PROMPT_ENSEMBLE_NO_CLEAR_GAIN)
13. **Current algorithmic bottleneck ranking** (BBox localization > Detection recall > Prompt sensitivity > NanoSAM segmentation)
14. Why NanoOWL is VLM but NanoSAM is not
15. VLM vs Multimodal LLM (NanoOWL vs Qwen-VL conceptual difference)
16. 20 interview questions covering all key concepts

All explanations grounded in THIS PROJECT'S actual implementation (third_party/nanoowl source, E1-E5 evidence, Stage F experiments), NOT generic knowledge.

---

## Key Findings

### Algorithm Understanding (verified from source)
- OWL-ViT's CLS*patch fusion (element-wise multiply) injects global context into 576 local representations
- Per-patch logit_shift and logit_scale (576 learned scalars each) replace CLIP's single global temperature
- Text embeddings are cacheable (72% overhead saved when cached, per E3)
- No explicit background class in the matching equation (low scores implicitly = "not here")

### Prompt Sensitivity
- Within-category prompt similarity 0.734 (moderate sensitivity, not robustly identical)
- Templates ("a <cat>", "photo of...") hurt detection recall by 4-5 cases out of 50
- Synonyms cause localization collapse (box_iou 0.170 vs 0.513) — AVOID
- Prompt ensemble did NOT help

### Grounding Error Propagation (PRIMARY FINDING)
- **Pearson r=0.9306** between box_iou and end-to-end mask_iou
- Box_iou [0,0.25) → mask_iou 0.0047 (catastrophic failure)
- Box_iou [0.75,1.0] → mask_iou 0.7661 (near GT-box ceiling)
- **17/50 cases** classified as B_localization_propagation (vs 8 detection failures, 3 segmentation failures)

### Algorithmic Bottleneck Ranking
1. **BBox localization** (17/50 cases, correlation r=0.93) — HIGHEST SEVERITY
2. **Detection recall** (8/50 cases, 16% failure rate) — MODERATE
3. **Prompt sensitivity** (4-5 cases lost to templates) — LOW-MODERATE
4. **NanoSAM segmentation** (3/50 cases, conditional on GT box) — LOWEST

---

## Scope Adherence

**Stage F DID**:
- Source code analysis (third_party/nanoowl, verified from actual deployed code)
- Small controlled algorithm experiments (text embeddings, prompt variants, prompt ensemble, grounding correlation)
- Algorithm documentation (VLM deep dive, interview prep)

**Stage F did NOT** (per scope control):
- Retrain NanoOWL or deploy Grounding DINO (explicitly prohibited)
- Modify third_party/nanoowl or third_party/nanosam source (project-side scripts only)
- New large-scale training, QAT, INT8 rework, ROS2, camera, C++, CUDA kernels (all out of scope)
- 5090 training work (Jetson-only, analysis-only stage)

---

## Evidence Provenance

All findings traceable to:
- **Source code**: third_party/nanoowl/nanoowl/owl_predictor.py (read 2026-08-28)
- **Live model**: google/owlvit-base-patch32 loaded on-device, config/weights verified
- **E1-E5 frozen evidence**: results/stage_e2_nanoowl_quality/, stage_e3_nanoowl_benchmark/, stage_e4_joint_pipeline/, stage_e5_final/
- **Stage F experiments**: results/stage_f_vlm_algorithm/ (all new CSVs, JSONs generated 2026-08-28, reproducible via scripts/vlm_analysis/)

---

## Git Status

All docs, scripts, and small evidence files committed. Large untracked artifacts (models/*.engine, data/calibration/) remain on disk per .gitignore (legitimate evidence referenced by committed docs, NOT Stage F-generated cruft).

**Commit**: `analysis: deepen NanoOWL vision-language algorithm study`  
**Tracked working tree**: CLEAN
