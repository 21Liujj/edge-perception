# Stage E2: NanoOWL Zero-Shot Quality Freeze & Error Analysis

**Status: PASS** (2026-08-27)

## Deliverables

1. **FULL protocol verified**: reused  Stage B2 protocol verbatim (COCO val2017 FULL, 5000 images, prompt template `a {category_name}`, threshold 0.10, top-k 100). No new protocol designed.
2. **Manifest SHA256**: N/A (no manifest, full val2017)
3. **Engine SHA256**: `d3f723f97d5e466805105f8b25da83d898e215c09598fb2b41c06d5254d09037` (verified)
4. **Annotations SHA256**: `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` (verified)
5. **Historical FULL run evidence**: `results/algorithm_b2_nanoowl/full/` audited in place, not re-run
6. **Metrics** (from verified on-disk JSON):
   - AP: **0.258616**
   - AP50: **0.407885**
   - AP75: **0.272948**
   - APsmall: **0.095263**
   - APmedium: **0.264089**
   - APlarge: **0.442186**
7. **Best categories** (top 3 by AP): cat (0.6151), train (0.5922), airplane (0.5754)
8. **Worst categories** (bottom 3 by AP): mouse (0.0000), knife (0.0251), book (0.0419)
9. **Major failure modes**: (a) semantic confusion (car↔truck, chair↔couch, knife→fork/spoon, person→furniture), (b) small-object miss (area < 32²px), (c) crowded scene (co-occurs with small miss), (d) localization error (fp_iou50), (e) language ambiguity (evidenced via prompt sensitivity)
10. **Prompt sensitivity conclusion**: three templates (`{c}`, `a {c}`, `photo of a {c}`) localize target equally often (17-18/20 images), but bare noun produces more raw candidates and higher scores. Sample too small (5 categories, 20 images) to generalize.
11. **Docs**: `docs/nanoowl_quality.md`
12. **Evidence**: `results/stage_e2_nanoowl_quality/{full_metrics.json, category_metrics.csv, prompt_sensitivity.csv, prompt_sensitivity_raw.json, failure_cases.json, stage_summary.md}`, plus 15 pre-existing visualization images from Stage B2 (`results/algorithm_b2_nanoowl/full/visualizations/`)
13. **Commit**: see below
14. **Git status**: see below

## Final

**NANOOWL_ZERO_SHOT_QUALITY_FROZEN** — all metrics verified against historical FULL COCO run, category analysis completed, error/failure cases curated (small number per spec), prompt sensitivity mini study completed (20 images, 5 categories, 3 templates).
