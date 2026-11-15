# Stereo Vision for Pothole Detection and Severity Estimation: Final Report

*Figures referenced below are in the same folder (`reports/`). All numbers are copied from the result files in this folder; anything that is an assumption is labelled as one (see Section 14).*

## 1. Introduction

Potholes degrade road safety and ride comfort. An RGB image shows where a pothole is, but not reliable physical quantities such as depth, area or volume. This project builds a reproducible prototype that (1) segments potholes with a SegFormer-B0 model, (2) estimates road depth from calibrated stereo, (3) fits a local road plane, and (4) reports pothole depth, footprint area, volume and a rule-based severity **only when the geometry supports it**. Unreliable measurements are reported as `uncertain` or `unavailable` and never receive a severity label.

## 2. Problem statement

Build a pipeline that segments potholes, estimates road-surface depth from stereo, reconstructs the local road surface, estimates area / depth / volume where reliable, and flags uncertain measurements instead of reporting unsupported severity. Segmentation and stereo geometry must be evaluated separately on appropriate data, and combined on a small manually reviewed subset.

## 3. Objectives

1. Establish a SegFormer-B0 baseline (RGB, cross-entropy) on Pothole-600.
2. Investigate a lightweight modification, Neighborhood-Contrast Phase Routing (NCPR), against that baseline under an identical recipe.
3. Evaluate stereo geometry (calibration, depth conversion, disparity estimation) separately on RSRD.
4. Combine both on a manually reviewed RSRD subset with measurement-quality checks.
5. Deliver code, results, a demo and documentation that allow the results to be reproduced.

## 4. Contributions

* **NCPR (proposed model).** Each 2×2 neighbourhood of a feature map is split into four spatial phases; local contrast (absolute deviation of each phase from the neighbourhood mean) drives learned softmax routing weights that combine the phases. The routed features are **added** to the pretrained stride-2 patch-embedding convolution in SegFormer stages 2-4. The branch is zero-initialised, so training starts from the baseline, and its new parameters (+57,836, +1.56%) use a 10x learning rate. NCPR is closest to learnable and adaptive polyphase sampling and to content-adaptive downsampling for segmentation; it differs by taking the routing scores from local contrast inside a pretrained SegFormer. An earlier version that *replaced* the convolution (`ncpr_v1`) is kept as a separate documented model.
* **Integrated segmentation + geometry pipeline** with measurement-quality checks, evaluated on a small reviewed RSRD subset.
* **Honest uncertainty handling:** unreliable geometry gets no severity.

## 5. Datasets and related work

| Dataset | Role | Evidence for which claims |
|---|---|---|
| **Pothole-600** | segmentation train / val / test (RGB and masks; transformed disparity is provided but not used) | all segmentation results (Section 8) |
| **RSRD** (Road Surface Reconstruction Dataset, test split, 300 frames) | stereo geometry: calibration, unit check, disparity estimation, integrated measurements | all geometry results (Sections 9-10) |
| CARD, RELLIS-3D | related datasets only | **no result in this report uses them** |

Pothole-600 was collected with a stereo camera and provides RGB, transformed disparity and pixel-wise masks; no license is stated on the dataset page (cite the authors' papers). RSRD provides rectified stereo pairs with LiDAR-derived reference disparity and depth (stored scaled by 256) and per-recording calibration; the dataset page reports a dataset-level alignment error of about 1.2 cm and a disparity error of about 0.6 px, and states no license.

**Related work.** On Pothole-600, the dataset authors used CNNs with attention modules on RGB and transformed-disparity inputs (Fan et al., ECCV Workshops 2020, and earlier disparity-transformation work). NCPR relates to learnable polyphase sampling (Rojas-Gomez et al., NeurIPS 2022; adaptive polyphase sampling by Chaman and Dokmanic), content-adaptive downsampling (Hesse et al., CVPR Workshops 2023), learned downsampling near boundaries (Marin et al., 2019) and AutoFocusFormer (CVPR 2023). Stereo estimation uses RAFT-Stereo (Lipson et al., 3DV 2021). I did not find prior work applying contrast-routed phase selection to SegFormer for pothole segmentation, but this is based on a short search and is **not a novelty claim**.

## 6. Dataset audit

### 6.1 Pothole-600 (`audit_pothole600.csv`, `audit_overlays.png`)

| Item | Result |
|---|---|
| File schema | `training/`, `validation/`, `testing/`, each with `rgb/`, `label/`, `tdisp/` (all `.png`) |
| Images per split | 240 train / 180 validation / 180 test; no missing and no empty masks |
| Resolution | all 600 images are 400×400 (they are resized to 512×512 for training, i.e. upsampled) |
| Annotation | pixel-wise binary masks; pixel values {0, 255}; no ignore (void) label found |
| Pothole pixel fraction | mean 6.75%, min 1.5%, max 75.9% |
| Duplicates | 1 exact-duplicate group; 13 near-duplicate pairs (16×16 thumbnail L1 < 0.01), **none across splits** (no leakage detected) |
| Split decision | the dataset's predefined train / validation / test split is used unchanged; thresholds are tuned on validation only |
| Calibration | not needed (RGB only) |

![Audit overlays](audit_overlays.png)

Limitations of the audit: whether the masks cover *all* visible potholes was checked only by looking at overlays and by the failure analysis (Section 8.5), which shows partially labelled scenes; unannotated defects are therefore not treated as reliable background.

### 6.2 RSRD (`rsrd_unit_check.csv`)

| Item | Result |
|---|---|
| Frames | 300 test frames with matching left / depth / disparity files; sequence ID = recording date: 20230317 (23), 20230321 (10), 20230406 (23), 20230408 (120), 20230409 (124) |
| Image size | 1920×1080 |
| Calibration (dataset toolkit) | fx = fy = 2024.5 px, cx = 1033.4, cy = 498.6 (20230317, 20230321); fx = fy = 2022.6 px, cx = 1037.4, cy = 500.8 (20230406-20230409); baseline 118.8 mm (stored in mm, converted to metres) |
| Depth / disparity files | 16-bit PNG, value / 256; valid pixel = value > 0 |
| Reference coverage | 4.06% of pixels per frame on average (per-frame range 2.4-4.9%): sparse LiDAR scan lines |
| Rectification | taken from the dataset (assumed rectified). Not verified independently; an estimated-disparity error below 1 px on average (Section 9) is consistent with this. |
| Duplicates / near-duplicates | **not checked** for RSRD |

## 7. Methodology

1. **Baseline:** stock SegFormer-B0 (`nvidia/segformer-b0-finetuned-ade-512-512`), two-class head, RGB input, pixel-wise cross-entropy.
2. **Proposed:** the same model with residual NCPR in stages 2-4 (Section 4). Identical data, augmentation, optimizer, epochs, loss and seeds.
3. **Training recipe (both models):** AdamW, lr 5e-5 (proposed model: new NCPR parameters at 5e-4), weight decay 0.01, batch size 8, 20 epochs, 512×512 input, moderate augmentation (horizontal flip, random 80-100% crop, brightness / contrast jitter), seeds 42 / 43 / 44. The checkpoint with the best validation IoU is kept.
4. **Post-processing:** probability threshold and minimum region size are chosen on the **validation** split only (grid: thresholds 0.3-0.7, minimum region 0-400 px), then applied once to the test split. Connected components may merge touching defects.
5. **Stereo:** disparity from a pretrained RAFT-Stereo model (official princeton-vl code, `raftstereo-middlebury.pth`, run at half resolution), depth `Z = f·B/d` with f in pixels and B in metres.
6. **Geometry:** a plane is fitted by RANSAC to the road ring (60 px) around each defect, excluding the defect; depth is measured relative to that plane (p95 of the distance below the plane); footprint area is computed on the fitted plane (pixel area `z²/(fx·fy·|n·r|)`); volume = mean depth × area (approximation).
7. **Quality rules and severity:** a measurement is `unavailable` if fewer than 10% of defect pixels have depth, the road ring has too few depth pixels, the plane fit fails or its RMS error exceeds 1 cm; it is `uncertain` if the p95 depth is below 3× the plane-fit noise; otherwise it is `ok` and severity is "low" (< 2 cm), "medium" (< 5 cm) or "high" (assumed thresholds). The geometry code is unit-tested on synthetic data (a 4 cm pothole in sparse noisy depth is measured as 4.04 cm, area within 1%; invalid depth yields no severity; a 2 mm defect in 4 mm noise is `uncertain`).

## 8. Segmentation experiments (Pothole-600 test split, 180 images)

### 8.1 Results (mean ± std over 3 seeds; `segmentation_summary.csv`, `segmentation_runs.csv`)

| Model | Params | IoU (thr 0.5) | Dice (thr 0.5) | IoU (val-tuned) | Dice (val-tuned) | Precision (val-tuned) | Recall (val-tuned) |
|---|---|---|---|---|---|---|---|
| Baseline SegFormer-B0 | 3,714,658 | 0.6158 ± 0.0110 | 0.7622 ± 0.0085 | 0.6150 ± 0.0124 | 0.7615 ± 0.0095 | 0.7736 ± 0.0211 | 0.7512 ± 0.0351 |
| **Proposed (+ NCPR)** | 3,772,494 | **0.6240 ± 0.0026** | **0.7685 ± 0.0020** | **0.6249 ± 0.0030** | **0.7692 ± 0.0023** | 0.7967 ± 0.0069 | 0.7436 ± 0.0092 |

Per seed (IoU, threshold 0.5 / val-tuned):

| Seed | Baseline | Proposed | Best epoch (baseline / proposed) |
|---|---|---|---|
| 42 | 0.6248 / 0.6248 | 0.6227 / 0.6242 | 18 / 19 |
| 43 | 0.6190 / 0.6190 | 0.6223 / 0.6223 | 11 / 20 |
| 44 | 0.6035 / 0.6011 | 0.6270 / 0.6282 | 15 / 18 |

Each training run takes about 5 minutes on a Kaggle T4 (5.0 min baseline, 5.1 min proposed). Forward time was 12.3 ms (baseline) and 12.8 ms (proposed) per 512×512 image; GPU timing is noisy and **no speed difference is claimed**.

![Training curves](training_curves.png)

### 8.2 Statistics (computed from the per-image test IoUs)

Averaging each image's IoU over the 3 seeds, the proposed model is higher by **+0.0098** (threshold 0.5) and **+0.0091** (val-tuned) mean per-image IoU, and better on 126 / 128 of 180 images (worse on 54 / 52). The 95% bootstrap interval for the val-tuned difference is [+0.005, +0.013] when images are treated as independent, and [+0.002, +0.015] when blocks of 10 adjacent images are resampled together (an approximate correction, because the test set contains near-duplicate consecutive frames); for the threshold-0.5 version the block interval is [-0.000, +0.018]. Excluding the 23 hardest images, the gap is similar (0.7410 vs 0.7515 mean per-image IoU on 157 images), so it is not caused by a few scenes. p-values are deliberately not reported because images are not independent.

### 8.3 Interpretation

The proposed model is slightly better on average (about +0.01 IoU), has much lower seed-to-seed variance (std 0.003 vs 0.011), and has higher precision with slightly lower recall. However the gain is **small and not conclusive**: it is driven mostly by seed 44, where the baseline was weakest, and seed 42 is a tie. A "routing without contrast" ablation (`proposed_nocontrast`, implemented but **not run**) would be needed to attribute the gain to the contrast signal, as opposed to the extra capacity or the higher learning rate of the new branch. The first NCPR version (`ncpr_v1`, replacing the pretrained convolution) was below the baseline in an earlier 3-seed experiment (IoU about 0.603 vs 0.616; those numbers are from the first notebook run and are not included in the result files). Validation IoU is noisy from epoch to epoch (e.g. proposed, seed 42: 0.55, 0.66, 0.58 in epochs 18-20), so best-epoch selection is partly luck.

### 8.4 Qualitative results

![Qualitative comparison](qualitative_segmentation.png)

Both models give smooth, blob-like boundaries compared with the ragged ground truth (SegFormer predicts at quarter resolution), and the two models' masks are usually very similar.

### 8.5 Failure analysis (`segmentation_failures.png`)

![Worst test images, proposed model](segmentation_failures.png)

* 21-25 of 180 test images have IoU < 0.5 for each run of either model; none has IoU = 0 (no pothole is completely missed).
* **The failures are concentrated in a few scenes.** 17 test images have mean IoU < 0.3 over the six runs: images 0050-0061 (a large dark oil / wet stain in which the ground truth marks only a small corner while both models segment the whole stain), 0115, 0118 (thin, small potholes that are under-segmented) and 0169-0171. Allowing IoU < 0.5 adds 0163-0172, giving 23 hard images in 4 groups of consecutive frames.
* **Cause by type:** (a) ambiguous or partial labels (the oil-stain scene), (b) small or thin defects, (c) boundary error (smooth predictions vs ragged labels). Because several failing images are near-identical consecutive frames of one scene, failures are correlated and the effective test size is smaller than 180.
* Connected components may merge touching defects; this was not measured separately.

### 8.6 Not done

The brief suggests comparing class-weighted cross-entropy and CE + Dice if defects occupy few pixels. Defects cover 6.75% of pixels on average; the code supports both (`CFG["loss"]`) but only plain cross-entropy was trained and evaluated.

## 9. Stereo geometry experiments (RSRD, evaluated separately from segmentation)

### 9.1 Unit / consistency check (`rsrd_unit_check.csv`)

Converting the reference disparity to depth with the calibrated focal length (pixels) and baseline (metres) and comparing it with the reference depth over 300 frames and 25,257,491 valid pixels gives **MAE 2.2 mm and RMSE 2.5 mm** (per-frame MAE 2.07-2.38 mm). This verifies units, focal length, baseline and the scale factor of 256. It is a **consistency check, not an accuracy test**: both reference maps come from the same fused LiDAR, so the agreement is expected.

### 9.2 Disparity estimation against the reference (`rsrd_stereo_estimation_eval.csv`, 8 frames)

Pretrained RAFT-Stereo (official weights, half resolution, upsampled): estimated-disparity coverage 100% of the reference-valid pixels, **EPE 0.88 px, bad-pixel rate (D1) 2.1%, depth MAE 11.5 cm, depth RMSE 20.3 cm**. Per frame: EPE 0.56-1.12 px; depth MAE 4.6-26.2 cm (worst frame 20230409020034.200).

For comparison, a first run with OpenCV SGBM (used as a fallback before the RAFT-Stereo weights were available; these numbers were printed in the notebook but are not saved in the result files) gave coverage 88.5%, EPE 0.68 px, D1 0.3%, depth MAE 9.0 cm and RMSE 44.6 cm. The two methods are **not directly comparable** because SGBM is scored only on the 88.5% of pixels it matched, whereas RAFT-Stereo is scored on all of them; RAFT-Stereo's depth RMSE is less than half of SGBM's, i.e. it makes far fewer large errors.

Absolute depth error from estimated disparity (about 10 cm at range) is larger than a typical pothole depth (2-5 cm). Measurements are therefore taken relative to a locally fitted plane and are rejected when not clearly above the fit noise. The reference itself has an alignment error of about 1.2 cm (dataset documentation).

## 10. Integrated pipeline on a manually reviewed RSRD subset

RSRD has no pothole masks. The proposed model produced masks on the 300 left images; 199 frames contained some candidate, which is far too many because of the domain shift from Pothole-600 (close-up road-surface images) to a forward-facing driving camera. A strict rule (probability > 0.8, area from 2,000 px up to 5% of the frame, centre in the lower 60% of the image) left 8 frames, which were reviewed by eye and all kept. **These masks are model-proposed and manually reviewed, not human-annotated.** Each defect was measured twice, with the sparse LiDAR reference disparity and with the RAFT-Stereo estimate (`integrated_predictions.csv` / `.jsonl`).

| Disparity source | ok | uncertain | unavailable |
|---|---|---|---|
| Reference (sparse LiDAR) | 0 | 0 | 8 (only 0.8-7.6% of defect pixels have depth; the gate is 10%) |
| Estimated (RAFT-Stereo) | **1** | 7 | 0 |

Estimated-disparity measurements:

| Frame | Seg. score | Depth p95 (mm) | Plane-fit error (mm) | Depth / noise | Area (m²) | Volume (L) | Status |
|---|---|---|---|---|---|---|---|
| 20230408032741.200 | 0.92 | **20.1** | 3.6 | 5.6 | 0.457 | 4.5 | **ok, severity "medium"** |
| 20230408042104.200 | 0.88 | 1.9 | 0.9 | 2.1 | 0.244 | 0.1 | uncertain |
| 20230408040538.600 | 0.84 | 6.2 | 4.4 | 1.4 | 0.014 | 0.1 | uncertain |
| 20230408032628.000 | 0.83 | 7.9 | 5.4 | 1.5 | 0.072 | 0.3 | uncertain |
| 20230409021821.400 | 0.82 | 0.6 | 1.1 | 0.5 | 0.006 | < 0.1 | uncertain |
| 20230408023832.000 | 0.81 | 0.5 | 1.2 | 0.4 | 0.017 | < 0.1 | uncertain |
| 20230408045315.200 | 0.81 | 1.7 | 1.0 | 1.7 | 0.008 | < 0.1 | uncertain |
| 20230408023906.200 | 0.81 | 4.0 | 2.2 | 1.8 | 0.167 | 0.2 | uncertain |

![Integrated examples](integrated_examples_estimated_depth.png)

The reference-depth panels in `integrated_examples.png` are nearly blank because the LiDAR reference consists of thin scan lines; the figure above shows the estimated depth with the mask outlined.

**What this shows.**
* One defect (depth about 2.0 cm, area 0.46 m², about 4.5 L) was measured; its depth sits on the 2 cm boundary of the assumed thresholds, so the label "medium" should be read as "about 2 cm, borderline low/medium".
* Seven defects were flagged `uncertain` because their estimated depth (0.5-8 mm) is not clearly above the plane-fit noise (0.9-5.4 mm); they receive **no severity**.
* The tree-shadow frame (20230408042104.200) is a segmentation false positive; the geometry measures 1.9 mm, i.e. no depression, so the quality check correctly rejects it.
* In frame 20230408040538.600 the mask covers only a small part of a real defect, so its depth is underestimated (a mask-quality limitation, not a stereo error).
* **This is a demonstration of the pipeline and its quality checks. Depth and severity accuracy are not validated** (8 frames, no ground-truth pothole depths).

## 11. Demo

`demo/app.py` (Gradio) accepts a single road image, an image plus a metric depth map (`.npy`, metres) with calibration, or a calibrated rectified stereo pair (left, right, fx, fy, cx, cy, baseline) and shows the mask overlay, numeric measurements per defect, and a concise reason whenever a measurement is unavailable. Behaviour was exercised offline with stubbed models on seven cases: image only (no calibration), calibration without depth, depth with invalid pixels inside the defect, depth of the wrong size, a valid depth map (a 4 cm synthetic pothole is reported as 4.0 cm, "medium"), a stereo pair with baseline, and an image with no detection. In every unreliable case the demo prints "measurement UNAVAILABLE" with the reason and assigns **no severity**. The demo can be launched from the notebook (`LAUNCH_DEMO = True`) or with `python demo/app.py --ckpt checkpoints/proposed_ce_s42.pth --model proposed`. *(Add screenshots here if you have them: `reports/demo_*.png`.)*

## 12. Limitations and failure summary

* The proposed model's gain is small (about +0.01 IoU), not conclusive across 3 seeds, and the contrast ablation was not run.
* Segmentation failures are concentrated in a few scenes with ambiguous or partial labels, and in small / thin defects; boundaries are smoother than the ground truth.
* Pothole-600 is a close-up road-surface dataset; transfer to the forward-facing RSRD camera is poor (199 of 300 frames had candidates), which required a strict rule and manual review.
* RSRD reference depth covers about 4% of pixels and has about 1.2 cm alignment error; the reference-vs-reference unit check is circular.
* Estimated stereo depth has about 10 cm MAE at range, larger than pothole depth; stereo was evaluated on 8 pairs only, at half resolution.
* Only 8 RSRD frames and 1 successful measurement; no ground-truth pothole depth, so depth accuracy and severity are not validated.
* RSRD near-duplicates were not checked; the stereo-estimation numbers of the first (SGBM) run are not in the result files.
* The class-weighted / CE + Dice comparison suggested in the brief was not run.
* A fresh-environment reproduction of inference from the README has not been independently tested; the geometry unit test (`python -m src.geometry.test_synthetic`) passes.

## 13. Reproducibility

Code and results: https://github.com/myriad-8/pothole-stereo-ncpr. The full workflow is `notebooks/stereo_pothole_main.ipynb` (Kaggle, GPU T4, saved outputs), which calls the same code that is in `src/`. Seeds, configuration (`config.json`, `configs/default.yaml`), package versions (Python 3.13, torch 2.11.0+cu130, torchvision 0.26.0, transformers 5.20.2; `environment.json`, `requirements_frozen.txt`) and the sample manifest (`data/manifest.csv`) are recorded. Checkpoints are not in git; they are hosted at https://huggingface.co/Sahithiii437/pothole-segformer-ncpr. Datasets are not redistributed; check each dataset's and pretrained model's terms before redistributing.

## 14. Measured results vs assumptions

| Measured (from the result files) | Assumed / design choice (not independently validated) |
|---|---|
| Segmentation metrics per model and seed on the Pothole-600 test split; parameter counts; timing | Severity thresholds: below 2 cm low, below 5 cm medium, otherwise high |
| RSRD unit / consistency check; disparity-estimation error vs reference on 8 pairs | Quality gates: at least 10% valid depth inside a defect, plane-fit RMS error at most 1 cm, depth at least 3× the fit noise |
| Integrated measurements on 8 reviewed frames (reference and estimated disparity) | Volume = mean depth × footprint area (approximation) |
| Validation-tuned threshold and minimum region size (chosen per run) | RSRD masks: model-proposed and manually reviewed; segmentation score is not a measurement confidence |

## Deliverables checklist against the project brief

| Required deliverable | Where | Status |
|---|---|---|
| Dataset audit with overlays, calibration checks, file schemas, split decisions | Section 6, `audit_*`, notebook | done (RSRD near-duplicate check not done) |
| Reproducible segmentation code, saved model, validation curves, held-out metrics | `src/`, notebook, checkpoints (Hugging Face), `training_curves.png`, Section 8 | done |
| Stereo inference and reconstruction with numeric outputs, unit checks, reference-based evaluation | `src/stereo`, `src/geometry`, Section 9 | done (8 pairs) |
| Combined pipeline on a reviewed subset with masks, fitted surfaces, measurements, severity rules, uncertain cases | Section 10, `integrated_predictions.*`, `rsrd_masks/` | done (8 frames) |
| Streamlit / Gradio demo with overlays, measurements and reasons | `demo/app.py`, Section 11 | done (screenshots optional) |
| README and final report with comparative results and failure examples | `README.md`, this report | done |

## References

* Fan, R., Ai, X., Dahnoun, N. Road surface 3D reconstruction based on dense subpixel disparity map estimation. IEEE TIP 27(6), 2018.
* Fan, R., Ozgunalp, U., Hosking, B., Liu, M., Pitas, I. Pothole detection based on disparity transformation and road surface modeling. IEEE TIP 29, 2019.
* Fan, R., Wang, H., Wang, Y., Liu, M., Pitas, I. et al. We learn better road pothole detection: from attention aggregation to adversarial domain adaptation. ECCV Workshops, 2020.
* Zhao, T., Xie, Y., Ding, M., Yang, L., Tomizuka, M., Wei, Y. A road surface reconstruction dataset for autonomous driving. Scientific Data 11, 2024.
* Lipson, L., Teed, Z., Deng, J. RAFT-Stereo: Multilevel recurrent field transforms for stereo matching. 3DV 2021.
* Xie, E. et al. SegFormer: Simple and efficient design for semantic segmentation with transformers. NeurIPS 2021.
* Rojas-Gomez, R. et al. Learnable polyphase sampling for shift-invariant and equivariant convolutional networks. NeurIPS 2022.
* Hesse, R., Schaub-Meyer, S., Roth, S. Content-adaptive downsampling in convolutional neural networks. CVPR Workshops 2023.
* Marin, D. et al. Efficient segmentation: learning downsampling near semantic boundaries. ICCV 2019.

*(Verify the author lists and venues of the references before submitting.)*
