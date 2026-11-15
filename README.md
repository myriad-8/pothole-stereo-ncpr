# Stereo Vision for Pothole Detection and Severity Estimation

A reproducible prototype that (1) segments potholes from road images, (2) estimates road-surface depth from calibrated stereo, (3) fits a local road plane, and (4) reports pothole depth, footprint area, volume and a rule-based severity **only when the geometry is reliable**. Unreliable cases are reported as `uncertain` / `unavailable` and never receive a severity label.

* **Baseline:** stock SegFormer-B0 (Hugging Face), RGB input, cross-entropy.
* **Proposed:** SegFormer-B0 + **NCPR** (Neighborhood-Contrast Phase Routing): a zero-initialised, contrast-guided downsampling branch added to the pretrained stride-2 patch-embedding convolutions of stages 2-4 (+57,836 parameters, +1.56%).
* **Segmentation** is evaluated on Pothole-600; **stereo geometry** is evaluated separately on RSRD; the two are combined on a small manually reviewed RSRD subset.

## Key results

All numbers are measured outputs of the notebook (files in `reports/`). Severity thresholds are **assumed**, not validated.

### Segmentation: Pothole-600 test split (180 images), mean ± std over 3 seeds (42, 43, 44)

| Model | Params | IoU (thr 0.5) | Dice (thr 0.5) | IoU (val-tuned) | Dice (val-tuned) | Precision / Recall (val-tuned) |
|---|---|---|---|---|---|---|
| Baseline SegFormer-B0 | 3,714,658 | 0.616 ± 0.011 | 0.762 ± 0.009 | 0.615 ± 0.012 | 0.762 ± 0.010 | 0.774 / 0.751 |
| **Proposed (SegFormer-B0 + NCPR)** | 3,772,494 | **0.624 ± 0.003** | **0.769 ± 0.002** | **0.625 ± 0.003** | **0.769 ± 0.002** | 0.797 / 0.744 |

*Val-tuned = probability threshold and minimum region size chosen on the validation split only.*

How to read this: the proposed model is slightly better on average and much more stable across seeds, with higher precision and slightly lower recall. The gain is **small (about +0.01 IoU) and not conclusive**: it is driven mostly by one seed (seed 42 is a tie), and the "NCPR without contrast" ablation (`proposed_nocontrast`) has not been run. Averaged over seeds, per-image IoU improves by +0.009 (95% CI from a block bootstrap over adjacent images: about +0.002 to +0.015; the test set contains near-duplicate frames, so images are not independent). Training takes about 5 minutes per run on a Kaggle T4 for both models.

### Stereo geometry: RSRD test split (300 frames)

| Check | Result |
|---|---|
| Unit / consistency check (reference disparity -> depth vs reference depth; validates focal length in pixels, baseline in metres, scale 256) | MAE 2.2 mm, RMSE 2.5 mm over 25,257,491 valid pixels. *This is a consistency check, not stereo accuracy: both references come from the same fused LiDAR.* |
| Reference depth coverage | 4.06% of pixels per frame (sparse LiDAR scan lines) |
| Stereo estimation vs reference, 8 pairs (RAFT-Stereo, pretrained `raftstereo-middlebury.pth`, run at half resolution) | coverage 100%, EPE 0.88 px, bad-pixel rate (D1) 2.1%, depth MAE 11.5 cm, depth RMSE 20.3 cm |

Estimated-depth error (about 10 cm at range) is larger than a typical pothole depth (2-5 cm), so absolute depth from estimated disparity is not reliable on its own; measurements are taken **relative to a locally fitted road plane** and are rejected when they are not clearly above the plane-fit noise.

### Integrated pipeline: 8 manually reviewed RSRD frames / 8 defects

| Disparity source | ok | uncertain | unavailable |
|---|---|---|---|
| Reference (sparse LiDAR) | 0 | 0 | 8 (only 0.8-7.5% of defect pixels have depth; gate is 10%) |
| Estimated (RAFT-Stereo) | 1 | 7 (depth within plane-fit noise) | 0 |

The single measured defect (`20230408032741.200`): depth p95 about 2.0 cm, footprint area 0.46 m², volume about 4.5 L, severity "medium" (on the 2 cm boundary of the assumed thresholds). The other seven received **no severity**. One uncertain case (tree-shadow frame `20230408042104.200`, 1.9 mm) is a segmentation false positive that the geometry correctly rejects. With 8 frames and no ground-truth pothole depths, this demonstrates the pipeline and its quality checks; it does **not** validate depth accuracy.

## Repository layout

```
notebooks/stereo_pothole_main.ipynb   full workflow (run on Kaggle), outputs saved
src/models/        baseline.py, ncpr_v1.py, proposed.py  (one file per model)
src/datasets/      Pothole-600 loader
src/segmentation/  training loop, full-resolution inference
src/stereo/        disparity estimation (RAFT-Stereo, SGBM fallback), disparity -> depth
src/geometry/      road-plane fit, measurements, severity, synthetic unit test
src/evaluation/    metrics, validation-based post-processing tuning, bootstrap
configs/           default.yaml
data/              manifest.csv (common sample manifest), notes
demo/app.py        Gradio demo
reports/           result tables, figures, predictions, environment (see below)
checkpoints/       NOT versioned (see "Model checkpoints")
```

`reports/` contains: `segmentation_summary.csv`, `segmentation_runs.csv`, `segmentation_results.json`, `training_curves.png`, `qualitative_segmentation.png`, `segmentation_failures.png`, `audit_pothole600.csv`, `audit_overlays.png`, `rsrd_unit_check.csv`, `rsrd_stereo_estimation_eval.csv`, `integrated_predictions.csv/.jsonl`, `integrated_examples*.png`, `rsrd_masks/`, `config.json`, `environment.json`, `requirements_frozen.txt`.

## Setup

```bash
pip install -r requirements.txt
python -m src.geometry.test_synthetic      # geometry unit test; must print PASS
```

Versions used for the reported results (Kaggle, Tesla T4): Python 3.13, torch 2.11.0+cu130, torchvision 0.26.0, transformers 5.20.2 (full list in `reports/requirements_frozen.txt` and `reports/environment.json`).

## Data access

| Dataset | Use | Source | Terms |
|---|---|---|---|
| Pothole-600 | segmentation train / val / test (240 / 180 / 180 images, no missing or empty masks) | https://sites.google.com/view/pothole-600/dataset , Kaggle `rangerfan/pothole-600` | No license is stated on the dataset page (publicly available for research; the authors ask users to cite their papers). Check the Kaggle dataset page before redistributing; the data is **not** included in this repo. |
| RSRD (Road Surface Reconstruction Dataset) | stereo geometry (test split, 300 samples, plus calibration from the dev toolkit) | https://thu-rsxd.com/rsrd/ , dev kit https://github.com/ztsrxh/RSRD_dev_toolkit | No license is stated on the dataset page. Check the download page terms before redistributing; the data is **not** included in this repo. |
| CARD, RELLIS-3D | related datasets only | listed in the project brief | **Not used** to produce any result. |

Version note: the notebook downloads the RSRD test split from the link in the notebook (figshare) at run time; record the download date when you reproduce.

## Reproduce

1. **Kaggle notebook** (GPU T4, Internet on, add the `rangerfan/pothole-600` dataset). Upload `notebooks/stereo_pothole_main.ipynb` and run it top to bottom. Training resumes automatically; choose models with `CFG["models_to_run"]` (`baseline`, `proposed`, `ncpr_v1`, `proposed_nocontrast`). Seeds, epochs, loss and learning rate are in `CFG` and in `configs/default.yaml` (20 epochs, AdamW, lr 5e-5, batch 8, 512 px, cross-entropy; the proposed model trains its new NCPR parameters at 10x the learning rate).
2. **Stereo model:** the notebook clones the official RAFT-Stereo repository and downloads its pretrained weights with the authors' script. If that fails it falls back to OpenCV SGBM and prints the reason; the method actually used is recorded in `integrated_predictions.csv` (`stereo_method`).
3. **Manual review step:** frames with real potholes are chosen by eye from a strict candidate grid (`ACCEPTED_IDS`). The RSRD masks are model-proposed and manually reviewed, not human-annotated.

## Inference examples

Segmentation of one image:

```python
import cv2, torch
from src.models import REGISTRY
from src.segmentation.infer import predict_prob_fullres

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = REGISTRY["proposed"]("nvidia/segformer-b0-finetuned-ade-512-512").to(dev)
model.load_state_dict(torch.load("checkpoints/proposed_ce_s42.pth", map_location=dev)); model.eval()
img = cv2.cvtColor(cv2.imread("road.jpg"), cv2.COLOR_BGR2RGB)
mask = predict_prob_fullres(model, img, dev) > 0.5          # pothole mask at the original resolution
```

Measurement from a rectified stereo pair with calibration (fx, fy, cx, cy in pixels; baseline in metres):

```python
import numpy as np
from src.stereo.disparity import estimate_disparity, disparity_to_depth
from src.geometry.surface import measure_defect

disp, method = estimate_disparity(left_rgb, right_rgb, dev)
depth = disparity_to_depth(disp, fx, baseline_m)
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
result = measure_defect(depth, defect_mask, K)               # dict; severity is None unless measurement_status == "ok"
```

Demo (overlay, numeric measurements, and the reason when a measurement is unavailable; accepts a single image, an image plus metric depth `.npy`, or a calibrated stereo pair):

```bash
python demo/app.py --ckpt checkpoints/proposed_ce_s42.pth --model proposed
```

## Prediction record

`reports/integrated_predictions.csv` / `.jsonl` columns: `dataset, sample_id, defect_id, mask_path, detection_score, depth_p95_m, area_m2, volume_m3, severity, measurement_status, valid_geometry_fraction, surface_fit_error_m, reason, disparity_source, stereo_method`. Unavailable numeric values are `null` in the JSONL. `measurement_status` is `ok`, `uncertain` or `unavailable`. `detection_score` is the mean segmentation probability and is **not** a confidence for the depth measurement. Read the CSV with `pd.read_csv(path, dtype={"sample_id": str})`, because frame IDs such as `20230408032628.000` otherwise lose their trailing zeros.

## Model checkpoints

Checkpoints are not stored in git. Download `proposed_ce_s42.pth` and `baseline_ce_s42.pth` from: **[Hugging Face model page](https://huggingface.co/Sahithiii437/pothole-segformer-ncpr)** and place them in `checkpoints/`. Pretrained sources: SegFormer-B0 `nvidia/segformer-b0-finetuned-ade-512-512` (Hugging Face); RAFT-Stereo https://github.com/princeton-vl/RAFT-Stereo (`raftstereo-middlebury.pth`). Check the terms of each pretrained model before redistributing.

## Known limitations

* **NCPR is preliminary.** The gain over the baseline is small (about +0.01 IoU), not conclusive across 3 seeds, and the contrast-ablation has not been run. An earlier version that replaced the convolution (`ncpr_v1`) was lower than the baseline in a first 3-seed experiment (IoU about 0.603 vs 0.616) and is kept only as a documented variant.
* **Validation IoU is noisy** from epoch to epoch (swings of 0.05-0.10), so best-epoch selection is partly luck.
* **Label ambiguity and near-duplicates.** The hardest test images (17 with IoU < 0.3 in every run) form a few consecutive scenes, mainly a large dark oil/wet stain where the ground truth marks only a small part. Boundaries are smooth compared with the ragged ground truth. Connected components may merge touching defects.
* **Domain shift.** Pothole-600 consists of close-up road-surface images; RSRD is a forward-facing driving camera. Many RSRD proposals are false positives (shadows, cracks, markings), hence the strict candidate rule and the manual review. In one reviewed frame the mask covers only part of a real defect, so its depth is underestimated.
* **Sparse reference depth.** RSRD reference depth covers about 4% of pixels; the dataset's own alignment error is about 1.2 cm and its disparity error about 0.6 px, comparable to shallow potholes. The reference-vs-reference unit check is circular and says nothing about stereo accuracy.
* **Stereo error.** Estimated depth has about 10 cm MAE at range; only measurements clearly above the plane-fit noise are accepted. Pothole depth accuracy is **not validated** (8 frames, no ground-truth depths).
* **Assumed thresholds.** Severity (below 2 cm low, below 5 cm medium, otherwise high), quality gates (valid-depth fraction, plane-fit error, depth-to-noise ratio of 3) and the volume approximation (mean depth times footprint area) are design choices, not standards.
* RAFT-Stereo runs at half resolution for memory reasons; RSRD frames are assumed rectified as provided by the dataset.
* CARD and RELLIS-3D are not evaluated.

## Citations

* Fan, R., Ai, X., Dahnoun, N. *Road surface 3D reconstruction based on dense subpixel disparity map estimation.* IEEE TIP 27(6), 2018.
* Fan, R., Ozgunalp, U., Hosking, B., Liu, M., Pitas, I. *Pothole detection based on disparity transformation and road surface modeling.* IEEE TIP 29, 2019.
* Fan, R., Liu, M. *Road damage detection based on unsupervised disparity map segmentation.* IEEE T-ITS, 2019.
* Fan, R. et al. *We learn better road pothole detection: from attention aggregation to adversarial domain adaptation.* ECCV Workshops, 2020.
* Zhao, T., Xie, Y., Ding, M., Yang, L., Tomizuka, M., Wei, Y. *A road surface reconstruction dataset for autonomous driving.* Scientific Data 11, 2024. doi:10.1038/s41597-024-03261-9.
* Lipson, L., Teed, Z., Deng, J. *RAFT-Stereo: Multilevel recurrent field transforms for stereo matching.* 3DV 2021.
* Xie, E. et al. *SegFormer: Simple and efficient design for semantic segmentation with transformers.* NeurIPS 2021.
