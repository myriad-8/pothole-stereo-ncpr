"""Segmentation metrics, validation-based post-processing tuning, paired bootstrap."""
import numpy as np
import cv2
from src.datasets.pothole600 import IGNORE_INDEX

def postprocess(prob, thr=0.5, min_px=0):
    """Probability map -> binary mask; connected components smaller than min_px are removed.
    Note: connected components may merge touching defects."""
    pred = prob > thr
    if min_px > 0 and pred.any():
        n, lab, st, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
        pred = keep[lab]
    return pred


def seg_metrics(probs, masks, thr=0.5, min_px=0):
    """Global IoU/Dice/precision/recall (void pixels excluded) + per-image IoU (1.0 if both empty)."""
    tp = fp = fn = 0; per = []
    for p, m in zip(probs, masks):
        valid = m != IGNORE_INDEX
        pred = postprocess(p.astype(np.float32), thr, min_px) & valid
        gt = (m == 1) & valid
        t, f_p, f_n = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
        u = t + f_p + f_n
        per.append(t / u if u > 0 else 1.0)
        tp += t; fp += f_p; fn += f_n
    return dict(iou=tp / (tp + fp + fn + 1e-9), dice=2 * tp / (2 * tp + fp + fn + 1e-9),
                precision=tp / (tp + fp + 1e-9), recall=tp / (tp + fn + 1e-9),
                mean_image_iou=float(np.mean(per))), per


def tune_postprocess(val_probs, val_masks, thrs=(0.3, 0.4, 0.5, 0.6, 0.7), min_pxs=(0, 50, 100, 200, 400)):
    """Choose probability threshold and minimum region size on VALIDATION data only."""
    best = (-1.0, 0.5, 0)
    for t in thrs:
        for mp in min_pxs:
            iou = seg_metrics(val_probs, val_masks, t, mp)[0]["iou"]
            if iou > best[0]:
                best = (iou, t, mp)
    return dict(val_iou=best[0], thr=best[1], min_px=best[2])


def paired_bootstrap(a, b, n=10000, seed=0):
    """Paired bootstrap over test images: mean difference, 95% CI, wins, losses."""
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), int((d > 0).sum()), int((d < 0).sum())
