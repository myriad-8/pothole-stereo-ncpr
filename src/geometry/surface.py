"""Road-plane fitting and pothole measurements (metres). Unavailable values stay None."""
import numpy as np
import cv2

# ---- Road-plane fitting and pothole measurement (all outputs in metres) ----
SEVERITY_BINS_M = [(0.02, "low"), (0.05, "medium"), (float("inf"), "high")]   # ASSUMED thresholds, not a standard


def backproject(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = (u - K[0, 2]) * depth / K[0, 0]
    Y = (v - K[1, 2]) * depth / K[1, 1]
    return np.stack([X, Y, depth], axis=-1)


def fit_plane_ransac(pts, iters=300, thresh=0.01, seed=0):
    """Plane n.p + d = 0 with |n| = 1, oriented so the camera (origin) has d > 0."""
    if len(pts) < 100:
        return None
    rng = np.random.default_rng(seed)
    best_inl, best_n = None, -1
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn_ = np.linalg.norm(n)
        if nn_ < 1e-9:
            continue
        n = n / nn_
        inl = np.abs(pts @ n - n @ s[0]) < thresh
        if inl.sum() > best_n:
            best_n, best_inl = inl.sum(), inl
    if best_inl is None:
        return None
    P = pts[best_inl]
    c = P.mean(0)
    n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    d = -n @ c
    if d < 0:
        n, d = -n, -d
    return n, d, best_inl


def severity_from_depth(depth_p95_m):
    for limit, name in SEVERITY_BINS_M:
        if depth_p95_m < limit:
            return name


def measure_defect(depth, mask, K, ring_px=60, min_valid_frac=0.1, min_mask_pts=30,
                   min_road_pts=500, max_fit_err_m=0.01, min_depth_snr=3.0):
    """depth: HxW metres (0/NaN = invalid, may be sparse). mask: HxW bool (one connected defect).
    Depth is measured relative to a plane fitted to the road ring around the defect.
    Returns a dict; if anything is unreliable the severity stays None (never invented)."""
    out = dict(depth_p95_m=None, area_m2=None, volume_m3=None, severity=None,
               measurement_status="unavailable", reason="",
               valid_geometry_fraction=None, surface_fit_error_m=None)
    mask = mask.astype(bool)
    if mask.sum() == 0:
        out["reason"] = "empty mask"; return out
    valid = np.isfinite(depth) & (depth > 0)
    nvalid = int((valid & mask).sum())
    out["valid_geometry_fraction"] = float(nvalid / mask.sum())
    if nvalid < min_mask_pts or out["valid_geometry_fraction"] < min_valid_frac:
        out["reason"] = "too few valid depth pixels inside defect"; return out

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    near = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    skip = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    road = near & ~skip & valid
    if road.sum() < min_road_pts:
        out["reason"] = "not enough road depth pixels around defect"; return out

    P = backproject(np.where(valid, depth, 0.0), K)
    fit = fit_plane_ransac(P[road])
    if fit is None:
        out["reason"] = "plane fit failed"; return out
    n, d, inl = fit
    fit_err = float(np.sqrt(((P[road][inl] @ n + d) ** 2).mean()))
    out["surface_fit_error_m"] = fit_err
    if fit_err > max_fit_err_m:
        out["reason"] = "road plane fit error too large"; return out

    m = mask & valid
    below = np.clip(-(P[m] @ n + d), 0, None)          # >0 means deeper than the road plane
    out["depth_p95_m"] = float(np.percentile(below, 95))

    # footprint area on the fitted road plane (independent of how sparse the depth is):
    # pixel area on a plane = z^2 / (fx*fy*|n.r|), r = (x/fx, y/fy, 1) is the pixel ray.
    vs, us = np.nonzero(mask)
    r = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones(len(us))], 1)
    ndr = r @ n
    ok = ndr < -1e-3
    z = -d / ndr[ok]
    if ok.sum() < 0.5 * len(us):
        out["reason"] = "defect rays do not intersect the road plane"; return out
    area = float((z ** 2 / (K[0, 0] * K[1, 1] * np.abs(ndr[ok]))).sum() * (len(us) / ok.sum()))
    out["area_m2"] = area
    out["volume_m3"] = float(below.mean() * area)      # approximate: mean depth x footprint
    # a depth that is not clearly larger than the road-plane fit noise cannot be told apart from road roughness / stereo noise
    if out["depth_p95_m"] < min_depth_snr * max(fit_err, 0.002):
        out["measurement_status"] = "uncertain"
        out["reason"] = "defect depth is within the road-plane fit noise"
        return out
    out["severity"] = severity_from_depth(out["depth_p95_m"])
    out["measurement_status"] = "ok"
    return out
