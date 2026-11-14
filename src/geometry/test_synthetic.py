"""Unit test: synthetic flat road with a 4 cm pothole and sparse noisy depth. Run: python -m src.geometry.test_synthetic"""
import numpy as np
from src.geometry.surface import measure_defect

H, W = 1080, 1920
K = np.array([[1500., 0, 960.], [0, 1500., 540.], [0, 0, 1]]); cam_h = 1.5
u, v = np.meshgrid(np.arange(W), np.arange(H)); ry = (v - K[1, 2]) / K[1, 1]
depth = np.zeros((H, W)); ok = ry > 0.03; depth[ok] = cam_h / ry[ok]
mask = np.zeros((H, W), bool); mask[700:760, 900:1050] = True; depth[mask] = (cam_h + 0.04) / ry[mask]
rng = np.random.default_rng(0); sparse = rng.random((H, W)) < 0.15
d_sp = np.where(sparse, depth + rng.normal(0, 0.002, (H, W)), 0)
r = measure_defect(d_sp, mask, K)
to_plane = lambda u_, v_: ((u_ - K[0, 2]) / K[0, 0] * cam_h / ((v_ - K[1, 2]) / K[1, 1]), cam_h / ((v_ - K[1, 2]) / K[1, 1]))
c = [to_plane(900, 700), to_plane(1050, 700), to_plane(1050, 760), to_plane(900, 760)]
true_area = 0.5 * abs(sum(c[i][0] * c[(i + 1) % 4][1] - c[(i + 1) % 4][0] * c[i][1] for i in range(4)))
assert r["measurement_status"] == "ok" and abs(r["depth_p95_m"] - 0.04) < 0.006, r
assert abs(r["area_m2"] / true_area - 1) < 0.05, (r["area_m2"], true_area)
d2 = d_sp.copy(); d2[mask] = 0; r2 = measure_defect(d2, mask, K)
assert r2["severity"] is None and r2["measurement_status"] == "unavailable", r2
print("PASS", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})
