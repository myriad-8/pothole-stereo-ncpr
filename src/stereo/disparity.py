"""Disparity estimation (pretrained RAFT-Stereo, SGBM fallback) and disparity->depth."""
import numpy as np
import cv2

def disparity_to_depth(disp, fx_px, baseline_m, min_disp=0.5):
    """Z = f * B / d, with f in PIXELS, B in METRES, d in PIXELS. Invalid disparity -> depth 0."""
    z = np.zeros(disp.shape, dtype=np.float32)
    ok = np.isfinite(disp) & (disp > min_disp)
    z[ok] = fx_px * baseline_m / disp[ok]
    return z


def sgbm_disparity(left_rgb, right_rgb):
    """Classical OpenCV SGBM fallback. Inputs must be RECTIFIED. Returns disparity in pixels."""
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=256, blockSize=5, P1=8 * 25, P2=32 * 25,
                               uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return sg.compute(g(left_rgb), g(right_rgb)).astype(np.float32) / 16.0


def setup_raft_stereo(root="/kaggle/working/RAFT-Stereo"):
    """Clone the official RAFT-Stereo repo and download its pretrained weights with the authors' script (needs Internet)."""
    import subprocess
    from pathlib import Path
    root = Path(root)
    if not root.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/princeton-vl/RAFT-Stereo", str(root)], check=True)
    if not (root / "models" / "raftstereo-middlebury.pth").exists():
        subprocess.run("bash download_models.sh", shell=True, cwd=root, check=True)
    return root


def raft_stereo_disparity(left_rgb, right_rgb, device=None, scale=0.5, repo="/kaggle/working/RAFT-Stereo",
                          ckpt="raftstereo-middlebury.pth", iters=32):
    """Pretrained RAFT-Stereo (official princeton-vl code + weights), run as a subprocess at `scale` x resolution
    (full-resolution pairs exceed T4 memory). The disparity is rescaled back to full-resolution pixels."""
    import subprocess, sys, tempfile
    from pathlib import Path
    root = setup_raft_stereo(repo)
    H, W = left_rgb.shape[:2]; h, w = int(H * scale), int(W * scale)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, im in (("left", left_rgb), ("right", right_rgb)):
            small = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(td / f"{name}.png"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        cmd = [sys.executable, "demo.py", "--restore_ckpt", f"models/{ckpt}", "--corr_implementation", "alt", "--mixed_precision",
               "--valid_iters", str(iters), "-l", str(td / "left.png"), "-r", str(td / "right.png"),
               "--output_directory", str(td / "out"), "--save_numpy"]
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-600:])
        d = np.load(next((td / "out").glob("*.npy"))).astype(np.float32)
    if np.median(d) < 0:                          # the demo saves negative disparity; make it positive
        d = -d
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR) * (W / w)


def estimate_disparity(left_rgb, right_rgb, device=None, method="raft"):
    """Returns (disparity_px, method_name). Falls back to SGBM if the pretrained model cannot run."""
    if method == "raft":
        try:
            return raft_stereo_disparity(left_rgb, right_rgb, device), "RAFT-Stereo (official princeton-vl code + pretrained weights)"
        except Exception as e:
            print("RAFT-Stereo failed ->", repr(e)[-600:], "-> using SGBM fallback")
    return sgbm_disparity(left_rgb, right_rgb), "OpenCV SGBM (classical fallback)"
