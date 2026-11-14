"""Run: python demo/app.py --ckpt checkpoints/proposed_ce_s42.pth --model proposed"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import gradio as gr
from src.models import REGISTRY
from src.segmentation.infer import predict_prob_fullres
from src.stereo.disparity import estimate_disparity, disparity_to_depth
from src.geometry.surface import measure_defect

def make_demo(model, device, gr):
    """Gradio demo: image (segmentation only) or calibrated stereo pair / metric depth (measurements)."""
    def demo_fn(left, right, depth_file, fx, fy, cx, cy, baseline_m, thr):
        if left is None:
            return None, "Upload a road image."
        prob = predict_prob_fullres(model, left, device); mask = prob > thr
        ov = left.copy(); ov[mask] = (0.5 * ov[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        if mask.sum() == 0:
            return ov, "No pothole detected."
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        comps = [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= 200]
        lines = [f"{len(comps)} defect region(s). Mean segmentation score {prob[mask].mean():.2f} (NOT a measurement confidence)."]
        if not all([fx, fy, cx, cy]):
            return ov, "\n".join(lines + ["Measurement UNAVAILABLE: no calibration supplied. No severity assigned."])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        depth = None
        if right is not None and baseline_m:
            if right.shape != left.shape:
                return ov, "\n".join(lines + ["Measurement UNAVAILABLE: left and right images differ in size."])
            disp, method = estimate_disparity(left, right, device)
            depth = disparity_to_depth(disp, fx, baseline_m); lines.append("Depth source: " + method)
        elif depth_file is not None:
            depth = np.load(getattr(depth_file, "name", depth_file)); lines.append("Depth source: supplied .npy (metres)")
        if depth is None:
            return ov, "\n".join(lines + ["Measurement UNAVAILABLE: need a right image + baseline, or a metric depth map. No severity assigned."])
        if depth.shape != mask.shape:
            return ov, "\n".join(lines + [f"Measurement UNAVAILABLE: depth {depth.shape} does not match image {mask.shape}."])
        for j in comps:
            r = measure_defect(depth, lab == j, K)
            if r["measurement_status"] == "ok":
                lines.append(f"Defect {j}: depth_p95 {r['depth_p95_m']*100:.1f} cm | area {r['area_m2']:.3f} m2 | "
                             f"volume {r['volume_m3']*1000:.1f} L | severity {r['severity']} (assumed thresholds) | "
                             f"plane-fit error {r['surface_fit_error_m']*1000:.1f} mm")
            else:
                lines.append(f"Defect {j}: measurement UNAVAILABLE - {r['reason']}. No severity assigned.")
        return ov, "\n".join(lines)

    return gr.Interface(
        demo_fn,
        [gr.Image(type="numpy", label="Left / single road image"), gr.Image(type="numpy", label="Right image (optional, rectified)"),
         gr.File(label="Metric depth .npy in metres (optional)"),
         gr.Number(label="fx (px)"), gr.Number(label="fy (px)"), gr.Number(label="cx (px)"), gr.Number(label="cy (px)"),
         gr.Number(label="Baseline (m)"), gr.Slider(0.1, 0.9, 0.5, label="Mask threshold")],
        [gr.Image(label="Overlay"), gr.Textbox(label="Measurements / reasons", lines=8)],
        title="Pothole segmentation + geometric severity")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--model", default="proposed", choices=sorted(REGISTRY))
    ap.add_argument("--pretrained", default="nvidia/segformer-b0-finetuned-ade-512-512"); ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = REGISTRY[a.model](a.pretrained).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
    make_demo(model, dev, gr).launch(share=a.share)
