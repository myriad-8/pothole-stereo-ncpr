"""Full-resolution inference for one image."""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from src.datasets.pothole600 import MEAN, STD

@torch.no_grad()
def predict_prob_fullres(model, img_rgb, device, size=512):
    """uint8 RGB image of any size -> pothole probability map at the original resolution."""
    model.eval(); h, w = img_rgb.shape[:2]
    x = (cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()[None].to(device)
    lg = F.interpolate(model(x), size=(h, w), mode="bilinear", align_corners=False)
    return F.softmax(lg, 1)[0, 1].cpu().numpy()
