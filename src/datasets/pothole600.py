"""Pothole-600 loader (RGB + binary mask)."""
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IGNORE_INDEX = 255          # void pixels: excluded from the loss and from all metrics


class Pothole600(Dataset):
    """RGB image + binary pothole mask (RGB-only, as in the project brief).
    Masks are resized with nearest-neighbour interpolation.
    Training augmentation (moderate): horizontal flip, random 80-100% crop, brightness/contrast jitter."""

    def __init__(self, root, split, size=512, train=False, void_value=None):
        self.d, self.size, self.train, self.void_value = Path(root) / split, size, train, void_value
        self.names = sorted(p.name for p in (self.d / "rgb").glob("*.png"))
        for n in self.names:
            assert (self.d / "label" / n).exists(), f"missing mask for {n}"

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        rgb = Image.open(self.d / "rgb" / n).convert("RGB")
        m = Image.open(self.d / "label" / n).convert("L")
        if self.train:
            if random.random() < 0.5:
                rgb, m = ImageOps.mirror(rgb), ImageOps.mirror(m)
            if random.random() < 0.5:
                w, h = rgb.size
                s = random.uniform(0.8, 1.0); cw, ch = int(w * s), int(h * s)
                x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
                box = (x0, y0, x0 + cw, y0 + ch)
                rgb, m = rgb.crop(box), m.crop(box)
            if random.random() < 0.8:
                rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.8, 1.2))
                rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.8, 1.2))
        s2 = (self.size, self.size)
        rgb = np.asarray(rgb.resize(s2, Image.BILINEAR), np.float32) / 255.0
        m = np.asarray(m.resize(s2, Image.NEAREST))
        y = (m > 0).astype(np.int64)
        if self.void_value is not None:
            y[m == self.void_value] = IGNORE_INDEX
        x = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy()).float()
        return {"rgb": x, "mask": torch.from_numpy(y), "name": n}
